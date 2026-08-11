"""P3 - IoVFD: Flower server (chieu chung cat LOCAL ensemble -> GLOBAL, data-free).

Bai bao: Cao, Di, Jin, FGCS 183 (2026).

Moi round, sau khi nhan trong so tu cac client, server lam 3 viec:

  1. Lay nhan theo TAN SUAT NHAN cua client (Eq. 9), gap nhieu hon loan vao
     nhan (Eq. 10-12) de duoc nhan mem y*.
  2. Huan luyen GENERATOR (Eq. 20):
        L_G = KL(sigma(Output_LM), y*)        # Eq. 18, khop nhan MEM
            + L_dis                            # Eq. 19, da dang theo nhan
  3. CHUNG CAT ensemble -> global (Eq. 26):
        L_GM = CE(ST(1)GM, y) + lambda_1 * KL(ST(t)GM, ST(t)LM)
     tuc co CA so hang hoc nhan that, khong chi mot minh KL.

KHONG co FedAvg: Algorithm 1 cua bai khong he trung binh tham so — global
duoc tao HOAN TOAN bang chung cat. Bat lai bang --fedavg-init de doi chung.

KHONG co so hang doi khang: Eq. 20 chi co hai so hang. --beta-adv (mac dinh
0) la phan them cua ho DFAD/DENSE, khong phai cua bai nay.

Chay:
  python server_iov.py --rounds 30 --num-clients 10
  python server_iov.py --rounds 30 --kd-steps 0     # tat KD -> thanh FedAvg
  python server_iov.py --mode test --ckpt out/checkpoints/latest.pth
"""
import argparse
import copy
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import (FitRes, Parameters, Scalar, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402
from model_cnn1d import CNN1D_IDS, INPUT_LEN, NUM_GLOBAL_CLASSES  # noqa: E402
from generator import (ChaoticNoise, Generator,                 # noqa: E402
                       diversity_loss, ensemble_logits, kl_divergence,
                       kl_to_soft_label, noisy_onehot, sample_labels,
                       student_ce)

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


class IoVFDStrategy(fl.server.strategy.FedAvg):
    """FedAvg + chung cat data-free tu ensemble client vao global."""

    def __init__(self, model, generator, noise, device, ckpt_dir: str,
                 start_round: int = 0, kd_steps: int = 30, gen_steps: int = 1,
                 stu_steps: int = 5, kd_batch: int = 128, lr_g: float = 1e-4,
                 lr_s: float = 1e-4, temperature: float = 3.0,
                 beta_div: float = 1.0, beta_adv: float = 0.0,
                 lambda1: float = 0.2, fedavg_init: bool = False,
                 weighted_ensemble: bool = False,
                 n_classes: int = NUM_GLOBAL_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.generator = generator
        self.noise = noise
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.start_round = start_round
        self.kd_steps = kd_steps
        self.gen_steps = gen_steps
        self.stu_steps = stu_steps
        self.kd_batch = kd_batch
        self.T = temperature
        self.beta_div = beta_div
        self.beta_adv = beta_adv
        self.weighted_ensemble = weighted_ensemble
        self.lambda1 = lambda1              # Eq. (26), bai dat 0.2
        self.fedavg_init = fedavg_init
        self.n_classes = n_classes
        self.label_dist = None              # Eq. (9), gop tu label_counts client
        self.opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.lr_s = lr_s
        self._template = copy.deepcopy(model).cpu()

    # ---- chung cat data-free ---------------------------------------------
    def _build_ensemble(self, all_ndarrays):
        models = []
        for nds in all_ndarrays:
            m = copy.deepcopy(self._template)
            m.load_state_dict(C.ndarrays_to_state_dict(m, nds))
            m.to(self.device).eval()
            for q in m.parameters():
                q.requires_grad_(False)
            models.append(m)
        return models

    def _lay_mau(self, b, dev):
        """Eq. (9) -> y ; Eq. (11,12) -> z ; Eq. (10) -> y*."""
        y = sample_labels(b, self.label_dist, dev)
        z = self.noise.gaussian(b).to(dev)
        return y, noisy_onehot(y, z, self.n_classes)

    def _distill(self, teachers, weights, rnd) -> Dict[str, float]:
        student = self.model.to(self.device)
        opt_s = optim.Adam(student.parameters(), lr=self.lr_s)
        g, dev, b = self.generator.to(self.device), self.device, self.kd_batch
        # Eq. (17)/(23) lay trung binh KHONG trong so 1/n. weights chi dung
        # khi nguoi dung co tinh bat --weighted-ensemble.
        w = weights if self.weighted_ensemble else None
        stats = {"gen_loss": 0.0, "stu_loss": 0.0, "stu_ce": 0.0, "n": 0}

        for _ in range(self.kd_steps):
            # --- (a) buoc generator: Eq. (20) L_G = L_kl_same + L_dis -------
            student.eval()
            g.train()
            for _ in range(self.gen_steps):
                y, y_star = self._lay_mau(b, dev)
                x = g(y_star)
                t_out = ensemble_logits(teachers, x, w)
                loss_g = (kl_to_soft_label(t_out, y_star)          # Eq. (18)
                          + self.beta_div * diversity_loss(x, y_star))  # Eq. (19)
                if self.beta_adv:      # KHONG co trong bai, mac dinh tat
                    loss_g = loss_g - self.beta_adv * kl_divergence(
                        student(x), t_out, self.T)
                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()
                stats["gen_loss"] += loss_g.item()

            # --- (b) buoc global: Eq. (26) L_GM = L_student + l1 * L_kd -----
            g.eval()
            student.train()
            for _ in range(self.stu_steps):
                with torch.no_grad():
                    y, y_star = self._lay_mau(b, dev)
                    x = g(y_star).detach()
                    t_out = ensemble_logits(teachers, x, w).detach()
                s_out = student(x)
                ce = student_ce(s_out, y)                          # Eq. (24)
                kd = kl_divergence(s_out, t_out, self.T)           # Eq. (25)
                loss_s = ce + self.lambda1 * kd                    # Eq. (26)
                opt_s.zero_grad()
                loss_s.backward()
                opt_s.step()
                stats["stu_loss"] += loss_s.item()
                stats["stu_ce"] += ce.item()
            stats["n"] += 1

        n = max(stats["n"], 1)
        out = {"gen_loss": stats["gen_loss"] / (n * max(self.gen_steps, 1)),
               "stu_loss": stats["stu_loss"] / (n * max(self.stu_steps, 1)),
               "stu_ce": stats["stu_ce"] / (n * max(self.stu_steps, 1))}
        logger.info(f"[Round {rnd}] data-free KD {self.kd_steps} vong: "
                    f"gen_loss={out['gen_loss']:.4f} "
                    f"stu_loss={out['stu_loss']:.4f} (CE {out['stu_ce']:.4f})")
        return out

    # ---- Flower API -------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is None:
            return None, metrics
        if failures:
            logger.warning(f"[Round {server_round}] {len(failures)} client loi")

        losses = [(r.num_examples, r.metrics.get("train_loss", 0.0)) for _, r in results]
        n_tot = sum(n for n, _ in losses) or 1
        metrics["train_loss"] = sum(n * l for n, l in losses) / n_tot
        metrics["kd_loss_client"] = float(np.mean(
            [r.metrics.get("kd_loss", 0.0) for _, r in results]))
        metrics["num_clients"] = len(results)
        # Kiem chung co che ca nhan hoa: log cua client khong ra khoi Ray actor,
        # nen so client thuc su TIEP TUC model cuc bo phai duoc bao qua metrics.
        # Tu round 2 tro di con so nay phai bang num_clients; neu bang 0 nghia
        # la trang thai cuc bo khong song sot va co che loi cua bai da hong.
        tiep = sum(int(r.metrics.get("resumed_local", 0)) for _, r in results)
        metrics["resumed_local"] = tiep
        if server_round > 1 and tiep < len(results):
            logger.warning(f"[Round {server_round}] CANH BAO: chi {tiep}/"
                           f"{len(results)} client tiep duoc model cuc bo — "
                           f"co che ca nhan hoa cua bai dang hong")
        elif tiep:
            logger.info(f"[Round {server_round}] {tiep}/{len(results)} client "
                        f"tiep tuc model cuc bo (Algorithm 1 dong 10)")

        # Eq. (9): p(y) tu TAN SUAT NHAN cua du lieu client. Client da gui san
        # label_counts; truoc day server nhan roi vut di, generator van lay
        # nhan bang randint DEU -> sai phan bo, va sai han voi bo CICIoV lech
        # lop nang.
        tong = np.zeros(self.n_classes, dtype=np.float64)
        for _, r in results:
            lc = r.metrics.get("label_counts")
            if lc:
                v = np.fromstring(lc, sep=",")
                tong[:len(v)] += v[:self.n_classes]
        if tong.sum() > 0:
            self.label_dist = tong / tong.sum()
            if server_round == 1:
                logger.info("Eq.9 p(y) tu client: "
                            + " ".join(f"{v:.3f}" for v in self.label_dist))

        # Algorithm 1 KHONG co buoc trung binh tham so: global chi duoc tao
        # bang chung cat. --fedavg-init bat lai de doi chung.
        if self.fedavg_init:
            self.model.load_state_dict(
                C.ndarrays_to_state_dict(self.model, parameters_to_ndarrays(params)))

        if self.kd_steps > 0 and len(results) >= 2:
            teachers = self._build_ensemble(
                [parameters_to_ndarrays(r.parameters) for _, r in results])
            weights = [r.num_examples for _, r in results]
            metrics.update(self._distill(teachers, weights, server_round))
            del teachers
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            params = ndarrays_to_parameters(C.get_model_parameters(self.model))

        abs_round = self.start_round + server_round
        C.save_checkpoint(self.ckpt_dir, abs_round, self.model.state_dict(),
                          extra={"train_loss": metrics.get("train_loss"),
                                 "generator_state_dict": self.generator.state_dict()})
        return params, metrics


# ----------------------------------------------------------------------------
def make_evaluate_fn(model, loader, criterion, device, csv_file, out_dir,
                     class_names, total_rounds, start_round, task, cm_every=0):
    def evaluate_fn(server_round: int, parameters, config):
        if server_round == 0:
            return None
        abs_round = start_round + server_round
        model.load_state_dict(C.ndarrays_to_state_dict(model, parameters))
        model.to(device)
        m, y_true, y_pred = C.evaluate(model, loader, criterion, device)
        C.log_and_save_metrics(abs_round, m, csv_file)
        # Ghi confusion matrix o cuoi task, VA dinh ky neu bat --cm-every,
        # de bi cat giua chung van con ban gan nhat.
        if server_round == total_rounds or (cm_every and abs_round % cm_every == 0):
            tag = f"task{task}" if task is not None else "final"
            C.save_confusion_matrix(y_true, y_pred, out_dir, tag, class_names)
        return m["loss"], {k: v for k, v in m.items() if k != "loss"}
    return evaluate_fn


def build_generator(args, device):
    """Eq. (13)-(16) cua bai.

    Ban MLP cu (z 100 chieu + Embedding(y) -> MLP) DA BO HAN: duong ong cua
    bai chi sinh ra nhan mem y*, khong co vector nhieu rieng de dua vao no.
    Giu lai chi tao ra mot co --gen-arch mlp chac chan sap khi ai do bat
    (da do: TypeError, MLPGenerator.forward() thieu doi so 'y').
    """
    return Generator(NUM_GLOBAL_CLASSES, INPUT_LEN, args.gen_seq_len,
                     args.gen_channels, args.gen_heads, args.gen_blocks).to(device)


def fit_config_fn(local_epochs, lr, lambda_kd):
    def fn(server_round: int) -> Dict[str, Scalar]:
        return {"server_round": server_round, "local_epochs": local_epochs,
                "lr": lr, "lambda_kd": lambda_kd}
    return fn


def run_test(args, model, device):
    ckpt = args.ckpt or os.path.join(args.out_dir, "checkpoints", "latest.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt}")
    rnd, _ = C.load_checkpoint(ckpt, model)
    model.to(device)
    logger.info(f"Nap checkpoint {ckpt} (round {rnd})")
    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    m, y_true, y_pred = C.evaluate(model, loader, nn.CrossEntropyLoss(), device)
    logger.info(C.format_metrics(rnd, m))
    C.append_csv_row(os.path.join(args.out_dir, "test_metrics.csv"),
                     [rnd] + [round(m[k], 6) for k in C.METRIC_KEYS])
    tag = f"test_task{args.task}" if args.task is not None else "test"
    C.save_confusion_matrix(y_true, y_pred, args.out_dir, tag,
                            C.load_class_names(args.data_dir))


def main():
    p = argparse.ArgumentParser(description="P3 IoVFD Flower server (dual KD)")
    p.add_argument("--mode", choices=["train", "resume", "test"], default="train")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Bai dat 0.0001")
    p.add_argument("--lambda-kd", type=float, default=0.2,
                   help="lambda_2 cua Eq.(29) o client. Bai dat 0.2")
    p.add_argument("--lambda1", type=float, default=0.2,
                   help="lambda_1 cua Eq.(26) o global. Bai dat 0.2")
    p.add_argument("--fedavg-init", action="store_true",
                   help="Trung binh tham so truoc khi chung cat. Algorithm 1 "
                        "cua bai KHONG lam vay — chi de doi chung")
    p.add_argument("--weighted-ensemble", action="store_true",
                   help="Trung binh ensemble theo so mau. Eq.(17) la 1/n "
                        "KHONG trong so — chi de doi chung")
    p.add_argument("--dropout", type=float, default=0.15)
    # --- data-free KD ---
    p.add_argument("--kd-steps", type=int, default=30, help="0 = tat KD (FedAvg)")
    p.add_argument("--gen-steps", type=int, default=1)
    p.add_argument("--stu-steps", type=int, default=5)
    p.add_argument("--kd-batch", type=int, default=128)
    p.add_argument("--noise-dim", type=int, default=100)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-s", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--beta-div", type=float, default=1.0)
    p.add_argument("--beta-adv", type=float, default=0.0,
                   help="So hang doi khang. Eq.(20) chi co HAI so hang, khong "
                        "co cai nay — mac dinh tat")
    p.add_argument("--gen-seq-len", type=int, default=8)
    p.add_argument("--gen-channels", type=int, default=32)
    p.add_argument("--gen-heads", type=int, default=4)
    p.add_argument("--gen-blocks", type=int, default=3,
                   help="Bai noi 'three stacked ... blocks'")
    p.add_argument("--noise-mode", choices=["paper", "boxmuller"], default="paper",
                   help="paper = Eq.(12) nguyen van (z in [0.242, 0.399], "
                        "KHONG phai Gauss chuan). boxmuller = Gauss that")
    p.add_argument("--chaos-u0", type=float, default=0.01,
                   help="Gia tri khoi tao he logistic. Bai dat 0.01")
    p.add_argument("--chaos-r", type=float, default=4.0,
                   help="Tham so anh xa logistic (>3.57 la hon loan)")
    # --- chung ---
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--address", type=str, default="0.0.0.0:8083")
    p.add_argument("--test-samples", type=int, default=1_000_000)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--cm-every", type=int, default=0,
                   help="Ghi confusion matrix moi N round (0 = chi cuoi task)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    C.setup_logging(os.path.join(args.out_dir, "server.log"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiet bi: {device} | che do: {args.mode} | "
                f"kd_steps={args.kd_steps} | task: {args.task}")

    model = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, args.dropout).to(device)
    generator = build_generator(args, device)
    noise = ChaoticNoise(NUM_GLOBAL_CLASSES, r=args.chaos_r, u0=args.chaos_u0,
                         device=str(device), mode=args.noise_mode)

    if args.mode == "test":
        run_test(args, model, device)
        return

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    start_round = 0
    if args.mode == "resume":
        ckpt = args.ckpt or os.path.join(ckpt_dir, "latest.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Khong co checkpoint de resume: {ckpt}")
        start_round, extra = C.load_checkpoint(ckpt, model)
        model.to(device)
        if "generator_state_dict" in extra:
            generator.load_state_dict(extra["generator_state_dict"])
            logger.info("Da nap lai trong so generator")
        logger.info(f"Resume tu round {start_round} ({ckpt})")

    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    class_names = C.load_class_names(args.data_dir)
    suffix = f"_task{args.task}" if args.task is not None else ""
    csv_file = os.path.join(args.out_dir, f"metrics{suffix}.csv")

    strategy = IoVFDStrategy(
        model=model, generator=generator, noise=noise, device=device,
        ckpt_dir=ckpt_dir, start_round=start_round,
        kd_steps=args.kd_steps, gen_steps=args.gen_steps, stu_steps=args.stu_steps,
        kd_batch=args.kd_batch, lr_g=args.lr_g, lr_s=args.lr_s,
        temperature=args.temperature, beta_div=args.beta_div, beta_adv=args.beta_adv,
        lambda1=args.lambda1, fedavg_init=args.fedavg_init,
        weighted_ensemble=args.weighted_ensemble,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=max(2, int(args.num_clients * args.fraction_fit)),
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        initial_parameters=ndarrays_to_parameters(C.get_model_parameters(model)),
        on_fit_config_fn=fit_config_fn(args.local_epochs, args.lr, args.lambda_kd),
        evaluate_fn=make_evaluate_fn(model, loader, nn.CrossEntropyLoss(), device,
                                     csv_file, args.out_dir, class_names,
                                     args.rounds, start_round, args.task, args.cm_every),
    )

    logger.info(f"Server lang nghe {args.address} | {args.rounds} round | CSV -> {csv_file}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    logger.info(f"Xong. Ket qua trong {args.out_dir}")


if __name__ == "__main__":
    main()
