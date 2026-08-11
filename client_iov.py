"""P3 - IoVFD: Flower client (chieu chung cat GLOBAL -> LOCAL).

Bai bao: Cao, Di, Jin, FGCS 183 (2026).

"Dual knowledge distillation" gom hai chieu:
  - Tai CLIENT (file nay): model global lam GIAO VIEN cho model local.
        Eq. (29):  L_LM = L_student + lambda_2 * L_kd
        Eq. (27):  L_student = CE(ST(1)LM, y)          <- nhan THAT
        Eq. (28):  L_kd      = KL(ST(t)LM, ST(t)GM)
    Bai dat lambda_2 = 0.2.
    Model local KHONG bi ghi de boi trong so global -> giu tinh ca nhan hoa,
    day la diem khac FedAvg. Global chi vao qua so hang KL. Dau ra cua
    Algorithm 1 la CHINH cac model local nay, khong phai global.

TRANG THAI CUC BO PHAI GHI RA DIA
---------------------------------
Trong che do simulation cua Flower, doi tuong client bi TAO LAI moi round nen
self.model khong song sot — model "ca nhan hoa" se bi dung lai tu dau moi
round, tuc co che loi cua bai bi hong AM THAM. Truoc day run_sim.py chan han
P3 bang sys.exit() vi ly do do. Nay trang thai duoc luu vao --state-dir nen
chay simulation duoc, va co kiem chung trong smoke_test.
  - Tai SERVER (server_iov.py): ensemble cac model local lam giao vien cho
    global, chung cat tren pseudo-data do generator sinh (data-free).

Chay:
  python client_iov.py --client-id 0
  python client_iov.py --client-id 0 --lambda-kd 1.0 --task 0
"""
import argparse
import logging
import os
import sys

import flwr as fl
import numpy as np
import torch
import torch.optim as optim

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:   # repo doc lap: khong co thu muc nay
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402
from model_cnn1d import CNN1D_IDS, FocalLoss, INPUT_LEN, NUM_GLOBAL_CLASSES  # noqa: E402
from generator import kl_divergence              # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"


class IoVFDClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dir, device, max_samples, batch_size,
                 task, lr, dropout, lambda_kd, temperature, personalized,
                 state_dir=None, pretrain_epochs=1):
        self.cid = client_id
        self.device = device
        self.lr = lr
        self.lambda_kd = lambda_kd
        self.T = temperature
        self.personalized = personalized
        self.state_dir = state_dir
        self.pretrain_epochs = pretrain_epochs
        self.task = task
        self.da_nap_cuc_bo = False

        x, y = C.load_client_data(data_dir, client_id, task, max_samples)
        self.loader = C.make_loader(x, y, batch_size, shuffle=True)
        self.n_samples = len(y)
        self.label_counts = np.bincount(y, minlength=NUM_GLOBAL_CLASSES).tolist()

        # model local (ben vung qua cac round) + ban sao global lam giao vien
        self.model = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, dropout).to(device)
        self.teacher = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, dropout).to(device)
        self.criterion = FocalLoss(alpha=C.make_focal_alpha(y).to(device), gamma=2.0)

    # ---- trang thai cuc bo (song sot qua viec Ray tao lai client) ---------
    def _state_path(self):
        ten = "flat" if self.task is None else f"task{self.task}"
        return os.path.join(self.state_dir, f"client_{self.cid}_{ten}.npz")

    def _load_local(self):
        if not self.state_dir or not os.path.exists(self._state_path()):
            return None
        try:
            with np.load(self._state_path()) as z:
                return [z[f"a{i}"] for i in range(len(z.files))]
        except Exception as e:
            logger.warning(f"[Client {self.cid}] khong doc duoc trang thai: {e}")
            return None

    def _save_local(self):
        if not self.state_dir:
            return
        os.makedirs(self.state_dir, exist_ok=True)
        np.savez(self._state_path(),
                 **{f"a{i}": a for i, a in
                    enumerate(C.get_model_parameters(self.model))})

    def _pretrain(self):
        """Algorithm 1 dong 2: local model duoc PRETRAIN truoc vong FL.

        Chi CE tren du lieu cuc bo, chua co giao vien nao.
        """
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.pretrain_epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                opt.zero_grad()
                self.criterion(self.model(xb), yb).backward()
                opt.step()
        logger.info(f"[Client {self.cid}] pretrain {self.pretrain_epochs} epoch "
                    f"(Algorithm 1 dong 2)")

    # ---- Flower API -------------------------------------------------------
    def get_parameters(self, config):
        return C.get_model_parameters(self.model)

    def fit(self, parameters, config):
        epochs = int(config.get("local_epochs", 1))
        rnd = int(config.get("server_round", 0))
        lr = float(config.get("lr", self.lr))
        lam = float(config.get("lambda_kd", self.lambda_kd))

        # global -> giao vien (dong bang)
        self.teacher.load_state_dict(C.ndarrays_to_state_dict(self.teacher, parameters))
        self.teacher.eval()
        for q in self.teacher.parameters():
            q.requires_grad_(False)

        # Model local phai LIEN TUC qua cac round. Uu tien nap tu dia; chi khi
        # chua co gi (round dau) moi khoi tao tu global.
        # Ba truong hop, theo thu tu uu tien:
        #  1. Co trang thai tren dia  -> nap (che do simulation, doi tuong moi)
        #  2. Round dau / tat ca nhan hoa -> khoi tao tu global
        #  3. Con lai -> GIU NGUYEN self.model dang co trong bo nho
        # Truong hop 3 la duong gRPC khi khong dat --state-dir: doi tuong
        # client van song nen model cuc bo nam san trong bo nho. Bo qua no se
        # lam global ghi de model local MOI ROUND, tuc hong dung co che cua
        # bai theo huong nguoc lai.
        truoc = self._load_local() if self.personalized else None
        if truoc is not None:
            self.model.load_state_dict(C.ndarrays_to_state_dict(self.model, truoc))
            self.da_nap_cuc_bo = True
        elif rnd <= 1 or not self.personalized:
            self.model.load_state_dict(
                C.ndarrays_to_state_dict(self.model, parameters))
            self.da_nap_cuc_bo = False
            if self.pretrain_epochs > 0 and self.personalized:
                self._pretrain()
        else:
            self.da_nap_cuc_bo = True          # tiep model dang co trong bo nho

        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=lr)
        sum_ce, sum_kd, n_batches = 0.0, 0.0, 0
        for _ in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                opt.zero_grad()
                out = self.model(xb)
                ce = self.criterion(out, yb)
                with torch.no_grad():
                    t_out = self.teacher(xb)
                kd = kl_divergence(out, t_out, self.T)
                (ce + lam * kd).backward()
                opt.step()
                sum_ce += ce.item()
                sum_kd += kd.item()
                n_batches += 1

        nb = max(n_batches, 1)
        self._save_local()
        logger.info(f"[Client {self.cid}][Round {rnd}] n={self.n_samples} "
                    f"ce={sum_ce / nb:.4f} kd={sum_kd / nb:.4f} (lambda={lam})"
                    + (" | tiep model cuc bo" if self.da_nap_cuc_bo else ""))
        return (C.get_model_parameters(self.model), self.n_samples,
                {"train_loss": sum_ce / nb, "kd_loss": sum_kd / nb,
                 "resumed_local": int(self.da_nap_cuc_bo),
                 "label_counts": ",".join(map(str, self.label_counts))})

    def evaluate(self, parameters, config):
        return 0.0, self.n_samples, {}


def main():
    p = argparse.ArgumentParser(description="P3 IoVFD Flower client")
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--server", type=str, default="127.0.0.1:8083")
    p.add_argument("--max-samples", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Bai bao dung batch=32")
    p.add_argument("--lr", type=float, default=1e-4, help="Bai dat 0.0001")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lambda-kd", type=float, default=0.2,
                   help="lambda_2 cua Eq.(29). Bai dat 0.2")
    p.add_argument("--state-dir", type=str, default=None,
                   help="Noi luu model cuc bo. BAT BUOC o che do simulation")
    p.add_argument("--pretrain-epochs", type=int, default=1,
                   help="Algorithm 1 dong 2: pretrain local truoc vong FL")
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--no-personalized", action="store_true",
                   help="Ghi de local bang global moi round (thanh FedAvg + KD)")
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS))
    args = p.parse_args()

    C.setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = IoVFDClient(args.client_id, args.data_dir, device, args.max_samples,
                         args.batch_size, args.task, args.lr, args.dropout,
                         args.lambda_kd, args.temperature,
                         personalized=not args.no_personalized,
                         state_dir=args.state_dir,
                         pretrain_epochs=args.pretrain_epochs)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
