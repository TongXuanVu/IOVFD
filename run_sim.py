"""Chay DU 100 client bang che do simulation CUA FLOWER (flwr.simulation).

TAI SAO
-------
`run_fl.py` sinh moi client thanh mot tien trinh rieng noi voi server qua gRPC.
Moi tien trinh nap mot ban torch (~300MB RSS) nen 100 client can khoang 30GB —
Kaggle T4 chi co 29GB, OOM truoc khi train duoc gi.

Flower co san che do simulation: cac client la "actor" Ray dung chung mot pool
tien trinh, nen bo nho phu thuoc SO ACTOR CHAY SONG SONG chu khong phai so
client. Do thuc te: 100 client, dinh 2.06GB.

QUAN TRONG: file nay KHONG cai lai thuat toan. No dung lai y nguyen
`VanFedStrategy` va `make_evaluate_fn` trong server_iov.py, va lop client trong
client_iov.py. Cung strategy, cung phep tong hop, cung dinh dang dau ra
(metrics CSV, checkpoint, confusion matrix) — nen `collect_results.py` dung duoc
khong can sua, va ket qua so sanh duoc voi run_fl.py.

  run_fl.py   : Flower that (gRPC, moi client mot tien trinh). It client.
  run_sim.py  : Flower simulation (Ray). Du 100 client.

Client khong co shard cua task nao thi khong nam trong danh sach cua task do —
khong train, khong duoc tong hop. Quay lai o task sau thi nhan trong so global
moi nhat, vi moi round client deu nap global truoc khi train.

Cai dat: pip install "flwr[simulation]"

Chay:
  python run_sim.py --data-dir <DATA> --clients 100 --rounds 30
  python run_sim.py --data-dir <DATA> --clients 100 --rounds 30 --max-samples 0
"""
import argparse
import glob
import logging
import os
import sys
import time

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import ndarrays_to_parameters

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:
    sys.path.insert(0, _P1)

import common as C                                        # noqa: E402
import server_iov as S                                    # noqa: E402  (dung lai strategy)

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_P4 = os.path.exists(os.path.join(ROOT, "models_sdn.py"))     # SDN-FL IDS
IS_P2 = os.path.exists(os.path.join(ROOT, "model_kanconv.py"))  # FedIoV
IS_P3 = os.path.exists(os.path.join(ROOT, "generator.py"))      # IoVFD
logger = logging.getLogger(__name__)

if IS_P3:
    sys.exit(
        "run_sim.py KHONG dung duoc cho IoVFD (P3).\n\n"
        "Trong che do simulation cua Flower, doi tuong client bi TAO MOI moi\n"
        "round — trang thai cuc bo khong song sot (da do bang thuc nghiem).\n"
        "Client cua P3 giu model local CA NHAN HOA qua cac round, va chi nap\n"
        "global khi rnd <= 1; neu chay simulation thi tu round 2 tro di no se\n"
        "train tren model khoi tao ngau nhien -> hong dung co che loi cua bai,\n"
        "ma hong AM THAM.\n\n"
        "Dung run_fl.py cho P3 (moi client mot tien trinh nen model local ben\n"
        "vung), hoac sua client de luu/nap trang thai ra dia theo client id.")

if IS_P2:
    from client_iov import FedIoVClient as ClientCls           # noqa: E402
    from model_kanconv import KANConvNet, NUM_GLOBAL_CLASSES, INPUT_LEN  # noqa: E402

    def build_model(arch, num_classes, dropout, hidden=64, layers=2, **kw):
        return KANConvNet(INPUT_LEN, num_classes, dropout,
                          kw.get("width", (16, 32)), kw.get("grid_size", 5),
                          kw.get("spline_order", 3))
elif IS_P4:
    from client_iov import SDNControllerClient as ClientCls   # noqa: E402
    from models_sdn import build_model, NUM_GLOBAL_CLASSES    # noqa: E402
else:
    from client_iov import VanFedClient as ClientCls          # noqa: E402
    from model_cnn1d import (CNN1D_IDS, INPUT_LEN,            # noqa: E402
                             NUM_GLOBAL_CLASSES)

    def build_model(arch, num_classes, dropout, hidden=64, layers=2):
        return CNN1D_IDS(INPUT_LEN, num_classes, dropout)


def clients_with_data(data_dir, client_ids, task):
    fed = os.path.join(data_dir, "federated_data")
    ok = []
    for cid in client_ids:
        if task is None:
            has = (any(os.path.exists(os.path.join(fed, f"client_{cid}_task_{t}.pt"))
                       for t in range(1, C.NUM_TASKS + 1))
                   or os.path.exists(os.path.join(fed, f"client_{cid}.pt")))
        else:
            has = os.path.exists(os.path.join(fed, f"client_{cid}_task_{task + 1}.pt"))
        if has:
            ok.append(cid)
    return ok


def rounds_done(out_dir, task):
    if task is None:
        cands = [p for p in glob.glob(os.path.join(out_dir, "metrics*.csv"))
                 if "task" not in os.path.basename(p)
                 and not os.path.basename(p).startswith("test_")]
    else:
        cands = glob.glob(os.path.join(out_dir, f"metrics*task{task}.csv"))
    n = 0
    for p in cands:
        try:
            with open(p, encoding="utf-8") as f:
                n = max(n, sum(1 for _ in f) - 1)
        except OSError:
            pass
    return max(n, 0)


def make_client_fn(ids, args, task, device):
    """partition-id cua Ray -> client id that -> doi tuong client cua repo."""
    def client_fn(ctx):
        pid = int(ctx.node_config.get("partition-id", 0))
        cid = ids[pid % len(ids)]
        if IS_P2:
            atk = args.attackers.get(cid, "none")
            c = ClientCls(cid, args.data_dir, device, args.max_samples,
                          args.batch_size, task, args.lr, args.dropout,
                          tuple(args.width), args.grid_size, args.spline_order,
                          atk, args.attack_scale, args.seed)
        elif IS_P4:
            c = ClientCls(cid, args.data_dir, device, args.max_samples,
                          args.batch_size, task, args.lr, args.dropout,
                          args.arch, args.hidden, args.layers,
                          args.throughput, args.latency, args.node_trust,
                          args.simulate_sdn, args.jitter, args.seed)
        else:
            c = ClientCls(cid, args.data_dir, device, args.max_samples,
                          args.batch_size, task, args.lr, args.dropout)
        return c.to_client()
    return client_fn


def main():
    p = argparse.ArgumentParser(
        description="Chay du 100 client bang Flower simulation")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="out_sim")
    p.add_argument("--clients", type=int, default=100)
    p.add_argument("--client-ids", type=int, nargs="+", default=None)
    p.add_argument("--rounds", type=int, default=30, help="Round MOI task")
    p.add_argument("--tasks", type=str, default="all", help="'all' | 'none' | '0,1,2'")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--max-samples", type=int, default=500_000,
                   help="Moi client. 0 = dung HET du lieu")
    p.add_argument("--test-samples", type=int, default=1_000_000)
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--cm-every", type=int, default=0)
    p.add_argument("--restart", action="store_true")
    p.add_argument("--actor-cpus", type=float, default=1.0,
                   help="CPU cho MOI client song song. Tang len de giam so client "
                        "chay dong thoi neu thieu RAM")
    p.add_argument("--actor-gpus", type=float, default=0.0,
                   help="Ty le GPU moi client, vd 0.1 = toi da 10 client/GPU")
    # --- rieng P2 (FedIoV) ---
    p.add_argument("--width", type=int, nargs=2, default=[16, 32])
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--krum-m", type=int, default=5)
    p.add_argument("--byzantine", type=int, default=2)
    p.add_argument("--strategy", choices=["multikrum", "fedavg"], default="multikrum")
    p.add_argument("--attack-ids", type=int, nargs="*", default=[],
                   help="Client id bi bien thanh doc hai, de kiem chung Multi-Krum")
    p.add_argument("--attack", choices=["none", "signflip", "gauss", "label"],
                   default="signflip", help="Kieu tan cong cho --attack-ids")
    p.add_argument("--attack-scale", type=float, default=5.0)
    # --- rieng P4 ---
    p.add_argument("--arch", choices=["cnn", "rnn"], default="cnn")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--weighting", choices=["trust", "state", "samples"], default="trust")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--trust-ema", type=float, default=0.7)
    p.add_argument("--throughput", type=float, default=50.0)
    p.add_argument("--latency", type=float, default=50.0)
    p.add_argument("--node-trust", type=float, default=1.0)
    p.add_argument("--simulate-sdn", action="store_true")
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sfx_arch = f"_{args.arch}" if IS_P4 else ""
    C.setup_logging(os.path.join(args.out_dir, f"sim{sfx_arch}.log"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tasks.strip().lower() == "none":
        tasks = [None]
    elif args.tasks.strip().lower() == "all":
        tasks = list(range(C.NUM_TASKS))
    else:
        tasks = [int(t) for t in args.tasks.replace(" ", "").split(",")]

    pool = args.client_ids if args.client_ids else list(range(args.clients))
    per_task = {t: clients_with_data(args.data_dir, pool, t) for t in tasks}

    logger.info(f"Thiet bi: {device} | pool {len(pool)} client | "
                f"{args.rounds} round/task | Flower simulation (Ray)")
    logger.info("Client co du lieu tung task: "
                + ", ".join(f"task{t}={len(per_task[t])}" for t in tasks))

    args.attackers = {c: args.attack for c in args.attack_ids}
    if IS_P2:
        model = build_model(args.arch, NUM_GLOBAL_CLASSES, args.dropout,
                            width=tuple(args.width), grid_size=args.grid_size,
                            spline_order=args.spline_order).to(device)
    else:
        model = build_model(args.arch, NUM_GLOBAL_CLASSES, args.dropout,
                            args.hidden, args.layers).to(device)
    ckpt_dir = os.path.join(args.out_dir, f"checkpoints{sfx_arch}")

    progress = {t: 0 for t in tasks} if args.restart else \
               {t: rounds_done(args.out_dir, t) for t in tasks}
    start_round = sum(min(progress[t], args.rounds) for t in tasks)
    latest = os.path.join(ckpt_dir, "latest.pth")
    if not args.restart and os.path.exists(latest):
        r, _ = C.load_checkpoint(latest, model)
        model.to(device)
        start_round = max(start_round, r)
        logger.info(f"Chay tiep tu checkpoint round {r}")
    if any(progress.values()):
        logger.info("Tiep tuc: " + ", ".join(f"task{t}={progress[t]}/{args.rounds}"
                                             for t in tasks if progress[t]))

    class_names = C.load_class_names(args.data_dir)
    t0 = time.time()

    for task in tasks:
        done = progress.get(task, 0)
        if done >= args.rounds:
            logger.info(f"[task {task}] da xong {done}/{args.rounds} round — bo qua")
            continue
        ids = per_task[task]
        if not ids:
            logger.warning(f"[task {task}] khong client nao co du lieu — bo qua")
            continue
        remaining = args.rounds - done

        loader, _ = C.load_global_test(args.data_dir, args.test_samples, task)
        sfx = f"_task{task}" if task is not None else ""
        csv_file = os.path.join(args.out_dir, f"metrics{sfx_arch}{sfx}.csv")

        common_kw = dict(
            model=model, ckpt_dir=ckpt_dir, start_round=start_round,
            fraction_fit=args.fraction_fit, fraction_evaluate=0.0,
            min_fit_clients=max(1, int(len(ids) * args.fraction_fit)),
            min_evaluate_clients=0, min_available_clients=len(ids),
            initial_parameters=ndarrays_to_parameters(C.get_model_parameters(model)),
            on_fit_config_fn=S.fit_config_fn(args.local_epochs, args.lr),
        )
        if IS_P2:
            ev = S.make_evaluate_fn(model, loader, nn.CrossEntropyLoss(), device,
                                    csv_file, args.out_dir, class_names, remaining,
                                    start_round, task, args.cm_every)
            strategy = S.MultiKrumStrategy(
                krum_m=args.krum_m, n_byzantine=args.byzantine,
                use_krum=(args.strategy == "multikrum"), evaluate_fn=ev, **common_kw)
        elif IS_P4:
            ev = S.make_evaluate_fn(model, loader, nn.CrossEntropyLoss(), device,
                                    csv_file, args.out_dir, class_names, remaining,
                                    start_round, task, args.arch, args.cm_every)
            strategy = S.TrustWeightedFedAvg(
                weighting=args.weighting, alpha=args.alpha, beta=args.beta,
                gamma=args.gamma, trust_ema=args.trust_ema,
                weight_log=os.path.join(args.out_dir,
                                        f"client_weights{sfx_arch}{sfx}.csv"),
                evaluate_fn=ev, **common_kw)
        else:
            ev = S.make_evaluate_fn(model, loader, nn.CrossEntropyLoss(), device,
                                    csv_file, args.out_dir, class_names, remaining,
                                    start_round, task, args.cm_every)
            strategy = S.VanFedStrategy(evaluate_fn=ev, **common_kw)

        logger.info(f"===== Task {task}: {len(ids)} client, {remaining} round =====")
        fl.simulation.start_simulation(
            client_fn=make_client_fn(ids, args, task, device),
            num_clients=len(ids),
            config=fl.server.ServerConfig(num_rounds=remaining),
            strategy=strategy,
            client_resources={"num_cpus": args.actor_cpus,
                              "num_gpus": args.actor_gpus},
        )
        start_round += remaining

    logger.info(f"Xong sau {(time.time() - t0) / 60:.1f} phut. Ket qua: {args.out_dir}")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith((".csv", ".png", ".txt")):
            logger.info(f"  {f}")


if __name__ == "__main__":
    main()
