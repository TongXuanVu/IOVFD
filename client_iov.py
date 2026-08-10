"""P3 - IoVFD: Flower client (chieu chung cat GLOBAL -> LOCAL).

Bai bao: Cao, Di, Jin, FGCS 183 (2026).

"Dual knowledge distillation" gom hai chieu:
  - Tai CLIENT (file nay): model global lam GIAO VIEN cho model local.
        L = FocalLoss(local(x), y) + lambda_kd * KL(local(x) || global(x))
    Model local KHONG bi ghi de boi trong so global -> giu tinh ca nhan hoa,
    day la diem khac FedAvg. Global chi vao qua so hang KL.
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
                 task, lr, dropout, lambda_kd, temperature, personalized):
        self.cid = client_id
        self.device = device
        self.lr = lr
        self.lambda_kd = lambda_kd
        self.T = temperature
        self.personalized = personalized

        x, y = C.load_client_data(data_dir, client_id, task, max_samples)
        self.loader = C.make_loader(x, y, batch_size, shuffle=True)
        self.n_samples = len(y)
        self.label_counts = np.bincount(y, minlength=NUM_GLOBAL_CLASSES).tolist()

        # model local (ben vung qua cac round) + ban sao global lam giao vien
        self.model = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, dropout).to(device)
        self.teacher = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, dropout).to(device)
        self.criterion = FocalLoss(alpha=C.make_focal_alpha(y).to(device), gamma=2.0)

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

        # round 0 hoac che do khong ca nhan hoa: khoi tao local tu global
        if rnd <= 1 or not self.personalized:
            self.model.load_state_dict(C.ndarrays_to_state_dict(self.model, parameters))

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
        logger.info(f"[Client {self.cid}][Round {rnd}] n={self.n_samples} "
                    f"ce={sum_ce / nb:.4f} kd={sum_kd / nb:.4f} (lambda={lam})")
        return (C.get_model_parameters(self.model), self.n_samples,
                {"train_loss": sum_ce / nb, "kd_loss": sum_kd / nb,
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lambda-kd", type=float, default=1.0,
                   help="Trong so KL(local || global)")
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
                         personalized=not args.no_personalized)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
