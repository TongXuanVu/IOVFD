"""IoVFD — mot lenh chay het, dung CHE DO CUA BAI BAO.

Cao, Di, Jin, "IoVFD: An anomaly detection method for the Internet of Vehicles
based on federated learning and dual knowledge distillation", FGCS 183 (2026)
108588.

MAC DINH = dung bai nhat:
  generator          Eq. (14)-(16): Linear -> 3 khoi (MultiHead+Conv1D) -> Linear
                     Dau vao generator la NHAN MEM y*, khong phai vector nhieu.
  --noise-mode paper Eq. (12) nguyen van (xem canh bao ben duoi)
  chaos r=4, u0=0.01 dung so cua bai
  lambda1 = lambda2 = 0.2, lr = 1e-4, batch = 32   (muc 4.1 cua bai)
  KHONG FedAvg       Algorithm 1 khong he trung binh tham so
  KHONG doi khang    Eq. (20) chi co hai so hang
  full data          moi client dung het shard, danh gia tren het tap test

CHO LECH SO VOI BAI, phai ghi trong bao cao:
  1. Bai la bai toan NHI PHAN (normal/abnormal, Bang 3). CICIoV co 13 lop.
     Eq. (10) y* = (1-z, z) duoc mo rong: lop dung giu (1-z), phan z chia deu
     cho 12 lop con lai. Voi 2 lop thi trung y nguyen Eq. (10).
  2. Eq. (12) z = (1/sqrt(2pi)) exp(-u^2/2) duoc bai goi la "standard Gaussian
     noise sampling" nhung do la HAM MAT DO chu khong phai phep lay mau: ket
     qua luon nam trong [0.2420, 0.3989], khong bao gio am. Dung lam nhan mem
     thi hop ly, goi la Gauss thi sai. --noise-mode boxmuller de doi chung.
  3. Bai khong cho so chieu nhung, so head, kich thuoc kernel cua generator.
     seq_len=8, channels=32, 4 head, kernel 3 la lua chon cua ta. So khoi = 3
     thi dung bai.
  4. Bai dung LSTM/CNN cho model phat hien; ta giu CNN1D lam backbone chung
     voi P1/P2/P4 de so sanh cong bang.
  5. Bai dat kich thuoc tap gia L = 50000. O day pseudo-data duoc sinh THEO LO
     moi buoc chung cat (kd_steps * stu_steps * kd_batch mau moi round) chu
     khong dung san mot tap co dinh — tuong duong ve luong, khac ve cach cap.

Chay:
  python main.py --data_dir /kaggle/input/... --num_users 100
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser(description="IoVFD: mot lenh chay het")
    p.add_argument("--data_dir", required=True,
                   help="Thu muc chua federated_data/ va global_test_data.pt")
    p.add_argument("--out_dir", default=os.path.join(HERE, "out"))
    p.add_argument("--num_users", type=int, default=100)
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--com_round", type=int, default=30, help="Round MOI task")
    p.add_argument("--local_ep", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Bai dat 32")
    p.add_argument("--lr", type=float, default=1e-4, help="Bai dat 0.0001")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--max_samples", type=int, default=0, help="0 = full data")
    p.add_argument("--test_samples", type=int, default=0,
                   help="0 = danh gia tren HET tap test moi round")
    # --- chung cat doi (muc 3.2, 3.3) ---
    p.add_argument("--lambda1", type=float, default=0.2,
                   help="lambda_1 cua Eq.(26), global. Bai dat 0.2")
    p.add_argument("--lambda_kd", type=float, default=0.2,
                   help="lambda_2 cua Eq.(29), client. Bai dat 0.2")
    p.add_argument("--temperature", type=float, default=3.0,
                   help="t cua soft target. Bai khong cho gia tri")
    p.add_argument("--pretrain_epochs", type=int, default=1,
                   help="Algorithm 1 dong 2: pretrain local truoc vong FL")
    p.add_argument("--no_personalized", action="store_true",
                   help="Ghi de local bang global moi round -> mat co che cua bai")
    # --- generator (muc 3.2.1) ---
    p.add_argument("--gen_blocks", type=int, default=3)
    p.add_argument("--gen_seq_len", type=int, default=8)
    p.add_argument("--gen_channels", type=int, default=32)
    p.add_argument("--gen_heads", type=int, default=4)
    p.add_argument("--noise_mode", choices=["paper", "boxmuller"], default="paper")
    p.add_argument("--chaos_r", type=float, default=4.0, help="Bai dat 4")
    p.add_argument("--chaos_u0", type=float, default=0.01, help="Bai dat 0.01")
    p.add_argument("--beta_div", type=float, default=1.0,
                   help="Trong so L_dis. Eq.(20) khong nhan he so -> de 1.0")
    p.add_argument("--beta_adv", type=float, default=0.0,
                   help="So hang doi khang — KHONG co trong Eq.(20)")
    p.add_argument("--kd_steps", type=int, default=30, help="0 = tat KD")
    p.add_argument("--gen_steps", type=int, default=1)
    p.add_argument("--stu_steps", type=int, default=5)
    p.add_argument("--kd_batch", type=int, default=128)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_s", type=float, default=1e-4)
    # --- doi chung ---
    p.add_argument("--fedavg_init", action="store_true",
                   help="Trung binh tham so truoc khi chung cat. Algorithm 1 "
                        "KHONG lam vay")
    p.add_argument("--weighted_ensemble", action="store_true",
                   help="Ensemble co trong so theo so mau. Eq.(17) la 1/n")
    # --- van hanh ---
    p.add_argument("--cm_every", type=int, default=5)
    p.add_argument("--flat", action="store_true",
                   help="Gop ca 5 task lam mot (khong class-incremental)")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--fed_subdir", default="federated_data",
                   choices=["federated_data", "federated_data_fewshot",
                            "federated_data_10shot"])
    p.add_argument("--actor_gpus", type=float, default=-1.0)
    p.add_argument("--actor_cpus", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    argv = [
        "run_sim.py",
        "--data-dir", a.data_dir,
        "--out-dir", a.out_dir,
        "--clients", str(a.num_users),
        "--rounds", str(a.com_round),
        "--tasks", "none" if a.flat else ",".join(str(t) for t in range(a.tasks)),
        "--local-epochs", str(a.local_ep),
        "--batch-size", str(a.batch_size),
        "--lr", str(a.lr),
        "--dropout", str(a.dropout),
        "--max-samples", str(a.max_samples),
        "--test-samples", str(a.test_samples),
        "--cm-every", str(a.cm_every),
        "--seed", str(a.seed),
        "--fed-subdir", a.fed_subdir,
        "--lambda-kd", str(a.lambda_kd),
        "--lambda1", str(a.lambda1),
        "--temperature", str(a.temperature),
        "--pretrain-epochs", str(a.pretrain_epochs),
        "--kd-steps", str(a.kd_steps),
        "--gen-steps", str(a.gen_steps),
        "--stu-steps", str(a.stu_steps),
        "--kd-batch", str(a.kd_batch),
        "--lr-g", str(a.lr_g),
        "--lr-s", str(a.lr_s),
        "--beta-div", str(a.beta_div),
        "--beta-adv", str(a.beta_adv),
        "--chaos-r", str(a.chaos_r),
        "--chaos-u0", str(a.chaos_u0),
        "--noise-mode", a.noise_mode,
        "--gen-seq-len", str(a.gen_seq_len),
        "--gen-channels", str(a.gen_channels),
        "--gen-heads", str(a.gen_heads),
        "--gen-blocks", str(a.gen_blocks),
        "--actor-gpus", str(a.actor_gpus),
        "--actor-cpus", str(a.actor_cpus),
    ]
    for co, bat in (("--no-personalized", a.no_personalized),
                    ("--fedavg-init", a.fedavg_init),
                    ("--weighted-ensemble", a.weighted_ensemble),
                    ("--restart", a.restart)):
        if bat:
            argv.append(co)

    print("=" * 70)
    print("IoVFD | Cao, Di, Jin, FGCS 183 (2026) 108588")
    print(f"  du lieu   : {a.data_dir}")
    print(f"  ket qua   : {a.out_dir}")
    print(f"  cau hinh  : {a.num_users} client | {a.tasks} task x {a.com_round} "
          f"round = {a.tasks * a.com_round} round")
    gen = ("Eq.(14)-(16): Linear -> %d x (MultiHead + Conv1D) -> Linear"
           % a.gen_blocks)
    print(f"  generator : {gen}")
    nhieu = ("Eq.(12) nguyen van — z in [0.242, 0.399], KHONG phai Gauss chuan"
             if a.noise_mode == "paper" else "Box-Muller (Gauss that, lech bai)")
    print(f"  nhieu     : logistic r={a.chaos_r}, u0={a.chaos_u0} | {nhieu}")
    print(f"  chung cat : Eq.(26) L_GM = CE + {a.lambda1}*KL | "
          f"Eq.(29) L_LM = CE + {a.lambda_kd}*KL")
    tong_hop = ("FedAvg roi chung cat (doi chung, KHONG theo Algorithm 1)"
                if a.fedavg_init else "chi chung cat (dung Algorithm 1)")
    print(f"  tong hop  : {tong_hop}")
    if a.beta_adv:
        print(f"  CANH BAO  : beta_adv={a.beta_adv} — Eq.(20) khong co so hang nay")
    if a.no_personalized:
        print("  CANH BAO  : tat ca nhan hoa — mat co che chinh cua bai")
    print("  LUU Y: bai la bai toan NHI PHAN; o day 13 lop nen Eq.(10) phai mo")
    print("         rong. Xem noisy_onehot() trong generator.py")
    print("=" * 70, flush=True)

    sys.argv = argv
    import run_sim
    run_sim.main()


if __name__ == "__main__":
    main()
