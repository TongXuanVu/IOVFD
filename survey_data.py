"""Khao sat bo du lieu TRUOC KHI chay: moi task co bao nhieu client, bao nhieu mau.

Tra loi dung cau hoi "so client tham gia moi task la bao nhieu" — bang so lieu
that chu khong phai gia dinh. Quan trong vi:

  - file .pt ton tai KHONG co nghia la co du lieu; shard co the rong hoac chi
    vai chuc mau, luc do client van "tham gia" nhung dong gop bang khong
  - bo 10client bi thua (co client thieu han vai task), bo 100client thi day du
  - so lop moi client thay duoc khac nhau (non-IID) — can de mo ta trong bao cao

Xuat ra man hinh + file CSV de dua thang vao phan mo ta du lieu.

Chay:
  python survey_data.py --data-dir /kaggle/input/iov-100client
  python survey_data.py --data-dir <DATA> --clients 0 1 2 3 4 --out khaosat.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

_P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "P1-VANFED-IDS")
if os.path.isdir(_P1) and _P1 not in sys.path:
    sys.path.insert(0, _P1)

import common as C                               # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Khao sat du lieu truoc khi chay")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--clients", type=int, nargs="+", default=None,
                   help="Mac dinh: do tim tat ca client co trong thu muc")
    p.add_argument("--min-samples", type=int, default=100,
                   help="Duoi nguong nay coi nhu client khong dong gop dang ke")
    p.add_argument("--out", type=str, default=None, help="Ghi bang ra file CSV")
    p.add_argument("--quiet", action="store_true", help="Khong in bang chi tiet")
    args = p.parse_args()

    fed = os.path.join(args.data_dir, "federated_data")
    if not os.path.isdir(fed):
        sys.exit(f"Khong thay {fed}")

    if args.clients:
        clients = args.clients
    else:
        ids = set()
        for f in os.listdir(fed):
            if f.startswith("client_") and f.endswith(".pt"):
                try:
                    ids.add(int(f.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        clients = sorted(ids)
    print(f"Thu muc : {args.data_dir}")
    print(f"Client  : {len(clients)} (id {min(clients)}..{max(clients)})\n")

    rows = []
    for t in range(C.NUM_TASKS):
        n_file = n_ok = 0
        tong, sizes, lop = 0, [], set()
        for cid in clients:
            path = os.path.join(fed, f"client_{cid}_task_{t + 1}.pt")
            if not os.path.exists(path):
                continue
            n_file += 1
            try:
                _, y = C._read_pt(path)
            except Exception as e:
                print(f"  Doc loi {os.path.basename(path)}: {e}")
                continue
            sizes.append(len(y))
            tong += len(y)
            lop.update(np.unique(y).tolist())
            if len(y) >= args.min_samples:
                n_ok += 1
        hoc_den = C.learned_classes(t)
        rows.append({
            "task": t,
            "client_co_file": n_file,
            f"client_tren_{args.min_samples}_mau": n_ok,
            "tong_mau": tong,
            "mau_it_nhat": min(sizes) if sizes else 0,
            "mau_trung_binh": int(np.mean(sizes)) if sizes else 0,
            "mau_nhieu_nhat": max(sizes) if sizes else 0,
            "lop_xuat_hien": len(lop),
            "lop_ky_vong_luy_ke": hoc_den,
        })

    print(f"{'task':>4} {'file':>5} {'>=' + str(args.min_samples) + ' mau':>9} "
          f"{'tong mau':>10} {'it nhat':>8} {'tb':>8} {'nhieu nhat':>11} "
          f"{'so lop':>7} {'ky vong':>8}")
    for r in rows:
        print(f"{r['task']:>4} {r['client_co_file']:>5} "
              f"{r[f'client_tren_{args.min_samples}_mau']:>9} {r['tong_mau']:>10} "
              f"{r['mau_it_nhat']:>8} {r['mau_trung_binh']:>8} "
              f"{r['mau_nhieu_nhat']:>11} {r['lop_xuat_hien']:>7} "
              f"{r['lop_ky_vong_luy_ke']:>8}")

    # --- canh bao ---
    print()
    canh_bao = []
    n_files = {r["task"]: r["client_co_file"] for r in rows}
    if len(set(n_files.values())) > 1:
        canh_bao.append(
            f"So client co du lieu KHAC NHAU giua cac task: {n_files}. "
            "run_fl.py se tu dung dung so client co du lieu cho tung task; "
            "muon so client CO DINH thi them --require-all-tasks.")
    for r in rows:
        k = f"client_tren_{args.min_samples}_mau"
        if r[k] < r["client_co_file"]:
            canh_bao.append(
                f"Task {r['task']}: {r['client_co_file'] - r[k]} client co file "
                f"nhung duoi {args.min_samples} mau — tham gia ma gan nhu khong "
                f"dong gop gi.")
        if r["mau_it_nhat"] == 0:
            canh_bao.append(f"Task {r['task']}: co shard RONG (0 mau).")
    if not canh_bao:
        print("Khong co gi bat thuong: moi task du client, moi client du mau.")
    for c in canh_bao:
        print(f"  CANH BAO: {c}")

    # --- global test ---
    gt = os.path.join(args.data_dir, "global_test_data.pt")
    if os.path.exists(gt):
        _, y = C._read_pt(gt)
        d = Counter(y.tolist())
        print(f"\nGlobal test: {len(y)} mau, {len(d)} lop "
              f"(it nhat {min(d.values())} mau/lop, nhieu nhat {max(d.values())})")
        thieu = [c for c in range(C.NUM_GLOBAL_CLASSES) if c not in d]
        if thieu:
            print(f"  CANH BAO: global test THIEU lop {thieu}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nDa ghi {args.out}")


if __name__ == "__main__":
    main()
