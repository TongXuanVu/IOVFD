"""Kiem tra common.py / model_cnn1d.py trong repo nay co lech ban goc khong.

4 repo dung chung hai file nay (moi repo giu mot ban sao de tu chay duoc). Neu
sua o mot repo ma quen dong bo, ket qua giua cac bai het so sanh duoc. Script
nay in hash de doi chieu nhanh giua cac repo.

  python check_shared.py
  python check_shared.py --against ../VANFED-IDS
"""
import argparse
import hashlib
import os
import sys

SHARED = ["common.py", "model_cnn1d.py"]
HERE = os.path.dirname(os.path.abspath(__file__))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--against", type=str, default=None,
                   help="Thu muc repo khac de so sanh")
    a = p.parse_args()

    bad = False
    for name in SHARED:
        mine = os.path.join(HERE, name)
        if not os.path.exists(mine):
            print(f"THIEU {name}")
            bad = True
            continue
        h = sha(mine)
        print(f"{name:20s} {h[:16]}")
        if a.against:
            other = os.path.join(a.against, name)
            if not os.path.exists(other):
                print(f"  -> {a.against} khong co {name}")
                bad = True
            elif sha(other) != h:
                print(f"  -> LECH so voi {a.against}")
                bad = True
            else:
                print(f"  -> khop {a.against}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
