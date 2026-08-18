"""Dong goi / khoi phuc trang thai chay do — de resume qua GitHub thay vi Kaggle dataset.

VAN DE: /kaggle/working bi xoa sach giua cac session. Cach chinh thong la
"Save Version" roi attach output cu lam input dataset — nhieu buoc tay, de
quen, va moi lan lai phai sua duong dan trong notebook.

CACH O DAY: nhet trang thai vao chinh repo (logs/resume/), day len GitHub.
Session sau chi can `git clone` la co du de chay tiep.

Chi luu nhung gi CAN de resume, khong luu het:
  checkpoints*/latest.pth      trong so model (P4 dung checkpoints_cnn)
  metrics_task*.csv            dem so round da xong (rounds_done() doc file nay)
  physics_branch_task*.pkl     cay lien ket — co san thi khong phai dung lai
  classification_report/cm/dst  ket qua da co, khong muon mat
  client_state_taskN.tar.gz    model cuc bo tung client (P2 Eq.15, P3 Alg.1
                               dong 10). CHI dong goi task DANG chay — task
                               cu khong bao gio duoc doc lai.
                               Mat file nay thi co che ca nhan hoa bi DUT.
KHONG luu: checkpoint tung round (.pth moi round ~170KB x 150 round = 26MB,
phinh lich su git), va file .png (ve lai duoc tu CSV).

Chay:
  python resume_state.py --status              # dang o round nao
  python resume_state.py --save                # out/ -> logs/resume/  (tren may, sau khi tai ket qua ve)
  python resume_state.py --load                # logs/resume/ -> out/  (tren Kaggle, truoc khi train)
"""
import argparse
import glob
import os
import re
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESUME = os.path.join(HERE, "logs", "resume")

# (mau glob, co giu khong) — theo thu tu uu tien khi in
MAU = [
    "metrics*.csv",
    "client_weights_*.csv",
    "physics_branch_task*.pkl",
    "classification_report_*.txt",
    "confusion_matrix_*.csv",
    "dst_fusion_*.json",
    "sim*.log",
]


def thu_muc_ckpt(goc):
    """Tra ve ten cac thu muc checkpoint co trong `goc`.

    P1 dung "checkpoints", P4 dung "checkpoints_cnn" / "checkpoints_rnn" (co
    hau to kien truc). Do bang glob thay vi viet cung ten.
    """
    return sorted(os.path.basename(d) for d in
                  glob.glob(os.path.join(goc, "checkpoints*"))
                  if os.path.isdir(d))


def task_cuc_bo(thu_muc):
    """Task moi nhat co client_state. Task cu khong bao gio duoc doc lai nen
    khong can dong goi (100 client x 5 task = qua nang cho git)."""
    cs = os.path.join(thu_muc, "client_state")
    if not os.path.isdir(cs):
        return None, []
    theo_task = {}
    for p in glob.glob(os.path.join(cs, "client_*_*.npz")):
        m = re.search(r"_(task\d+|flat)\.npz$", os.path.basename(p))
        if m:
            theo_task.setdefault(m.group(1), []).append(p)
    if not theo_task:
        return None, []
    t = sorted(theo_task)[-1]
    return t, sorted(theo_task[t])


def dem_round(thu_muc):
    """Doc so round da xong tung task — dung dung logic cua run_sim.rounds_done()."""
    ket = {}
    for p in sorted(glob.glob(os.path.join(thu_muc, "metrics*.csv"))):
        ten = os.path.basename(p)
        if ten.startswith("test_"):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
        except OSError:
            n = 0
        ket[ten] = max(n, 0)
    return ket


def round_checkpoint(thu_muc):
    """(ten thu muc ckpt, round) cua tung ban latest.pth tim duoc."""
    ket = []
    for ten in thu_muc_ckpt(thu_muc):
        p = os.path.join(thu_muc, ten, "latest.pth")
        if not os.path.exists(p):
            continue
        try:
            import torch
            r = torch.load(p, map_location="cpu", weights_only=False).get("round")
        except Exception as e:
            print(f"  (khong doc duoc {ten}/latest.pth: {e})")
            r = None
        ket.append((ten, r))
    return ket


def in_trang_thai(ten, thu_muc):
    print(f"\n{ten}: {thu_muc}")
    if not os.path.isdir(thu_muc):
        print("  (chua co)")
        return
    ck = round_checkpoint(thu_muc)
    if ck:
        for ten_ck, r in ck:
            print(f"  {ten_ck}/latest.pth : round {r}")
    else:
        print("  latest.pth       : KHONG CO")
    d = dem_round(thu_muc)
    if d:
        for k, v in d.items():
            print(f"  {k:<22}: {v} round")
    else:
        print("  (chua co metrics CSV)")
    for pkl in sorted(glob.glob(os.path.join(thu_muc, "physics_branch_task*.pkl"))):
        print(f"  cay co san        : {os.path.basename(pkl)}")
    t, files = task_cuc_bo(thu_muc)
    if t:
        print(f"  client_state      : {len(files)} client ({t})")
    for tar in sorted(glob.glob(os.path.join(thu_muc, "client_state_*.tar.gz"))):
        print(f"  client_state (nen): {os.path.basename(tar)}")


def chep(nguon, dich, ten_file):
    os.makedirs(os.path.dirname(dich), exist_ok=True)
    shutil.copy2(nguon, dich)
    return ten_file


def save(out_dir, force, bo_client_state=False):
    if not os.path.isdir(out_dir):
        sys.exit(f"Khong thay {out_dir} — chua chay lan nao?")
    cu, moi = dem_round(RESUME), dem_round(out_dir)
    tong_cu, tong_moi = sum(cu.values()), sum(moi.values())
    if tong_moi < tong_cu and not force:
        sys.exit(f"DUNG LAI: out/ dang o {tong_moi} round, con logs/resume/ da co "
                 f"{tong_cu} round. Luu de se LUI tien do.\n"
                 f"Chac chan thi them --force.")

    os.makedirs(RESUME, exist_ok=True)
    da = []
    for ten in thu_muc_ckpt(out_dir):
        latest = os.path.join(out_dir, ten, "latest.pth")
        if os.path.exists(latest):
            da.append(chep(latest, os.path.join(RESUME, ten, "latest.pth"),
                           f"{ten}/latest.pth"))
    for mau in MAU:
        for p in sorted(glob.glob(os.path.join(out_dir, mau))):
            da.append(chep(p, os.path.join(RESUME, os.path.basename(p)),
                           os.path.basename(p)))

    for cu_tar in glob.glob(os.path.join(RESUME, "client_state_*.tar.gz")):
        os.remove(cu_tar)                       # chi giu task moi nhat
    t, files = task_cuc_bo(out_dir)
    if t and not bo_client_state:
        tar = os.path.join(RESUME, f"client_state_{t}.tar.gz")
        with tarfile.open(tar, "w:gz") as tf:
            for f in files:
                tf.add(f, arcname=os.path.join("client_state",
                                               os.path.basename(f)))
        da.append(f"client_state_{t}.tar.gz ({len(files)} client, "
                  f"{os.path.getsize(tar) / 1024 / 1024:.1f} MB)")
    elif t:
        print(f"  (bo qua client_state cua {t}: {len(files)} file)")

    print(f"Da dong goi {len(da)} file -> {RESUME}")
    for f in da:
        print(f"   {f}")
    print("\nBuoc tiep: commit + push, roi session sau chi can git clone.")


def load(out_dir, force):
    if not os.path.isdir(RESUME):
        sys.exit(f"Khong thay {RESUME} — repo chua co trang thai nao de khoi phuc.")
    cu, moi = dem_round(RESUME), dem_round(out_dir)
    tong_cu, tong_moi = sum(cu.values()), sum(moi.values())
    if tong_moi > tong_cu and not force:
        sys.exit(f"DUNG LAI: out/ dang o {tong_moi} round, moi hon logs/resume/ "
                 f"({tong_cu} round). Khoi phuc se GHI DE ket qua moi hon.\n"
                 f"Chac chan thi them --force.")

    os.makedirs(out_dir, exist_ok=True)
    da = []
    for ten in thu_muc_ckpt(RESUME):
        latest = os.path.join(RESUME, ten, "latest.pth")
        if os.path.exists(latest):
            da.append(chep(latest, os.path.join(out_dir, ten, "latest.pth"),
                           f"{ten}/latest.pth"))
    for p in sorted(glob.glob(os.path.join(RESUME, "*"))):
        if not os.path.isfile(p):
            continue
        if p.endswith(".tar.gz"):
            with tarfile.open(p, "r:gz") as tf:
                tf.extractall(out_dir)
            n = len(glob.glob(os.path.join(out_dir, "client_state", "*.npz")))
            da.append(f"{os.path.basename(p)} -> client_state/ ({n} file)")
            continue
        da.append(chep(p, os.path.join(out_dir, os.path.basename(p)),
                       os.path.basename(p)))
    print(f"Da khoi phuc {len(da)} file -> {out_dir}")
    in_trang_thai("Sau khi khoi phuc", out_dir)
    print("\nChay tiep binh thuong. TUYET DOI KHONG them --restart.")


def main():
    p = argparse.ArgumentParser(description="Dong goi/khoi phuc trang thai resume")
    p.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true", help="out/ -> logs/resume/")
    g.add_argument("--load", action="store_true", help="logs/resume/ -> out/")
    g.add_argument("--status", action="store_true", help="Chi xem, khong dong gi")
    p.add_argument("--force", action="store_true",
                   help="Bo qua canh bao lui tien do / ghi de")
    p.add_argument("--no-client-state", action="store_true",
                   help="Khong dong goi client_state (commit nhe hon, nhung "
                        "resume se lam DUT co che ca nhan hoa cua P2/P3)")
    a = p.parse_args()

    if a.status:
        in_trang_thai("Trong repo (logs/resume)", RESUME)
        in_trang_thai("Dang chay (out)", a.out_dir)
    elif a.save:
        save(a.out_dir, a.force, a.no_client_state)
    else:
        load(a.out_dir, a.force)


if __name__ == "__main__":
    main()
