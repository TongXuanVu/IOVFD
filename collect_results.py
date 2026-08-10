"""Gop ket qua nhieu lan chay thanh bang so sanh + do muc do QUEN.

Doc cac thu muc out_dir do run_fl.py / server_iov.py sinh ra, rui xuat:

  summary.csv           1 dong / (lan chay, task): metric cuoi task do
  forgetting_<run>.csv  ma tran quen: hang = task da hoc, cot = do sau task nao
  forgetting.csv        tong hop muc do quen cua tat ca lan chay
  comparison.csv        bang so sanh cuoi cung giua cac lan chay
  accuracy_curve.png    accuracy theo round, co vach ngan giua cac task
  forgetting_<run>.png  heatmap ma tran quen

Muc do quen tinh tu chinh cac confusion_matrix_task*.csv da luu san:
  acc(j, t) = accuracy trung binh tren CAC LOP CUA TASK j, do sau khi hoc xong task t
  forgetting(j) = max_{t < T} acc(j, t) - acc(j, T)          (T = task cuoi)
Day la dinh nghia chuan trong tai lieu class-incremental (Chaudhry et al. 2018).

Chay:
  python collect_results.py --runs P1=/kaggle/working/out_p1 P4-CNN=/kaggle/working/out_p4_cnn
  python collect_results.py --runs out_p1 out_p4_cnn --out-dir ket_qua
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

TASK_INCREMENTS = [3, 3, 3, 2, 2]
NUM_TASKS = len(TASK_INCREMENTS)
METRIC_KEYS = ["loss", "accuracy", "micro_precision", "micro_recall", "micro_f1",
               "macro_precision", "macro_recall", "macro_f1",
               "weighted_precision", "weighted_recall", "weighted_f1"]


def task_class_range(j):
    """Cac lop thuoc task j (0-indexed)."""
    start = sum(TASK_INCREMENTS[:j])
    return list(range(start, start + TASK_INCREMENTS[j]))


# ----------------------------------------------------------------------------
def read_metrics_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        rec = {"round": int(float(r["round"]))}
        for k in METRIC_KEYS:
            try:
                rec[k] = float(r[k])
            except (KeyError, ValueError):
                rec[k] = float("nan")
        out.append(rec)
    return out


def find_metric_files(run_dir):
    """Tra ve {task_or_None: duong_dan}. Ho tro ca metrics.csv lan metrics_cnn_task0.csv."""
    found = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "metrics*.csv"))):
        name = os.path.basename(path)
        if name.startswith("test_metrics"):
            continue
        m = re.search(r"task(\d+)", name)
        found[int(m.group(1)) if m else None] = path
    return found


def read_confusion(path):
    """Doc confusion matrix da chuan hoa -> (mang 2 chieu, danh sach ten lop)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    arr = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=float)
    return arr, names


def per_task_accuracy(run_dir):
    """acc[j][t] = accuracy tren lop cua task j, do sau khi hoc xong task t."""
    acc = np.full((NUM_TASKS, NUM_TASKS), np.nan)
    for t in range(NUM_TASKS):
        cands = glob.glob(os.path.join(
            run_dir, f"confusion_matrix_*task{t}_normalized.csv"))
        if not cands:
            continue
        cm, _ = read_confusion(sorted(cands)[0])
        diag = np.diag(cm)
        for j in range(t + 1):
            cls = [c for c in task_class_range(j) if c < len(diag)]
            if cls:
                acc[j, t] = float(np.mean(diag[cls]))
    return acc


def forgetting_from(acc):
    """forgetting(j) = max_{t<T} acc(j,t) - acc(j,T), chi tinh cho task da hoc xong."""
    last = None
    for t in range(NUM_TASKS - 1, -1, -1):
        if not np.all(np.isnan(acc[:, t])):
            last = t
            break
    if last is None:
        return {}, None
    out = {}
    for j in range(last):
        row = acc[j, j:last]
        if np.all(np.isnan(row)) or np.isnan(acc[j, last]):
            continue
        out[j] = float(np.nanmax(row) - acc[j, last])
    return out, last


# ----------------------------------------------------------------------------
def plot_curves(runs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Bo qua ve do thi ({e})")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    boundaries = set()
    for label, run_dir in runs.items():
        files = find_metric_files(run_dir)
        rounds, accs = [], []
        for task in sorted(files, key=lambda k: (k is not None, k)):
            recs = read_metrics_csv(files[task])
            rounds += [r["round"] for r in recs]
            accs += [r["accuracy"] for r in recs]
            if task is not None and recs:
                boundaries.add(recs[-1]["round"])
        if rounds:
            order = np.argsort(rounds)
            ax.plot(np.array(rounds)[order], np.array(accs)[order],
                    marker="o", ms=3, lw=1.4, label=label)
    for b in sorted(boundaries)[:-1]:
        ax.axvline(b + 0.5, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Round (lien tuc qua cac task; vach dut = doi task)")
    ax.set_ylabel("Accuracy tren global test (lop da hoc)")
    ax.set_title("Accuracy theo round")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "accuracy_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")

    for label, run_dir in runs.items():
        acc = per_task_accuracy(run_dir)
        if np.all(np.isnan(acc)):
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(acc, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_TASKS), [f"sau T{t}" for t in range(NUM_TASKS)])
        ax.set_yticks(range(NUM_TASKS), [f"task {j}" for j in range(NUM_TASKS)])
        ax.set_title(f"Accuracy tung task — {label}")
        for j in range(NUM_TASKS):
            for t in range(NUM_TASKS):
                if not np.isnan(acc[j, t]):
                    ax.text(t, j, f"{acc[j, t]:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if acc[j, t] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        path = os.path.join(out_dir, f"forgetting_{safe}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  {path}")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Gop ket qua + do muc do quen")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Duong dan out_dir, hoac NHAN=duong_dan")
    p.add_argument("--out-dir", type=str, default="ket_qua")
    args = p.parse_args()

    runs = {}
    for item in args.runs:
        if "=" in item and not os.path.isdir(item):
            label, path = item.split("=", 1)
        else:
            label, path = os.path.basename(os.path.normpath(item)), item
        if not os.path.isdir(path):
            print(f"Bo qua (khong phai thu muc): {path}")
            continue
        runs[label] = os.path.abspath(path)
    if not runs:
        sys.exit("Khong co thu muc hop le nao.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Gop {len(runs)} lan chay -> {os.path.abspath(args.out_dir)}\n")

    # --- summary: metric cuoi cung cua tung task ---
    summary_rows = []
    for label, run_dir in runs.items():
        files = find_metric_files(run_dir)
        if not files:
            print(f"[{label}] khong thay metrics*.csv — bo qua")
            continue
        for task in sorted(files, key=lambda k: (k is not None, k)):
            recs = read_metrics_csv(files[task])
            if not recs:
                continue
            last = recs[-1]
            summary_rows.append(
                {"run": label, "task": "gop" if task is None else task,
                 "rounds": len(recs), "last_round": last["round"],
                 **{k: round(last[k], 6) for k in METRIC_KEYS}})
            print(f"[{label}] task {task}: round {last['round']}, "
                  f"acc={last['accuracy']:.4f} macro_f1={last['macro_f1']:.4f}")

    if summary_rows:
        path = os.path.join(args.out_dir, "summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n  {path}")

    # --- ma tran quen ---
    forget_rows = []
    for label, run_dir in runs.items():
        acc = per_task_accuracy(run_dir)
        if np.all(np.isnan(acc)):
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        path = os.path.join(args.out_dir, f"forgetting_{safe}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task\\do_sau"] + [f"T{t}" for t in range(NUM_TASKS)])
            for j in range(NUM_TASKS):
                w.writerow([f"task{j}"] +
                           ["" if np.isnan(v) else f"{v:.6f}" for v in acc[j]])
        print(f"  {path}")

        fg, last = forgetting_from(acc)
        for j, v in fg.items():
            forget_rows.append({"run": label, "task": j,
                                "acc_khi_vua_hoc": round(float(acc[j, j]), 6),
                                "acc_sau_cung": round(float(acc[j, last]), 6),
                                "forgetting": round(v, 6)})
        if fg:
            avg_acc = float(np.nanmean(acc[:last + 1, last]))
            forget_rows.append({"run": label, "task": "TRUNG BINH",
                                "acc_khi_vua_hoc": "",
                                "acc_sau_cung": round(avg_acc, 6),
                                "forgetting": round(float(np.mean(list(fg.values()))), 6)})
            print(f"[{label}] quen trung binh = {np.mean(list(fg.values())):.4f} | "
                  f"accuracy trung binh cuoi = {avg_acc:.4f}")

    if forget_rows:
        path = os.path.join(args.out_dir, "forgetting.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(forget_rows[0]))
            w.writeheader()
            w.writerows(forget_rows)
        print(f"  {path}")

    # --- bang so sanh cuoi cung ---
    comp = []
    for label, run_dir in runs.items():
        files = find_metric_files(run_dir)
        if not files:
            continue
        last_task = sorted(files, key=lambda k: (k is not None, k))[-1]
        recs = read_metrics_csv(files[last_task])
        if not recs:
            continue
        last = recs[-1]
        acc = per_task_accuracy(run_dir)
        fg, _ = forgetting_from(acc)
        comp.append({"run": label,
                     "accuracy": round(last["accuracy"], 6),
                     "macro_f1": round(last["macro_f1"], 6),
                     "weighted_f1": round(last["weighted_f1"], 6),
                     "macro_precision": round(last["macro_precision"], 6),
                     "macro_recall": round(last["macro_recall"], 6),
                     "forgetting_tb": round(float(np.mean(list(fg.values()))), 6)
                                      if fg else ""})
    if comp:
        path = os.path.join(args.out_dir, "comparison.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comp[0]))
            w.writeheader()
            w.writerows(sorted(comp, key=lambda r: -r["macro_f1"]))
        print(f"  {path}")
        print("\n=== BANG SO SANH (sap theo macro-F1) ===")
        hdr = list(comp[0])
        print(" | ".join(h.ljust(14) for h in hdr))
        for r in sorted(comp, key=lambda r: -r["macro_f1"]):
            print(" | ".join(str(r[h]).ljust(14) for h in hdr))

    plot_curves(runs, args.out_dir)

    with open(os.path.join(args.out_dir, "runs.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    print(f"\nXong. Tat ca trong {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
