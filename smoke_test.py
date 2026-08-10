"""Kiem thu nhanh ca 4 project tren DU LIEU GIA (khong can CICIoV that).

Sinh du lieu dung dinh dang AFSIC-IoV roi chay lan luot:
  P1 VAN-FED-IDS, P2 FedIoV (Multi-Krum), P3 IoVFD (dual KD), P4 SDN-FL IDS.
Moi project chay 2 round voi vai client nho, sau do kiem tra 3 thu:
  - metrics CSV co du 12 cot va dung so dong
  - checkpoint latest.pth ton tai va nap lai duoc
  - confusion matrix CSV + classification report duoc sinh ra
Cuoi cung thu lai che do `--mode test` va `--mode resume`.

Chay:
  python smoke_test.py                  # tat ca
  python smoke_test.py --only p3        # chi mot project
  python smoke_test.py --keep           # giu lai thu muc tam de xem ket qua
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

NUM_CLASSES = 13
INPUT_LEN = 31
TASK_INCREMENTS = [3, 3, 3, 2, 2]

PROJECTS = {
    "p1": dict(dir="P1-VANFED-IDS", server="server_iov.py", client="client_iov.py",
               port=18081, server_args=[], client_args=[]),
    "p2": dict(dir="P2-FEDIOV", server="server_iov.py", client="client_iov.py",
               port=18082, server_args=["--krum-m", "3", "--byzantine", "1",
                                        "--width", "8", "16", "--grid-size", "3"],
               client_args=["--width", "8", "16", "--grid-size", "3"],
               attackers={3: "signflip"}),
    "p3": dict(dir="P3-IOVFD", server="server_iov.py", client="client_iov.py",
               port=18083, server_args=["--kd-steps", "3", "--kd-batch", "32"],
               client_args=["--batch-size", "64"]),
    "p4": dict(dir="P4-SDNFL-IDS", server="server_iov.py", client="client_iov.py",
               port=18084, server_args=["--arch", "cnn", "--weighting", "trust"],
               client_args=["--arch", "cnn", "--simulate-sdn"]),
}

# Repo doc lap: server_iov.py nam ngay canh file nay. Nhan dien project qua
# file dac trung cua tung bai roi chi kiem thu dung project do.
STANDALONE = os.path.exists(os.path.join(ROOT, "server_iov.py"))


def detect_project():
    for marker, key in (("model_kanconv.py", "p2"), ("generator.py", "p3"),
                        ("models_sdn.py", "p4")):
        if os.path.exists(os.path.join(ROOT, marker)):
            return key
    return "p1"


# ----------------------------------------------------------------------------
def make_fake_data(data_dir, n_clients, per_task=400, n_test=3000, seed=0):
    """Sinh du lieu gia dung layout AFSIC-IoV, co dich chuyen phan bo giua client."""
    rng = np.random.default_rng(seed)
    fed = os.path.join(data_dir, "federated_data")
    os.makedirs(fed, exist_ok=True)

    # tam cua tung lop trong khong gian 31 chieu -> bai toan hoc duoc
    centers = rng.normal(0, 2.0, (NUM_CLASSES, INPUT_LEN)).astype(np.float32)

    start = 0
    for t, inc in enumerate(TASK_INCREMENTS):
        classes = list(range(start, start + inc))
        start += inc
        for cid in range(n_clients):
            # non-IID: moi client thien ve mot lop trong task
            p = rng.dirichlet(np.full(len(classes), 0.5))
            y = rng.choice(classes, size=per_task, p=p).astype(np.int64)
            x = (centers[y] + rng.normal(0, 0.7, (per_task, INPUT_LEN))).astype(np.float16)
            torch.save({"x": torch.from_numpy(x), "y": torch.from_numpy(y)},
                       os.path.join(fed, f"client_{cid}_task_{t + 1}.pt"))

    y = rng.integers(0, NUM_CLASSES, n_test).astype(np.int64)
    x = (centers[y] + rng.normal(0, 0.7, (n_test, INPUT_LEN))).astype(np.float16)
    torch.save({"x": torch.from_numpy(x), "y": torch.from_numpy(y)},
               os.path.join(data_dir, "global_test_data.pt"))

    with open(os.path.join(data_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({f"attack_{i}": i for i in range(NUM_CLASSES)}, f)
    return data_dir


# ----------------------------------------------------------------------------
def run_federation(cfg, data_dir, out_dir, n_clients, rounds, extra_server=(), timeout=900):
    """Chay 1 server + n client, cho den khi server ket thuc."""
    pdir = os.path.join(ROOT, cfg["dir"])
    addr = f"127.0.0.1:{cfg['port']}"
    logs = os.path.join(out_dir, "_procs")
    os.makedirs(logs, exist_ok=True)

    def spawn(name, cmd):
        fh = open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8")
        return subprocess.Popen(cmd, cwd=pdir, stdout=fh, stderr=subprocess.STDOUT), fh

    server_cmd = [PY, cfg["server"], "--address", addr, "--rounds", str(rounds),
                  "--num-clients", str(n_clients), "--data-dir", data_dir,
                  "--out-dir", out_dir, "--test-samples", "0",
                  "--local-epochs", "1"] + cfg["server_args"] + list(extra_server)
    srv, srv_fh = spawn("server", server_cmd)
    time.sleep(6)

    procs = []
    for cid in range(n_clients):
        cmd = [PY, cfg["client"], "--client-id", str(cid), "--server", addr,
               "--data-dir", data_dir, "--max-samples", "0"] + cfg["client_args"]
        atk = cfg.get("attackers", {}).get(cid)
        if atk:
            cmd += ["--attack", atk]
        procs.append(spawn(f"client{cid}", cmd))
        time.sleep(0.6)

    try:
        rc = srv.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        srv.kill()
        rc = -1
    for p, fh in procs:
        if p.poll() is None:
            p.kill()
        fh.close()
    srv_fh.close()
    return rc, os.path.join(logs, "server.log")


def check_outputs(out_dir, rounds, expect_csv, cm_prefix):
    """Kiem tra CSV metric + checkpoint + confusion matrix."""
    problems = []

    csv_path = os.path.join(out_dir, expect_csv)
    if not os.path.exists(csv_path):
        problems.append(f"thieu {expect_csv}")
    else:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows[0]) != 12:
            problems.append(f"{expect_csv}: {len(rows[0])} cot (can 12)")
        if len(rows) - 1 != rounds:
            problems.append(f"{expect_csv}: {len(rows) - 1} dong (can {rounds})")
        for r in rows[1:]:
            if any(v.strip() in ("", "nan") for v in r):
                problems.append(f"{expect_csv}: co o trong/nan -> {r}")
                break

    ckpts = [d for d in os.listdir(out_dir) if d.startswith("checkpoints")]
    if not ckpts:
        problems.append("khong co thu muc checkpoint")
    else:
        latest = os.path.join(out_dir, ckpts[0], "latest.pth")
        if not os.path.exists(latest):
            problems.append("thieu latest.pth")
        else:
            blob = torch.load(latest, map_location="cpu", weights_only=False)
            if "model_state_dict" not in blob or "round" not in blob:
                problems.append("latest.pth sai dinh dang")

    files = os.listdir(out_dir)
    if not any(f.startswith(f"confusion_matrix_{cm_prefix}") and f.endswith(".csv")
               for f in files):
        problems.append(f"thieu confusion_matrix_{cm_prefix}*.csv")
    if not any(f.startswith("classification_report") for f in files):
        problems.append("thieu classification_report*.txt")
    return problems


def run_mode(cfg, data_dir, out_dir, mode, extra=(), wait=0.0):
    """Chay server o mot che do.

    wait > 0: chi khoi dong roi doi `wait` giay va kill. Dung cho --mode resume,
    vi resume se dung cho client ket noi va khong bao gio tu thoat.
    """
    pdir = os.path.join(ROOT, cfg["dir"])
    cmd = [PY, cfg["server"], "--mode", mode, "--data-dir", data_dir,
           "--out-dir", out_dir, "--test-samples", "0"] + cfg["server_args"] + list(extra)
    log = os.path.join(out_dir, "_procs", f"mode_{mode}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=pdir, stdout=fh, stderr=subprocess.STDOUT)
        if wait > 0:
            time.sleep(wait)
            rc = proc.poll()
            if rc is None:                      # van dang chay = da khoi dong duoc
                proc.kill()
                rc = 0
        else:
            try:
                rc = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -1
    with open(log, encoding="utf-8", errors="replace") as fh:
        return rc, fh.read()          # tra ve TOAN BO log, khong cat duoi


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Kiem thu nhanh 4 project tren du lieu gia")
    p.add_argument("--only", nargs="+", choices=list(PROJECTS), default=list(PROJECTS))
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--keep", action="store_true", help="Giu lai thu muc tam")
    p.add_argument("--workdir", type=str, default=None)
    args = p.parse_args()

    if STANDALONE:
        key = detect_project()
        PROJECTS[key]["dir"] = "."
        args.only = [key]
        print(f"Repo doc lap -> nhan dien project: {key.upper()}")

    work = args.workdir or tempfile.mkdtemp(prefix="rebuild_iov_smoke_")
    data_dir = os.path.join(work, "data")
    print(f"Thu muc tam: {work}")
    make_fake_data(data_dir, args.clients)
    print(f"Da sinh du lieu gia cho {args.clients} client x 5 task\n")

    results = {}
    for key in args.only:
        cfg = PROJECTS[key]
        out_dir = os.path.join(work, key)
        print(f"=== {key.upper()} ({cfg['dir']}) ===")

        t0 = time.time()
        rc, srv_log = run_federation(cfg, data_dir, out_dir, args.clients, args.rounds)
        dt = time.time() - t0
        if rc != 0:
            tail = open(srv_log, encoding="utf-8", errors="replace").read()[-2000:]
            results[key] = [f"server thoat voi ma {rc}"]
            print(f"  THAT BAI sau {dt:.0f}s. Duoi log server:\n{tail}\n")
            continue

        csv_name = "metrics_cnn.csv" if key == "p4" else "metrics.csv"
        cm_prefix = "cnn_final" if key == "p4" else "final"
        problems = check_outputs(out_dir, args.rounds, csv_name, cm_prefix)

        rc_t, out_t = run_mode(cfg, data_dir, out_dir, "test")
        if rc_t != 0:
            problems.append(f"--mode test loi: {out_t[-400:]}")
        # resume se treo cho client -> chi can no nap duoc checkpoint roi kill
        rc_r, out_r = run_mode(cfg, data_dir, out_dir, "resume",
                               ["--rounds", "1", "--num-clients", str(args.clients)],
                               wait=25.0)
        if "Resume tu round" not in out_r:
            problems.append(f"--mode resume khong nap duoc checkpoint: {out_r[-400:]}")

        results[key] = problems
        status = "OK" if not problems else f"{len(problems)} van de"
        print(f"  {status} ({dt:.0f}s)")
        for pr in problems:
            print(f"    - {pr}")
        print()

    print("=" * 60)
    ok = [k for k, v in results.items() if not v]
    bad = [k for k, v in results.items() if v]
    print(f"DAT : {', '.join(ok) if ok else '(khong co)'}")
    print(f"LOI : {', '.join(bad) if bad else '(khong co)'}")
    if args.keep or bad:
        print(f"\nKet qua giu tai: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
