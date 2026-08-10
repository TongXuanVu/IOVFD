"""Chay mot vong lien ket day du bang MOT lenh — dung cho Kaggle/Colab.

Kaggle notebook khong mo duoc nhieu terminal, nen script nay tu sinh 1 server +
N client duoi dang tien trinh con roi cho den khi xong.

Che do class-incremental (mac dinh): chay noi tiep task 0 -> 4. Task dau
`--mode train`, cac task sau `--mode resume` nen so round va checkpoint chay
lien tuc, dung de do muc do quen giua cac task.

  python run_fl.py --project p1 --data-dir /kaggle/input/iov-100client --clients 10
  python run_fl.py --project p4 --tasks none --rounds 30 --clients 10
  python run_fl.py --project p4 --client-extra --arch rnn --server-extra --arch rnn

Truoc khi chay se KIEM TRA client nao thuc su co file cua task do — bo qua
client thieu du lieu thay vi de server treo cho mai (bo 10client bi thua).
"""
import argparse
import os
import shlex
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

PROJECTS = {
    "p1": ("P1-VANFED-IDS", 8081),
    "p2": ("P2-FEDIOV", 8082),
    "p3": ("P3-IOVFD", 8083),
    "p4": ("P4-SDNFL-IDS", 8084),
}
NUM_TASKS = 5

# Hai kieu bo tri thu muc:
#   - monorepo Rebuild-IOV : server_iov.py nam trong P1-VANFED-IDS/, P2-FEDIOV/...
#                            -> phai chi dinh --project
#   - repo doc lap         : server_iov.py nam ngay canh file nay -> tu nhan dien
STANDALONE = os.path.exists(os.path.join(ROOT, "server_iov.py"))


def clients_with_data(data_dir, client_ids, task):
    """Loc ra nhung client co file .pt cho task nay (task=None -> can it nhat 1 file)."""
    fed = os.path.join(data_dir, "federated_data")
    ok = []
    for cid in client_ids:
        if task is None:
            has = (any(os.path.exists(os.path.join(fed, f"client_{cid}_task_{t}.pt"))
                       for t in range(1, NUM_TASKS + 1))
                   or os.path.exists(os.path.join(fed, f"client_{cid}.pt")))
        else:
            has = os.path.exists(os.path.join(fed, f"client_{cid}_task_{task + 1}.pt"))
        if has:
            ok.append(cid)
    return ok


def stream(proc, prefix, log_path):
    """Doc stdout cua tien trinh con, in ra man hinh va ghi file."""
    with open(log_path, "a", encoding="utf-8") as fh:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            fh.write(line + "\n")
            fh.flush()
            if prefix:
                print(f"{prefix} {line}", flush=True)


def spawn(cmd, cwd, log_path, prefix=None):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1)
    th = threading.Thread(target=stream, args=(proc, prefix, log_path), daemon=True)
    th.start()
    return proc, th


def run_one_task(pdir, port, args, task, mode, client_ids):
    """Chay 1 task: 1 server + len(client_ids) client, cho den khi server thoat."""
    addr = f"127.0.0.1:{port}"
    logs = os.path.join(args.out_dir, "_logs")
    os.makedirs(logs, exist_ok=True)
    sfx = "flat" if task is None else f"task{task}"

    server_cmd = [PY, "server_iov.py", "--mode", mode, "--address", addr,
                  "--rounds", str(args.rounds), "--num-clients", str(len(client_ids)),
                  "--data-dir", args.data_dir, "--out-dir", args.out_dir,
                  "--local-epochs", str(args.local_epochs),
                  "--test-samples", str(args.test_samples)]
    if task is not None:
        server_cmd += ["--task", str(task)]
    server_cmd += args.server_extra

    print(f"\n{'=' * 70}\n[{sfx}] mode={mode} | {len(client_ids)} client | "
          f"{args.rounds} round\n{'=' * 70}", flush=True)
    srv, srv_th = spawn(server_cmd, pdir, os.path.join(logs, f"server_{sfx}.log"), "|")
    time.sleep(args.server_warmup)

    procs = []
    for cid in client_ids:
        cmd = [PY, "client_iov.py", "--client-id", str(cid), "--server", addr,
               "--data-dir", args.data_dir, "--max-samples", str(args.max_samples),
               "--batch-size", str(args.batch_size)]
        if task is not None:
            cmd += ["--task", str(task)]
        cmd += args.client_extra
        procs.append(spawn(cmd, pdir, os.path.join(logs, f"client{cid}_{sfx}.log")))
        time.sleep(args.client_stagger)

    try:
        rc = srv.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"[{sfx}] QUA GIO ({args.timeout}s) — dung server", flush=True)
        srv.kill()
        rc = -1
    srv_th.join(timeout=10)

    for p, _ in procs:
        if p.poll() is None:
            p.terminate()
    time.sleep(2)
    for p, _ in procs:
        if p.poll() is None:
            p.kill()
    return rc


def main():
    p = argparse.ArgumentParser(
        description="Chay 1 server + N client bang mot lenh (Kaggle/Colab)")
    p.add_argument("--project", choices=list(PROJECTS), default=None,
                   help="Chi can khi chay trong monorepo Rebuild-IOV; "
                        "repo doc lap tu nhan dien")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Mac dinh: <project>/out")
    p.add_argument("--clients", type=int, default=10,
                   help="So tien trinh client (lay client id 0..N-1)")
    p.add_argument("--client-ids", type=int, nargs="+", default=None,
                   help="Chi dinh id cu the, thay cho --clients")
    p.add_argument("--rounds", type=int, default=30, help="So round MOI task")
    p.add_argument("--tasks", type=str, default="all",
                   help="'all' = 0..4 noi tiep | 'none' = gop het | '0,1,2'")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=500_000)
    p.add_argument("--test-samples", type=int, default=1_000_000)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--server-warmup", type=float, default=15.0,
                   help="Giay cho server nap global test truoc khi bat client")
    p.add_argument("--client-stagger", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=20_000, help="Giay cho MOI task")
    # PHAI dung dang co dau BANG: --server-extra="--arch rnn"
    # Neu viet --server-extra "--arch rnn" thi argparse tuong "--arch" la mot
    # option moi va bao loi "expected one argument".
    p.add_argument("--server-extra", type=str, default="",
                   help='Tham so them cho server. PHAI co dau =, vd: --server-extra="--arch rnn"')
    p.add_argument("--client-extra", type=str, default="",
                   help='Tham so them cho client. PHAI co dau =, vd: --client-extra="--arch rnn"')
    args = p.parse_args()
    args.server_extra = shlex.split(args.server_extra)
    args.client_extra = shlex.split(args.client_extra)

    if STANDALONE:
        pdir = ROOT
        pdir_name = os.path.basename(os.path.normpath(ROOT))
        port = args.port or 8080
    else:
        if not args.project:
            p.error("Trong monorepo Rebuild-IOV phai chi dinh --project p1|p2|p3|p4")
        pdir_name, default_port = PROJECTS[args.project]
        pdir = os.path.join(ROOT, pdir_name)
        port = args.port or default_port
    args.out_dir = os.path.abspath(args.out_dir or os.path.join(pdir, "out"))
    args.data_dir = os.path.abspath(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(args.data_dir, "federated_data")):
        sys.exit(f"Khong thay {args.data_dir}/federated_data — sai --data-dir?")

    if args.tasks.strip().lower() == "none":
        tasks = [None]
    elif args.tasks.strip().lower() == "all":
        tasks = list(range(NUM_TASKS))
    else:
        tasks = [int(t) for t in args.tasks.replace(" ", "").split(",")]

    pool = args.client_ids if args.client_ids else list(range(args.clients))
    print(f"Project : {args.project or pdir_name}")
    print(f"Du lieu : {args.data_dir}")
    print(f"Ket qua : {args.out_dir}")
    print(f"Task    : {tasks} | {args.rounds} round/task | pool {len(pool)} client")

    t_start = time.time()
    failed = []
    for i, task in enumerate(tasks):
        ids = clients_with_data(args.data_dir, pool, task)
        if not ids:
            print(f"[task {task}] KHONG client nao co du lieu — bo qua", flush=True)
            failed.append(task)
            continue
        if len(ids) < len(pool):
            print(f"[task {task}] chi {len(ids)}/{len(pool)} client co du lieu: {ids}",
                  flush=True)
        mode = "train" if i == 0 else "resume"
        rc = run_one_task(pdir, port, args, task, mode, ids)
        if rc != 0:
            print(f"[task {task}] server thoat voi ma {rc}", flush=True)
            failed.append(task)
            break
        time.sleep(3)

    dt = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Xong sau {dt / 60:.1f} phut. Ket qua: {args.out_dir}")
    if failed:
        print(f"Task loi: {failed} — xem {args.out_dir}/_logs/")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith((".csv", ".png", ".txt")):
            print(f"  {f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
