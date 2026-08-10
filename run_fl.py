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

CHAY TIEP KHI BI CAT GIUA CHUNG (mac dinh)
------------------------------------------
Kaggle het gio thi tien trinh bi giet bat cu luc nao. Moi round deu da ghi
xuong dia ngay (metrics CSV, checkpoint, log), nen khong mat gi.

Session sau, chay LAI DUNG LENH CU:

  - dem so dong trong metrics CSV de biet moi task da chay duoc bao nhieu round
  - task nao du round -> bo qua
  - task dang do -> `--mode resume`, chi chay so round CON THIEU
  - so round van danh lien tuc, CSV khong co dong trung

Muon lam lai tu dau thi them `--restart` (se bao loi neu out-dir da co ket qua,
de khong lo tay ghi de).

`--cm-every N` ghi confusion matrix moi N round thay vi chi ghi o cuoi task —
nen dat neu ban biet mot session khong chay het duoc mot task.
"""
import argparse
import glob
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


def rounds_done(out_dir, task):
    """So round da chay xong cho task nay, dem tu file metrics CSV.

    Doc CSV chu khong doc checkpoint de khong phai import torch trong launcher.
    CSV duoc ghi sau checkpoint trong cung mot round, nen con so nay bang hoac
    thap hon checkpoint dung MOT round -> resume cung lam lai nhieu nhat 1 round,
    khong bao gio nhay mat round nao.
    """
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
                n = max(n, sum(1 for _ in f) - 1)      # tru dong header
        except OSError:
            pass
    return max(n, 0)


def has_checkpoint(out_dir):
    return bool(glob.glob(os.path.join(out_dir, "checkpoints*", "latest.pth")))


def checkpoint_round(out_dir):
    """So round ghi trong latest.pth. Import torch muon vi thuong khong can toi."""
    cands = sorted(glob.glob(os.path.join(out_dir, "checkpoints*", "latest.pth")))
    if not cands:
        return 0
    try:
        import torch
        blob = torch.load(cands[0], map_location="cpu", weights_only=False)
        return int(blob.get("round", 0)) if isinstance(blob, dict) else 0
    except Exception as e:                                    # pragma: no cover
        print(f"Khong doc duoc {cands[0]}: {e}")
        return 0


def detect_progress(out_dir, tasks, rounds_per_task, done_rounds=None):
    """Tra ve ({task: so round da xong}, mo ta cach suy ra).

    Uu tien dem tu metrics CSV vi no chinh xac cho TUNG task. Neu khong co CSV
    (vd chi nhan duoc moi file .pth tu nguoi khac) thi suy tu tong so round ghi
    trong latest.pth: round danh lien tuc nen task thu i chiem khoang
    [i*R, (i+1)*R).

    CANH BAO: cach suy nay chi dung neu nguoi chay truoc dung CUNG mot --rounds.
    """
    prog = {t: rounds_done(out_dir, t) for t in tasks}
    if any(prog.values()):
        return prog, "dem so dong trong metrics CSV"

    total = done_rounds if done_rounds is not None else checkpoint_round(out_dir)
    if not total:
        return prog, "chua co gi"

    for i, t in enumerate(tasks):
        prog[t] = max(0, min(rounds_per_task, total - i * rounds_per_task))
    nguon = ("--done-rounds" if done_rounds is not None else "latest.pth")
    return prog, (f"suy tu {nguon} = {total} round tong cong, gia dinh "
                  f"{rounds_per_task} round/task")


def run_one_task(pdir, port, args, task, mode, client_ids, rounds):
    """Chay 1 task: 1 server + len(client_ids) client, cho den khi server thoat."""
    addr = f"127.0.0.1:{port}"
    logs = os.path.join(args.out_dir, "_logs")
    os.makedirs(logs, exist_ok=True)
    sfx = "flat" if task is None else f"task{task}"

    server_cmd = [PY, "server_iov.py", "--mode", mode, "--address", addr,
                  "--rounds", str(rounds), "--num-clients", str(len(client_ids)),
                  "--data-dir", args.data_dir, "--out-dir", args.out_dir,
                  "--local-epochs", str(args.local_epochs),
                  "--test-samples", str(args.test_samples)]
    if task is not None:
        server_cmd += ["--task", str(task)]
    if args.cm_every:
        server_cmd += ["--cm-every", str(args.cm_every)]
    server_cmd += args.server_extra

    print(f"\n{'=' * 70}\n[{sfx}] mode={mode} | {len(client_ids)} client | "
          f"{rounds} round\n{'=' * 70}", flush=True)
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
    p.add_argument("--cm-every", type=int, default=0,
                   help="Ghi confusion matrix moi N round (0 = chi ghi cuoi task). "
                        "Dat > 0 neu so bi cat giua chung truoc khi task ket thuc")
    p.add_argument("--done-rounds", type=int, default=None,
                   help="Khai bao thang da chay xong bao nhieu round TONG CONG. "
                        "Dung khi nhan checkpoint tu nguoi khac ma khong co CSV, "
                        "hoac khi so round trong latest.pth khong dang tin")
    p.add_argument("--restart", action="store_true",
                   help="Bo qua ket qua cu, bat dau lai tu dau. Mac dinh la CHAY TIEP "
                        "tu cho lan truoc dung — chay lai dung lenh cu la di tiep")
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

    # --- tiep tuc tu lan chay truoc (mac dinh) ---------------------------------
    # Kaggle het gio giua chung -> chay LAI DUNG LENH CU la di tiep tu cho do,
    # khong lam lai tu dau, khong ghi trung dong vao CSV.
    if args.restart:
        stale = [t for t in tasks if rounds_done(args.out_dir, t) > 0]
        if stale:
            sys.exit(f"--restart nhung {args.out_dir} da co ket qua cua task {stale}.\n"
                     f"Xoa thu muc do, hoac dung --out-dir khac, de khong ghi de nham.")
        progress = {t: 0 for t in tasks}
    else:
        progress, nguon = detect_progress(args.out_dir, tasks, args.rounds,
                                          args.done_rounds)
        if any(progress.values()):
            done_txt = ", ".join(f"task{t}={progress[t]}/{args.rounds}"
                                 for t in tasks if progress[t])
            print(f"Tiep tuc : {done_txt}   [{nguon}]")
            if "suy tu" in nguon:
                print("           LUU Y: khong co metrics CSV cua phan da chay. "
                      "Model chay tiep dung, nhung file CSV o day se THIEU cac "
                      "round truoc — xin ca thu muc out cua nguoi chay truoc "
                      "neu can du 150 dong.")

    t_start = time.time()
    failed = []
    for i, task in enumerate(tasks):
        done = progress.get(task, 0)
        if done >= args.rounds:
            print(f"[task {task}] da xong {done}/{args.rounds} round — bo qua",
                  flush=True)
            continue
        remaining = args.rounds - done

        ids = clients_with_data(args.data_dir, pool, task)
        if not ids:
            print(f"[task {task}] KHONG client nao co du lieu — bo qua", flush=True)
            failed.append(task)
            continue
        if len(ids) < len(pool):
            print(f"[task {task}] chi {len(ids)}/{len(pool)} client co du lieu: {ids}",
                  flush=True)
        # Co checkpoint roi thi LUON resume — ke ca task dau tien — vi co the
        # session truoc da chay do dang chinh task nay.
        mode = "resume" if has_checkpoint(args.out_dir) else "train"
        rc = run_one_task(pdir, port, args, task, mode, ids, remaining)
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
