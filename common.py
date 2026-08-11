"""Tien ich dung chung cho 4 baseline IoV (CNN1D + Flower).

Bao gom:
  - Nap du lieu CICIoV .pt: {'x': (N,31) float16, 'y': (N,) int64}
  - Che do class-incremental: TASK_INCREMENTS = [3,3,3,2,2]
  - 12 metric moi round -> CSV
  - Checkpoint moi round + nap lai (train / resume / test)
  - Confusion matrix cuoi task: CSV + PNG + classification report

Quy uoc data (khop AFSIC-IoV):
  <data_dir>/federated_data/client_<id>_task_<t>.pt   voi t = 1..5
  <data_dir>/global_test_data.pt
  <data_dir>/class_mapping.json
"""
import csv
import json
import logging
import os
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Hang so
# ----------------------------------------------------------------------------
# Thu muc con chua shard cua client. Bo du lieu co nhieu bien the:
#   federated_data          — day du
#   federated_data_fewshot  — ban few-shot
#   federated_data_10shot   — ban 10-shot
# Doi bang set_fed_subdir() truoc khi goi load_client_data().
FED_SUBDIR = "federated_data"


def set_fed_subdir(name):
    global FED_SUBDIR
    FED_SUBDIR = name


NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31
NUM_TASKS = 5
TASK_INCREMENTS = [3, 3, 3, 2, 2]          # giong AFSIC-IoV / FedLiTeCAN

METRIC_KEYS = [
    "loss", "accuracy",
    "micro_precision", "micro_recall", "micro_f1",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]
CSV_HEADER = ["round"] + METRIC_KEYS

FALLBACK_CLASS_NAMES = [
    "Benign", "DoS", "double", "force-neutral", "fuzzing", "interval",
    "rpm", "rpm-accessory", "speed", "speed-accessory", "standstill",
    "systematic", "triple",
]


def learned_classes(task: Optional[int]) -> int:
    """So lop da hoc tinh den het task nay (0-indexed). task=None -> tat ca."""
    if task is None:
        return NUM_GLOBAL_CLASSES
    return sum(TASK_INCREMENTS[:task + 1])


def load_class_names(data_dir: str) -> List[str]:
    path = os.path.join(data_dir, "class_mapping.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            mapping = json.load(f)
        names = [None] * len(mapping)
        for name, idx in mapping.items():
            names[int(idx)] = name
        return [n if n is not None else f"class_{i}" for i, n in enumerate(names)]
    return list(FALLBACK_CLASS_NAMES)


# ----------------------------------------------------------------------------
# Du lieu
# ----------------------------------------------------------------------------
def subsample_capped(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42):
    """Giu toan bo lop thieu so, cat bot lop da so cho den khi <= max_samples."""
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    classes = sorted(counts, key=lambda c: counts[c])   # lop it mau truoc
    remaining, keep_idx = max_samples, []
    for i, c in enumerate(classes):
        quota = remaining // (len(classes) - i)
        idx = np.where(y == c)[0]
        if len(idx) > quota:
            idx = rng.choice(idx, quota, replace=False)
        keep_idx.append(idx)
        remaining -= len(idx)
    keep = np.concatenate(keep_idx)
    rng.shuffle(keep)
    return x[keep], y[keep]


def _read_pt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "x" in blob:
        x, y = blob["x"], blob["y"]
    elif isinstance(blob, (list, tuple)) and len(blob) == 2:
        x, y = blob
    else:
        raise ValueError(f"Khong hieu dinh dang file: {path}")
    x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
    y = (y.numpy() if torch.is_tensor(y) else np.asarray(y)).astype(np.int64)
    return x, y


def load_client_data(data_dir: str, client_id: int, task: Optional[int],
                     max_samples: int = 500_000, seed: int = 42):
    """Nap du lieu 1 client.

    task = None -> gop toan bo 5 task (FL thuong, dung nhu 4 bai bao).
    task = 0..4 -> CHI nap dung task do (class-incremental, do muc do quen).
    """
    fed_dir = os.path.join(data_dir, FED_SUBDIR)
    if task is not None:
        paths = [os.path.join(fed_dir, f"client_{client_id}_task_{task + 1}.pt")]
    else:
        paths = [os.path.join(fed_dir, f"client_{client_id}_task_{t}.pt")
                 for t in range(1, NUM_TASKS + 1)]
        flat = os.path.join(fed_dir, f"client_{client_id}.pt")
        if os.path.exists(flat):
            paths = [flat]

    xs, ys = [], []
    for p in paths:
        if not os.path.exists(p):
            logger.warning(f"Bo qua (khong ton tai): {p}")
            continue
        xi, yi = _read_pt(p)
        xs.append(xi)
        ys.append(yi)
    if not xs:
        raise FileNotFoundError(
            f"Client {client_id} khong co file nao trong {fed_dir}")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    del xs, ys
    x, y = subsample_capped(x, y, max_samples, seed)
    x = x.astype(np.float32)
    logger.info(f"Client {client_id} (task={task}): n={len(y)} | "
                f"classes={dict(sorted(Counter(y.tolist()).items()))}")
    return x, y


def stratified_subsample(x, y, max_samples, seed=42):
    """Lay mau GIU NGUYEN TI LE cac lop.

    Khac subsample_capped (can bang lai lop): ham nay giu dung phan bo goc, nen
    metric do tren mau la UOC LUONG KHONG CHECH cua metric tren toan bo tap.
    Bat buoc dung cho tap TEST — neu can bang lai thi accuracy bao cao se khong
    con la accuracy tren tap test that.
    """
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    ty_le = max_samples / len(y)
    giu = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = max(1, int(round(len(idx) * ty_le)))       # giu it nhat 1 mau/lop
        giu.append(rng.choice(idx, min(k, len(idx)), replace=False))
    keep = np.concatenate(giu)
    rng.shuffle(keep)
    return x[keep], y[keep]


def load_global_test(data_dir: str, max_samples: int = 1_000_000,
                     task: Optional[int] = None, seed: int = 42):
    """Nap global test. task khac None -> loc ve cac lop DA HOC (0..n-1)."""
    path = os.path.join(data_dir, "global_test_data.pt")
    logger.info(f"Nap global test: {path}")
    x, y = _read_pt(path)
    logger.info(f"Global test goc: n={len(y)}")
    if task is not None:
        n_cls = learned_classes(task)
        keep = y < n_cls
        x, y = x[keep], y[keep]
        logger.info(f"Task {task}: loc ve lop 0-{n_cls - 1} -> n={len(y)}")
    n_goc = len(y)
    x, y = stratified_subsample(x, y, max_samples, seed)      # GIU ti le lop
    if max_samples != 0:
        x = x.astype(np.float32)
    if len(y) < n_goc:
        logger.info(f"Lay mau theo ti le: {n_goc} -> {len(y)} mau "
                    f"(phan bo lop giu nguyen, metric khong chech)")
    logger.info(f"Danh gia moi round tren n={len(y)} mau (dtype={x.dtype})")
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=4096, shuffle=False)
    return loader, y


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_focal_alpha(y: np.ndarray, num_classes: int = NUM_GLOBAL_CLASSES):
    """alpha = sqrt(N / n_c) — giong FedLiTeCAN."""
    cnt = Counter(y.tolist())
    total = len(y)
    return torch.tensor(
        [np.sqrt(total / cnt[c]) if cnt.get(c) else 1.0 for c in range(num_classes)],
        dtype=torch.float32)


# ----------------------------------------------------------------------------
# Metric
# ----------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    loss: float = float("nan")) -> Dict[str, float]:
    """Du 11 chi so: loss, accuracy, micro/macro/weighted P-R-F1."""
    m = {"loss": float(loss),
         "accuracy": float((y_true == y_pred).mean())}
    for avg in ("micro", "macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        m[f"{avg}_precision"] = float(p)
        m[f"{avg}_recall"] = float(r)
        m[f"{avg}_f1"] = float(f1)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Tra ve (metrics, y_true, y_pred). y_true/y_pred dung cho confusion matrix."""
    model.eval()
    loss_sum, n_batches = 0.0, 0
    preds_buf, targs_buf = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device).float(), yb.to(device)
        out = model(xb)
        loss_sum += criterion(out, yb).item()
        n_batches += 1
        preds_buf.append(out.argmax(1).cpu().numpy().astype(np.int16))
        targs_buf.append(yb.cpu().numpy().astype(np.int16))
    preds = np.concatenate(preds_buf)
    targs = np.concatenate(targs_buf)
    del preds_buf, targs_buf
    return compute_metrics(targs, preds, loss_sum / max(n_batches, 1)), targs, preds


def format_metrics(rnd: int, m: Dict[str, float]) -> str:
    return (f"[Round {rnd}] loss={m['loss']:.4f} acc={m['accuracy']:.4f} | "
            f"micro P/R/F1={m['micro_precision']:.4f}/{m['micro_recall']:.4f}/{m['micro_f1']:.4f} | "
            f"macro P/R/F1={m['macro_precision']:.4f}/{m['macro_recall']:.4f}/{m['macro_f1']:.4f} | "
            f"weighted P/R/F1={m['weighted_precision']:.4f}/{m['weighted_recall']:.4f}/{m['weighted_f1']:.4f}")


def append_csv_row(path: str, row: List):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow(row)


def log_and_save_metrics(rnd: int, m: Dict[str, float], csv_file: str):
    logger.info(format_metrics(rnd, m))
    append_csv_row(csv_file, [rnd] + [round(m[k], 6) for k in METRIC_KEYS])


# ----------------------------------------------------------------------------
# Confusion matrix (cuoi task)
# ----------------------------------------------------------------------------
def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str,
                          tag: str, class_names: Optional[List[str]] = None):
    """Luu confusion matrix: CSV (tho + chuan hoa), PNG, classification report."""
    os.makedirs(out_dir, exist_ok=True)
    n_cls = int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(n_cls))
    names = (class_names or FALLBACK_CLASS_NAMES)[:n_cls]
    if len(names) < n_cls:
        names += [f"class_{i}" for i in range(len(names), n_cls)]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    for arr, suffix, fmt in ((cm, "", "%d"), (cm_norm, "_normalized", "%.6f")):
        path = os.path.join(out_dir, f"confusion_matrix_{tag}{suffix}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + names)
            for i, row in enumerate(arr):
                w.writerow([names[i]] + [fmt % v for v in row])
        logger.info(f"Luu {path}")

    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=names, digits=4, zero_division=0)
    rp = os.path.join(out_dir, f"classification_report_{tag}.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Luu {rp}\n{report}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n_cls, 0.9 + 0.65 * n_cls))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_cls), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_cls), names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix ({tag}) — chuan hoa theo hang")
        thr = 0.5
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm_norm[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        png = os.path.join(out_dir, f"confusion_matrix_{tag}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        logger.info(f"Luu {png}")
    except Exception as e:                                    # pragma: no cover
        logger.warning(f"Bo qua ve PNG confusion matrix ({e}). CSV van co day du.")

    return cm


# ----------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------
def get_model_parameters(model) -> List[np.ndarray]:
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def ndarrays_to_state_dict(model, ndarrays) -> OrderedDict:
    keys = model.state_dict().keys()
    return OrderedDict({k: torch.tensor(v) for k, v in zip(keys, ndarrays)})


def save_checkpoint(ckpt_dir: str, rnd: int, state_dict, extra: Optional[Dict] = None) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {"round": rnd, "model_state_dict": state_dict}
    if extra:
        payload.update(extra)
    path = os.path.join(ckpt_dir, f"round_{rnd:03d}.pth")
    torch.save(payload, path)
    torch.save(payload, os.path.join(ckpt_dir, "latest.pth"))
    logger.info(f"[Round {rnd}] luu checkpoint -> {path}")
    return path


def load_checkpoint(path: str, model) -> Tuple[int, Dict]:
    """Tra ve (round da chay xong, phan extra trong checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        extra = {k: v for k, v in ckpt.items()
                 if k not in ("round", "model_state_dict")}
        return int(ckpt.get("round", 0)), extra
    model.load_state_dict(ckpt)
    return 0, {}


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Flower da co handler rieng; neu de no propagate len root thi moi dong log
    # bi in hai lan -> tren Kaggle output dai gap doi va rat kho doc.
    flwr_logger = logging.getLogger("flwr")
    flwr_logger.propagate = False
    if log_file:                     # van muon log cua Flower nam trong file
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        flwr_logger.addHandler(fh)
