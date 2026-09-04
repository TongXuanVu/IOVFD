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
import glob
import logging
import os
import sys
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


# Bo IoV va bo IoT long thu muc KHAC NHAU:
#   IoV: <root>/federated_data{,_fewshot,_10shot}/
#   IoT: <root>/100client/                              (full)
#        <root>/iot100client_fewshot/federated_data_fewshot/
#        <root>/iot100client_fewshot/federated_data_10shot/
# Ham nay tra ve duong dan TUONG DOI so voi data_dir cua thu muc chua shard.
BO_CUC_IOT = {
    "federated_data":          ["100client", "federated_data"],
    "federated_data_fewshot":  [os.path.join("iot100client_fewshot",
                                             "federated_data_fewshot")],
    "federated_data_10shot":   [os.path.join("iot100client_fewshot",
                                             "federated_data_10shot")],
}


def tim_fed_subdir(data_dir, ten):
    """Tim thu muc shard that su ton tai, thu ca hai bo cuc.

    Tra ve duong dan tuong doi; neu khong thay thi tra lai `ten` de loi bao o
    cho nap du lieu (co ten file cu the) chu khong im lang.
    """
    ung_vien = [ten] + BO_CUC_IOT.get(ten, [])
    for u in ung_vien:
        d = os.path.join(data_dir, u)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "client_*.pt")):
            if u != ten:
                logger.info(f"Bo cuc IoT: '{ten}' -> '{u}'")
            return u
    return ten


# --- ho so bo du lieu -------------------------------------------------------
# Mac dinh la CICIoV. Goi init_dataset() truoc khi dung de TU DO theo du lieu
# that; luc do bon bien duoi day bi ghi de. Cac noi dung phai doc qua
# `common.<TEN>` (tra cuu luc chay) chu KHONG duoc `from common import <TEN>`
# (chot gia tri luc import, mutation khong lan toi).
NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31
NUM_TASKS = 5
TASK_INCREMENTS = [3, 3, 3, 2, 2]          # giong AFSIC-IoV / FedLiTeCAN

TEN_BO = "iov"          # "iov" | "iot"
_LABEL_LUT = None       # np.ndarray: nhan goc -> nhan tuan tu, hoac None
TASK_LABELS = None      # list[list[int]]: nhan GOC cua tung task (chi bo IoT)


def _doc_task_mapping(data_dir, fed_subdir=None):
    """Doc task_mapping_label_ids.json: list[list[int]] nhan goc theo tung task.

    Bo IoV KHONG co file nay (nhan da tuan tu 0..12 san). Bo IoT co, va thu tu
    task phi tuan tu nen bat buoc phai remap.
    """
    if fed_subdir is None: fed_subdir = FED_SUBDIR
    for p in (os.path.join(data_dir, "task_mapping_label_ids.json"),
              os.path.join(data_dir, fed_subdir, "task_mapping_label_ids.json"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "task_mapping_label_ids.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list) and d and isinstance(d[0], list):
                return d, p
    return None, None


def _do_so_dac_trung(data_dir, fed_subdir):
    """Lay so cot tu MOT shard bat ky — khong doc ca file, chi lay shape."""
    for goc in (os.path.join(data_dir, fed_subdir), data_dir):
        for p in sorted(glob.glob(os.path.join(goc, "client_*.pt")))[:1]:
            try:
                x, _ = _read_pt(p)
                return int(x.shape[1]), p
            except Exception as e:
                logger.warning(f"Khong doc duoc {p}: {e}")
    t = os.path.join(data_dir, "global_test_data.pt")
    t2 = os.path.join(data_dir, fed_subdir, "global_test_data.pt")
    if os.path.exists(t2):
        t = t2
    if os.path.exists(t):
        x, _ = _read_pt(t)
        return int(x.shape[1]), t
    return None, None


def init_dataset(data_dir, fed_subdir=None):
    """Tu do ho so bo du lieu tu chinh du lieu. Goi TRUOC moi thu khac.

    - so dac trung: lay tu shape cua mot shard (khong hardcode)
    - so lop / so task / remap: tu task_mapping_label_ids.json neu co

    CAI BAY DA DUOC CHAN: file mapping cua IoT chua du id 0..33, nen neu ap
    nham len du lieu IoV (nhan 0..12) thi MOI nhan deu tra cuu duoc va bi doi
    am tham, khong mot loi nao duoc nem ra. Nen o day chi bat remap khi nhan
    THUC TE vuot qua pham vi cua bo IoV.
    """
    global NUM_GLOBAL_CLASSES, INPUT_LEN, NUM_TASKS, TASK_INCREMENTS
    global TEN_BO, _LABEL_LUT, TASK_LABELS, FED_SUBDIR

    if fed_subdir is None:
        fed_subdir = FED_SUBDIR
    fed_subdir = tim_fed_subdir(data_dir, fed_subdir)
    FED_SUBDIR = fed_subdir
    n_feat, nguon = _do_so_dac_trung(data_dir, fed_subdir)

    y_max = -1
    t = os.path.join(data_dir, "global_test_data.pt")
    t2 = os.path.join(data_dir, fed_subdir, "global_test_data.pt")
    if os.path.exists(t2):
        t = t2
    if os.path.exists(t):
        _, yy = _read_pt(t)
        y_max = int(np.asarray(yy).max())

    mapping, map_file = _doc_task_mapping(data_dir, fed_subdir)
    dung_remap = mapping is not None and y_max >= 13

    if dung_remap:
        TASK_LABELS = mapping
        TASK_INCREMENTS = [len(t_) for t_ in mapping]
        NUM_TASKS = len(mapping)
        phang = [c for t_ in mapping for c in t_]
        NUM_GLOBAL_CLASSES = len(phang)
        lut = np.full(max(phang) + 1, -1, dtype=np.int64)
        for moi, goc in enumerate(phang):
            lut[goc] = moi
        _LABEL_LUT = lut
        TEN_BO = "iot"
    else:
        TASK_LABELS, _LABEL_LUT, TEN_BO = None, None, "iov"
        if mapping is not None:
            logger.warning(
                f"Co {map_file} nhung nhan lon nhat trong tap test chi la "
                f"{y_max} (< 13) -> KHONG remap. Ap bang cua IoT len du lieu "
                f"IoV se doi nhan am tham vi id 0..12 deu nam trong bang.")

    if n_feat:
        INPUT_LEN = n_feat

    n_mod = _dong_bo_module()
    logger.info(
        f"Ho so bo du lieu: {TEN_BO} | {NUM_GLOBAL_CLASSES} lop | "
        f"{NUM_TASKS} task {TASK_INCREMENTS} | {INPUT_LEN} dac trung"
        + (f" (do tu {os.path.basename(nguon)})" if nguon else " (mac dinh)")
        + (f" | remap nhan theo {os.path.basename(map_file)}" if dung_remap else "")
        + f" | dong bo {n_mod} bien qua cac module")
    return profile_hien_tai()


def _dong_bo_module():
    """Ghi gia tri vua do duoc vao MOI module cua repo da import.

    Can thiet vi rat nhieu file lam `from common import NUM_GLOBAL_CLASSES`
    hoac `from model_cnn1d import INPUT_LEN` — kieu import nay CHOT gia tri
    ngay luc import, nen sua bien o common sau do khong lan toi. Duyet
    sys.modules va ghi de attribute cung ten, chi trong pham vi thu muc repo.
    """
    goc = os.path.dirname(os.path.abspath(__file__))
    ten = ("NUM_GLOBAL_CLASSES", "INPUT_LEN", "NUM_TASKS", "TASK_INCREMENTS")
    gia_tri = (NUM_GLOBAL_CLASSES, INPUT_LEN, NUM_TASKS, TASK_INCREMENTS)
    n = 0
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or os.path.dirname(os.path.abspath(f)) != goc:
            continue
        for k, v in zip(ten, gia_tri):
            if hasattr(mod, k):
                setattr(mod, k, v)
                n += 1
    return n


def profile_hien_tai():
    """Ho so gon nhe de chuyen sang tien trinh con (Ray actor)."""
    return dict(ten=TEN_BO, n_classes=NUM_GLOBAL_CLASSES, n_tasks=NUM_TASKS,
                increments=list(TASK_INCREMENTS), n_features=INPUT_LEN,
                task_labels=TASK_LABELS, remap=_LABEL_LUT is not None,
                fed_subdir=FED_SUBDIR)


def apply_profile(d):
    """Ap ho so da tinh san — KHONG doc dia.

    Ray tao actor trong tien trinh RIENG, o do common.py duoc import lai voi
    gia tri mac dinh cua CICIoV. Neu khong goi ham nay trong worker, client se
    dung model 13 lop tren du lieu 34 lop ma khong bao loi ro rang.
    Goi init_dataset() trong worker thi dung, nhung phai doc mot shard moi lan
    tao client (100 client x 150 round = 15000 lan doc dia thua).
    """
    global NUM_GLOBAL_CLASSES, INPUT_LEN, NUM_TASKS, TASK_INCREMENTS
    global TEN_BO, _LABEL_LUT, TASK_LABELS, FED_SUBDIR
    if not d:
        return
    NUM_GLOBAL_CLASSES = d["n_classes"]
    INPUT_LEN = d["n_features"]
    NUM_TASKS = d["n_tasks"]
    TASK_INCREMENTS = list(d["increments"])
    TEN_BO = d.get("ten", "iov")
    TASK_LABELS = d.get("task_labels")
    if d.get("fed_subdir"):
        FED_SUBDIR = d["fed_subdir"]
    if d.get("remap") and TASK_LABELS:
        phang = [c for t_ in TASK_LABELS for c in t_]
        lut = np.full(max(phang) + 1, -1, dtype=np.int64)
        for moi, goc in enumerate(phang):
            lut[goc] = moi
        _LABEL_LUT = lut
    else:
        _LABEL_LUT = None
    _dong_bo_module()


def remap_labels(y):
    """Nhan goc -> nhan tuan tu theo thu tu task. No-op voi bo da tuan tu."""
    if _LABEL_LUT is None:
        return y
    y = np.asarray(y)
    if y.size and int(y.max()) >= len(_LABEL_LUT):
        raise ValueError(f"Nhan {int(y.max())} vuot ngoai bang remap "
                         f"({len(_LABEL_LUT)} muc)")
    out = _LABEL_LUT[y.astype(np.int64)]
    if (out < 0).any():
        raise ValueError(f"Nhan {sorted(set(y[out < 0].tolist()))} khong co "
                         f"trong task_mapping_label_ids.json")
    return out

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
            i = int(idx)
            if i < len(names):
                names[i] = name
        names = [n if n is not None else f"class_{i}" for i, n in enumerate(names)]
        if _LABEL_LUT is not None:
            # class_mapping.json danh so theo nhan GOC; sau remap thu tu lop da
            # doi, khong sap lai thi confusion matrix gan sai ten cho moi o.
            phang = [c for t_ in (TASK_LABELS or []) for c in t_]
            names = [names[g] if g < len(names) else f"class_{g}" for g in phang]
        return names
    if _LABEL_LUT is not None:
        # Ten mac dinh duoi day la cua CICIoV — dung cho bo khac la sai ten.
        return [f"class_{i}" for i in range(NUM_GLOBAL_CLASSES)]
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

    ys = [remap_labels(yi) for yi in ys]        # bo IoT: nhan goc -> tuan tu
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
    path2 = os.path.join(data_dir, FED_SUBDIR, "global_test_data.pt")
    if os.path.exists(path2) and not os.path.exists(path):
        path = path2
    logger.info(f"Nap global test: {path}")
    x, y = _read_pt(path)
    logger.info(f"Global test goc: n={len(y)}")
    y = remap_labels(y)          # phai remap TRUOC khi loc `y < n_cls`
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


def make_focal_alpha(y: np.ndarray, num_classes: int = None):
    """alpha = sqrt(N / n_c) — giong FedLiTeCAN.

    num_classes=None -> lay NUM_GLOBAL_CLASSES LUC GOI. Truoc day day la
    gia tri mac dinh cua tham so, ma Python chot no ngay luc `def` — nen
    doi bo du lieu xong van sinh ra vector alpha 13 phan tu cho model 34
    lop, va torch nem loi 'weight tensor should be defined either for all
    34 classes or no classes'.
    """
    if num_classes is None:
        num_classes = NUM_GLOBAL_CLASSES
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
