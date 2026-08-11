"""P3 - IoVFD: bo sinh du lieu gia (data-free generator) + nhieu tu he hon loan.

Bai bao: Cao, Di, Jin, "IoVFD: ... dual knowledge distillation ...",
Future Generation Computer Systems 183 (2026) 108588.

Cai theo DUNG Eq. (9)-(20) cua bai. Ba diem ban truoc lam khac han:

  1. Bai KHONG dua vector nhieu z rieng vao generator. Nhieu duoc gap vao
     chinh cai NHAN: Eq. (10) tao y* = (1-z, z) roi Eq. (13) x~ = G(y*).
     Ban truoc: G(concat(z_100chieu, Embedding(y))) -- khong co trong bai.
  2. Kien truc: Eq. (14)-(16) la Linear -> 3 khoi (MultiHead + Conv1D) ->
     Linear. Ban truoc la MLP 3 lop, khong co attention lan conv.
  3. Eq. (20): L_G = L_kl_same + L_dis. KHONG co so hang doi khang.
     Ban truoc tru them beta_adv * KL(global || ensemble) -- day la loss cua
     ho DFAD/DENSE, khong phai cua bai nay.

MOT LOI TRONG BAI, phai ghi trong bao cao
-----------------------------------------
Eq. (12) viet   z_i = (1/sqrt(2*pi)) * exp(-u_i^2 / 2)
va goi day la "standard Gaussian noise sampling". Nhung do la HAM MAT DO
Gauss, khong phai phep lay mau Gauss. Voi u_i in (0,1) tu anh xa logistic,
ket qua z_i luon nam trong [0.2420, 0.3989] -- khong he co trung binh 0,
phuong sai 1, va khong bao gio am.

Dat vao Eq. (10) thi y* = (1-z, z) voi z in [0.24, 0.40], tuc la nhan mem
kieu label smoothing. Xet nhu vay thi CO LY, chi la ten goi sai. Nen mac
dinh o day chay dung Eq. (12) (--noise-mode paper); --noise-mode boxmuller
cho ban Gauss that de doi chung.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31


# ----------------------------------------------------------------------------
# Nhieu tu he hon loan  (Eq. 11, 12)
# ----------------------------------------------------------------------------
class ChaoticNoise:
    """Day hon loan logistic  u <- r*u*(1-u),  bai dung r=4, u0=0.01.

    Tai r=4 quy dao co dang giai tich u_n = sin^2(2^n * arcsin(sqrt(u_0))),
    goc bi nhan doi moi buoc nen float64 het bit man de sau ~52 buoc. Diem do
    KHONG lam day chet: da chay thu 200000 buoc voi r=4, u0=0.01 — day van
    song, trung binh 0.4989 (ly thuyet 0.5). Sau ~52 buoc no khong con la quy
    dao that cua u0 nua, nhung van la mot day tat dinh, phan bo dung.

    (Toi tung viet o day rang r=4 lam day sup do ve 0. Do la phong doan, do
    lai thi sai — r=3.99 cua ban cu con lech xa hon: trung binh 0.5315.)

    Canh chung ben duoi giu lai lam bao hiem re tien, thuc te chua kich hoat
    lan nao; `n_reseed` dem so lan phai dung toi.
    """

    def __init__(self, dim, r=4.0, u0=0.01, device="cpu", burn_in=200,
                 mode="paper"):
        self.dim = dim
        self.r = r
        self.device = device
        self.mode = mode
        self.n_reseed = 0
        # moi chieu mot quy dao rieng, lech nhau chut de khong dong pha
        s = torch.full((dim,), float(u0), dtype=torch.float64)
        s = s + torch.arange(dim, dtype=torch.float64) * 1e-4
        self.state = (s % 1.0).clamp(1e-6, 1.0 - 1e-6).to(device)
        self._u0 = float(u0)
        for _ in range(burn_in):
            self._step()

    def _step(self):
        self.state = self.r * self.state * (1.0 - self.state)
        # chan sup do: r=4 lam quy dao roi ve 0 sau vai chuc buoc trong float64
        xau = (self.state < 1e-9) | (self.state > 1.0 - 1e-9)
        if bool(xau.any()):
            self.n_reseed += int(xau.sum())
            lam_moi = torch.rand(int(xau.sum()), dtype=self.state.dtype,
                                 device=self.state.device) * 0.98 + 0.01
            self.state = self.state.masked_scatter(xau, lam_moi)
        self.state = self.state.clamp(1e-9, 1.0 - 1e-9)
        return self.state

    def uniform(self, batch):
        return torch.stack([self._step().clone() for _ in range(batch)]).float()

    def gaussian(self, batch):
        """Eq. (12) neu mode='paper'; Box-Muller that neu mode='boxmuller'."""
        u = self.uniform(batch)
        if self.mode == "paper":
            # z = (1/sqrt(2pi)) exp(-u^2/2)  -> luon trong [0.2420, 0.3989]
            return (torch.exp(-(u ** 2) / 2.0) / math.sqrt(2.0 * math.pi)).to(
                self.device)
        u2 = self.uniform(batch)
        z = torch.sqrt(-2.0 * torch.log(u.clamp_min(1e-7))) * \
            torch.cos(2.0 * torch.pi * u2)
        return z.to(self.device)


# ----------------------------------------------------------------------------
# Nhan mem co nhieu  (Eq. 9, 10)
# ----------------------------------------------------------------------------
def sample_labels(batch, label_dist, device):
    """Eq. (9): lay nhan theo TAN SUAT NHAN cua du lieu client, khong lay deu.

    label_dist: vector xac suat (K,) do server gop tu label_counts cac client.
    """
    if label_dist is None:
        return torch.randint(0, NUM_GLOBAL_CLASSES, (batch,), device=device)
    p = torch.as_tensor(label_dist, dtype=torch.float, device=device)
    p = p.clamp_min(0)
    if float(p.sum()) <= 0:
        return torch.randint(0, len(p), (batch,), device=device)
    return torch.multinomial(p / p.sum(), batch, replacement=True)


def noisy_onehot(y, z, num_classes=NUM_GLOBAL_CLASSES):
    """Eq. (10): y* = (1-z, z) hoac (z, 1-z).

    Bai la bai toan NHI PHAN (normal/abnormal). CICIoV co 13 lop nen phai mo
    rong: lop dung giu khoi luong (1-z), phan z con lai chia deu cho K-1 lop
    kia. Voi K=2 cong thuc nay TRUNG y nguyen Eq. (10). Day la lua chon cai
    dat, phai ghi trong bao cao.
    """
    if z.dim() == 2:
        z = z[:, 0]
    z = z.view(-1, 1).clamp(0.0, 1.0 - 1e-6)
    ys = F.one_hot(y, num_classes).float()
    return ys * (1.0 - z) + (1.0 - ys) * (z / max(num_classes - 1, 1))


# ----------------------------------------------------------------------------
# Generator  (Eq. 13-16, Fig. 4)
# ----------------------------------------------------------------------------
class DecodeBlock(nn.Module):
    """Eq. (15): ST = conv1d(MultiHead(E_y)). Mot khoi giai ma dac trung khong gian."""

    def __init__(self, channels, heads=4, kernel=3):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.norm2 = nn.LayerNorm(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, h):                       # h: (B, L, C)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        c = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.norm2(h + self.act(c))


class Generator(nn.Module):
    """y* -> x~. Eq. (14) Linear, Eq. (15) x3 khoi, Eq. (16) Linear.

    Bai KHONG cho so chieu d, so khoi attention head, hay kich thuoc kernel.
    Mac dinh o day: seq_len=8, channels=32 (d=256), 4 head, kernel 3 — la lua
    chon cai dat. So khoi = 3 thi dung bai ("three stacked ... blocks").
    """

    def __init__(self, num_classes=NUM_GLOBAL_CLASSES, out_dim=INPUT_LEN,
                 seq_len=8, channels=32, heads=4, n_blocks=3):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.channels = channels
        self.embed = nn.Linear(num_classes, seq_len * channels)      # Eq. (14)
        self.blocks = nn.ModuleList(                                  # Eq. (15)
            [DecodeBlock(channels, heads) for _ in range(n_blocks)])
        self.head = nn.Linear(seq_len * channels, out_dim)            # Eq. (16)
        self.out_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, y_star):
        h = self.embed(y_star).view(-1, self.seq_len, self.channels)
        for b in self.blocks:
            h = b(h)
        return self.out_bn(self.head(h.flatten(1)))


# ----------------------------------------------------------------------------
def soft_target(logits, T):
    """Eq. (22): softmax(logits / T)."""
    return F.softmax(logits / T, dim=1)


def kl_divergence(student_logits, teacher_logits, T=3.0):
    """KL(teacher || student) co lam mem nhiet do T (Eq. 25, 28)."""
    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def kl_to_soft_label(logits, y_star):
    """Eq. (18): KL(sigma(Output_LM), y*) — khop du doan ensemble voi nhan MEM.

    Khac cross_entropy(logits, y_hard) o ban truoc: dich la ca vector y* co
    nhieu, khong phai nhan cung.
    """
    return F.kl_div(F.log_softmax(logits, dim=1), y_star, reduction="batchmean")


def diversity_loss(x, y_star):
    """Eq. (19): L_dis = exp( -(1/Q^2) * SUM_ij ||x~_i - x~_j|| * ||y*_i - y*_j|| ).

    Cang nho cang tot => mau co nhan khac nhau phai nam xa nhau.
    Ban truoc dung -(d_x / d_z).mean() — dang khac han, va chia cho khoang
    cach NHIEU chu khong nhan voi khoang cach NHAN.

    Chu thich OCR: ban PDF cho ra "exp^Q x 1/Q SUM(...)"; he so chuan hoa
    khong doc duoc chac chan. O day lay trung binh tren Q^2 cap.
    """
    d_x = torch.cdist(x, x, p=2)
    d_y = torch.cdist(y_star, y_star, p=2)
    return torch.exp(-(d_x * d_y).mean())


def student_ce(logits, y, T=1.0):
    """Eq. (24)/(27): cross-entropy giua soft target o T=1 va nhan THAT."""
    return F.cross_entropy(logits / T, y)


def ensemble_logits(models, x, weights=None):
    """Eq. (17)/(23): Output_LM = (1/n) SUM_i M_i(x~).

    Bai lay trung binh KHONG trong so (1/n). weights=None thi dung dung bai;
    truyen weights vao se thanh trung binh co trong so (lech bai).

    KHONG bao no_grad: buoc huan luyen generator can gradient chay nguoc qua x
    ve toi G. Tham so cac model client da requires_grad_(False) o server.
    """
    outs = []
    for m in models:
        m.eval()
        outs.append(m(x))
    stack = torch.stack(outs)                    # (K, B, num_classes)
    if weights is None:
        return stack.mean(0)
    w = torch.as_tensor(weights, dtype=stack.dtype, device=stack.device)
    return (stack * (w / w.sum()).view(-1, 1, 1)).sum(0)


def onehot_loss(logits, y):                      # giu ten cu cho tuong thich
    return F.cross_entropy(logits, y)


if __name__ == "__main__":
    noise = ChaoticNoise(NUM_GLOBAL_CLASSES, r=4.0, u0=0.01)
    z = noise.gaussian(8)
    print(f"Eq.12 z: min={z.min():.4f} max={z.max():.4f} "
          f"(ky vong [0.2420, 0.3989]) | gieo lai {noise.n_reseed} lan")
    y = torch.randint(0, NUM_GLOBAL_CLASSES, (8,))
    ys = noisy_onehot(y, z)
    print(f"y* shape={tuple(ys.shape)} tong moi dong={ys.sum(1)[0]:.4f}")
    g = Generator()
    print("mau sinh ra:", tuple(g(ys).shape))
    print(f"Generator params: {sum(p.numel() for p in g.parameters()):,}")
