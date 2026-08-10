"""P3 - IoVFD: bo sinh du lieu gia (data-free generator) + nhieu tu he hon loan.

Bai bao: Cao, Di, Jin, "IoVFD: ... dual knowledge distillation ...",
Future Generation Computer Systems 183 (2026).

Hai thanh phan:
  1. ChaoticNoise -- bai bao lay nhieu Gauss tu MOT HE HON LOAN thay vi tu
     torch.randn. Cai o day bang anh xa logistic  x <- r*x*(1-x)  (r=3.99, che
     do hon loan) de sinh day uniform, roi Box-Muller doi sang Gauss. Uu diem
     bai bao neu: tat dinh (tai lap duoc tu 1 seed) nhung phan bo phu rong hon
     PRNG thong thuong -> pseudo-data da dang hon.
  2. Generator -- z (nhieu) + nhan y  ->  vector dac trung gia (B, 31).

Ba loss cua generator (dung o server_iov.py):
  L_oh   : ensemble phai TU TIN dung lop y da chon      (cross-entropy one-hot)
  L_div  : cac mau sinh ra phai KHAC NHAU               (diversity)
  L_adv  : global va ensemble phai BAT DONG             (-KL, adversarial)
  L_G = L_oh + beta_div * L_div + beta_adv * L_adv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31


class ChaoticNoise:
    """Sinh nhieu Gauss tat dinh tu anh xa logistic (he hon loan 1 chieu)."""

    def __init__(self, dim, r=3.99, seed=0.4142135, device="cpu", burn_in=200):
        self.dim = dim
        self.r = r
        self.device = device
        # trang thai: mot vector, moi phan tu la mot quy dao rieng
        s = torch.linspace(0.1, 0.9, dim, dtype=torch.float64)
        s = (s + seed) % 1.0
        s[s <= 0] += 0.5
        self.state = s.to(device)
        for _ in range(burn_in):                 # bo qua doan qua do
            self._step()

    def _step(self):
        self.state = self.r * self.state * (1.0 - self.state)
        self.state = self.state.clamp(1e-9, 1.0 - 1e-9)
        return self.state

    def uniform(self, batch):
        rows = [self._step().clone() for _ in range(batch)]
        return torch.stack(rows).float()

    def gaussian(self, batch):
        """Box-Muller tren hai day uniform lien tiep -> N(0,1), shape (batch, dim)."""
        u1 = self.uniform(batch).clamp_min(1e-7)
        u2 = self.uniform(batch)
        z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * torch.pi * u2)
        return z.to(self.device)


class Generator(nn.Module):
    """z + nhan -> vector dac trung gia cung dinh dang voi mau CICIoV that."""

    def __init__(self, noise_dim=100, num_classes=NUM_GLOBAL_CLASSES,
                 out_dim=INPUT_LEN, hidden=256, embed_dim=32):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, hidden),
            nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden * 2, out_dim),
        )
        self.out_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, z, y):
        h = torch.cat([z, self.label_emb(y)], dim=1)
        return self.out_bn(self.net(h))          # chuan hoa cho khop thong ke dau vao


# ----------------------------------------------------------------------------
# Cac ham loss
# ----------------------------------------------------------------------------
def kl_divergence(student_logits, teacher_logits, T=3.0):
    """KL(teacher || student) co lam mem nhiet do T."""
    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def diversity_loss(samples, z):
    """Khuyen khich mau khac nhau: khoang cach mau / khoang cach nhieu."""
    d_s = torch.cdist(samples, samples, p=2)
    d_z = torch.cdist(z, z, p=2)
    return -(d_s / (d_z + 1e-6)).mean()


def onehot_loss(logits, y):
    return F.cross_entropy(logits, y)


def ensemble_logits(models, x, weights=None):
    """Trung binh logits cua cac model client (co trong so theo so mau).

    KHONG bao no_grad: buoc huan luyen generator can gradient chay nguoc qua x
    ve toi G. Tham so cua cac model client da duoc requires_grad_(False) o
    server nen khong bi cap nhat.
    """
    outs = []
    for m in models:
        m.eval()
        outs.append(m(x))
    stack = torch.stack(outs)                    # (K, B, num_classes)
    if weights is None:
        return stack.mean(0)
    w = torch.as_tensor(weights, dtype=stack.dtype, device=stack.device)
    w = (w / w.sum()).view(-1, 1, 1)
    return (stack * w).sum(0)


if __name__ == "__main__":
    noise = ChaoticNoise(100, device="cpu")
    z = noise.gaussian(8)
    print(f"nhieu hon loan: shape={tuple(z.shape)} mean={z.mean():.3f} std={z.std():.3f}")
    g = Generator()
    y = torch.randint(0, NUM_GLOBAL_CLASSES, (8,))
    print("mau sinh ra:", tuple(g(z, y).shape))
    print(f"Generator params: {sum(p.numel() for p in g.parameters()):,}")
