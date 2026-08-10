"""CNN1D model cho IoV IDS (CICIoV: 31 features, 13 classes).

Giu NGUYEN kien truc cua FedLiTeCAN/model_cnn1d.py de 4 baseline + AFSIC-IoV
so sanh cong bang tren cung mot backbone.

  input (B, 31) -> (B, 1, 31) -> 3 khoi Conv1d -> GAP -> FC -> logits
"""
import torch
import torch.nn as nn

NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31


class CNN1D_IDS(nn.Module):
    def __init__(self, input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES, dropout=0.15):
        super().__init__()
        self.input_len = input_len
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # Block 1: (B,1,31) -> (B,32,15)
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2: (B,32,15) -> (B,64,7)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: (B,64,7) -> (B,128,1)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def embed(self, x):
        """Tra ve dac trung 128 chieu (dung cho knowledge distillation)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.features(x).squeeze(-1)

    def forward(self, x):
        # x: (B, 31) float32
        return self.classifier(self.embed(x))


class FocalLoss(nn.Module):
    """Focal loss voi alpha = sqrt-inverse class frequency."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


if __name__ == "__main__":
    m = CNN1D_IDS()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")
    print("Output shape:", m(torch.randn(4, INPUT_LEN)).shape)
