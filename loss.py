import torch
import math


def _ciede2000(lab1, lab2, kL=1, kC=1, kH=1, eps=1e-8):
    """
    Vectorised CIEDE2000.
    lab1, lab2: (B, 3, H, W) or (B, 3)
    Returns delta-E00 per pixel with shape (B, H, W) or (B,)
    """
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + eps)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + eps)
    C_mean = 0.5 * (C1 + C2)

    G = 0.5 * (1 - torch.sqrt((C_mean ** 7) / (C_mean ** 7 + 25 ** 7) + eps))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2

    C1p = torch.sqrt(a1p ** 2 + b1 ** 2 + eps)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2 + eps)
    Cp_mean = 0.5 * (C1p + C2p)

    h1p = torch.atan2(b1, a1p) % (2 * math.pi)
    h2p = torch.atan2(b2, a2p) % (2 * math.pi)

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = dhp - 2 * math.pi * torch.round(dhp / (2 * math.pi))
    dHp = 2 * torch.sqrt(C1p * C2p + eps) * torch.sin(dhp / 2)

    Lp_mean = 0.5 * (L1 + L2)
    hp_sum = h1p + h2p
    hp_mean = hp_sum / 2
    hp_mean = hp_mean - math.pi * ((h1p - h2p).abs() > math.pi).float()
    Hp_mean = hp_mean + 2 * math.pi * (hp_mean < 0).float()

    T = (1
         - 0.17 * torch.cos(Hp_mean - math.radians(30))
         + 0.24 * torch.cos(2 * Hp_mean)
         + 0.32 * torch.cos(3 * Hp_mean + math.radians(6))
         - 0.20 * torch.cos(4 * Hp_mean - math.radians(63)))

    Sl = 1 + (0.015 * (Lp_mean - 50) ** 2) / torch.sqrt(20 + (Lp_mean - 50) ** 2 + eps)
    Sc = 1 + 0.045 * Cp_mean
    Sh = 1 + 0.015 * Cp_mean * T

    Rt = -2 * torch.sqrt((Cp_mean ** 7) / (Cp_mean ** 7 + 25 ** 7) + eps) * torch.sin(
        math.radians(60) * torch.exp(-((Hp_mean - math.radians(275)) / math.radians(25)) ** 2))

    dE = torch.sqrt(
        (dLp / (kL * Sl)) ** 2 +
        (dCp / (kC * Sc)) ** 2 +
        (dHp / (kH * Sh)) ** 2 +
        Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)) + eps
    )
    return dE


class PBCLoss(torch.nn.Module):
    """
    Perceptual Balanced Color Loss.

    Combines CIEDE2000 (delta-E), chroma-weighted (a*, b*) error, and L* error.
    Chroma weighting emphasizes high-saturation regions.
    """
    def __init__(self, beta=2.0, gamma=2.0,
                 lambda_de=1.0, lambda_ab=0.5, lambda_l=0.2,
                 reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.beta = beta
        self.gamma = gamma
        self.lambda_de = lambda_de
        self.lambda_ab = lambda_ab
        self.lambda_l = lambda_l
        self.reduction = reduction

    def forward(self, pred_lab, target_lab):
        de = _ciede2000(pred_lab, target_lab)

        a_p, b_p = pred_lab[:, 1], pred_lab[:, 2]
        a_t, b_t = target_lab[:, 1], target_lab[:, 2]
        chroma = torch.sqrt(a_t ** 2 + b_t ** 2)
        w = 1.0 + self.beta * (chroma / 128.0) ** self.gamma
        ab_err = w * ((a_p - a_t) ** 2 + (b_p - b_t) ** 2)

        l_err = (pred_lab[:, 0] - target_lab[:, 0]) ** 2

        loss_map = self.lambda_de * de + self.lambda_ab * ab_err + self.lambda_l * l_err

        if self.reduction == 'none':
            return loss_map
        elif self.reduction == 'sum':
            return loss_map.sum()
        return loss_map.mean()
