import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, h_dim)

    def forward(self, x):
        return F.relu(self.l1(x))


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.l1(x))


class PreTrainingHead(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.l2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)


class CL(nn.Module):
    def __init__(self, in_dim, h_dim) -> None:
        super().__init__()
        self.x_loss_fn = torch.nn.MSELoss()
        self.encoder = Encoder(in_dim, h_dim)
        self.decoder = Decoder(h_dim, in_dim)
        self.projector = PreTrainingHead(h_dim, h_dim)

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        x_p = self.decoder(h_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return z_i, z_j, self.x_loss_fn(x_p, x_i)


class MaskEstimator(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.l1(x))


class FeatureEstimator(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.l1(x))


class VIMESelf(nn.Module):
    def __init__(self, in_dim, h_dim) -> None:
        super().__init__()
        self.m_loss_fn = torch.nn.BCELoss()
        self.x_loss_fn = torch.nn.MSELoss()

        self.encoder = Encoder(in_dim, h_dim)
        self.me = MaskEstimator(h_dim, in_dim)
        self.fe = FeatureEstimator(h_dim, in_dim)

    def forward(self, x, mask):
        h = self.encoder(x)
        x_p = self.fe(h)
        mask_p = self.me(h)
        return self.m_loss_fn(mask_p, mask), self.x_loss_fn(x_p, x)


class UnlabeledLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.mean(torch.var(x, 0))


class VIMESemi(nn.Module):
    def __init__(self, h_dim, l_dim) -> None:
        super().__init__()

        self.l1 = nn.Linear(h_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, l_dim)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


class PassNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
