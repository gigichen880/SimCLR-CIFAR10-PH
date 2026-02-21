"""
simclr.py (end-to-end updated)

Goal:
- Use Persistent Homology (PH) ONLY to define a similarity / neighborhood structure.
- Train the backbone with gradient descent by matching rep-similarities to PH-defined soft targets.
- We do NOT backprop through PH itself.

Methods (yaml / CLI):
  - method=baseline : standard SimCLR NT-Xent on rep
  - method=phsim    : PH-guided contrastive (PH defines soft positives; reps learn them)
  - method=hybrid   : alpha*baseline + (1-alpha)*phsim

Run examples:
  python simclr.py backbone=resnet18 method=baseline
  python simclr.py backbone=resnet18 method=phsim
  python simclr.py backbone=resnet18 method=hybrid loss.alpha=0.9

Notes:
- Uses ripser (CPU) for PH (robust on VMs); PH is detached.
- For CIFAR, we remove ResNet maxpool in stem to avoid 1x1 feature maps (PH needs >1 point).
- Saves train_history.csv, loss_curve.png, lr_curve.png into Hydra run dir.
"""

import os
import csv
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
from PIL import Image
from ripser import ripser
from models import SimCLR

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional: keep warnings quieter
import warnings
warnings.simplefilter("ignore", UserWarning)


# -------------------------
# Utilities
# -------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class HistoryLogger:
    def __init__(self, out_dir: str, filename: str = "train_history.csv"):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.csv_path = os.path.join(out_dir, filename)
        self.rows = []
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "lr"])

    def log_epoch(self, epoch: int, loss: float, lr: float):
        self.rows.append((epoch, loss, lr))
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, loss, lr])

    def plot(self):
        if not self.rows:
            return
        epochs = [r[0] for r in self.rows]
        losses = [r[1] for r in self.rows]
        lrs = [r[2] for r in self.rows]

        plt.figure()
        plt.plot(epochs, losses)
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        plt.title("Train Loss")
        plt.savefig(os.path.join(self.out_dir, "loss_curve.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, lrs)
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title("Learning Rate")
        plt.savefig(os.path.join(self.out_dir, "lr_curve.png"), dpi=150)
        plt.close()


class CIFAR10Pair(CIFAR10):
    """Generate mini-batch pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def get_color_distortion(s=0.5):
    """Color distortion = color jitter + grayscale (SimCLR appendix)."""
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([rnd_color_jitter, rnd_gray])

# -------------------------
# PH featurizer (ripser CPU; detached)
# -------------------------
class SoftPersistenceImage(nn.Module):
    """
    Diagram -> fixed vector via Gaussian bumps on (birth, persistence) grid.
    Filters inf/NaN points.
    """
    def __init__(self, grid_size=12, birth_range=(0.0, 4.0), pers_range=(0.0, 4.0), sigma=0.15):
        super().__init__()
        self.grid_size = grid_size
        self.birth_min, self.birth_max = birth_range
        self.pers_min, self.pers_max = pers_range
        self.sigma = sigma

        b = torch.linspace(self.birth_min, self.birth_max, grid_size)
        p = torch.linspace(self.pers_min, self.pers_max, grid_size)
        bb, pp = torch.meshgrid(b, p, indexing="ij")
        centers = torch.stack([bb.reshape(-1), pp.reshape(-1)], dim=1)  # (G,2)
        self.register_buffer("centers", centers)

    def forward(self, diagram_bd: torch.Tensor) -> torch.Tensor:
        if diagram_bd.numel() == 0:
            return diagram_bd.new_zeros(self.centers.shape[0])

        mask = torch.isfinite(diagram_bd).all(dim=1)
        diagram_bd = diagram_bd[mask]
        if diagram_bd.numel() == 0:
            return diagram_bd.new_zeros(self.centers.shape[0])

        birth = diagram_bd[:, 0]
        death = diagram_bd[:, 1]
        pers = (death - birth).clamp(min=0.0)

        pts = torch.stack([birth, pers], dim=1)  # (M,2)

        diff = pts[:, None, :] - self.centers[None, :, :]  # (M,G,2)
        dist2 = (diff ** 2).sum(dim=2)                     # (M,G)
        bumps = torch.exp(-dist2 / (2 * self.sigma * self.sigma))  # (M,G)

        feat = (bumps * pers[:, None]).sum(dim=0)  # (G,)
        return feat


class PHFeaturizer(nn.Module):
    """
    feature map -> point cloud -> ripser diagrams -> persistence image -> projected PH vector
    """
    def __init__(
        self,
        out_dim=128,
        num_points=25,
        pi_grid=12,
        sigma=0.15,
        birth_range=(0.0, 4.0),
        pers_range=(0.0, 4.0),
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_points = num_points

        self.pi = SoftPersistenceImage(
            grid_size=pi_grid,
            birth_range=birth_range,
            pers_range=pers_range,
            sigma=sigma,
        )
        raw_dim = pi_grid * pi_grid
        self.proj = nn.Linear(2 * raw_dim, out_dim)  # concat(H0 PI, H1 PI)

    def _to_pointcloud(self, h_map_small: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B, N, C) where N=H*W
        B, C, H, W = h_map_small.shape
        pts = h_map_small.permute(0, 2, 3, 1).reshape(B, H * W, C)
        N = H * W
        if self.num_points is not None and self.num_points < N:
            idx = torch.randperm(N, device=h_map_small.device)[: self.num_points]
            pts = pts[:, idx, :]
        return pts

    @staticmethod
    def _standardize_np(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        x = x / (x.std(axis=0, keepdims=True) + 1e-6)
        return x.astype(np.float32)

    def _vr_persistence(self, pts: torch.Tensor, hom_dim: int) -> torch.Tensor:
        pts_np = pts.detach().cpu().numpy().astype(np.float32)
        pts_np = self._standardize_np(pts_np)

        dgms = ripser(pts_np, maxdim=1)["dgms"]
        dgm = dgms[hom_dim]
        if dgm.size == 0:
            return pts.new_zeros((0, 2))
        return torch.from_numpy(dgm).to(device=pts.device, dtype=pts.dtype)

    def forward(self, h_map_small: torch.Tensor) -> torch.Tensor:
        pts_batch = self._to_pointcloud(h_map_small)  # (B,N,C)
        feats = []
        for b in range(pts_batch.size(0)):
            pts = pts_batch[b]

            d0 = self._vr_persistence(pts, hom_dim=0)
            f0 = self.pi(d0)

            d1 = self._vr_persistence(pts, hom_dim=1)
            f1 = self.pi(d1)

            feats.append(torch.cat([f0, f1], dim=0))

        feats = torch.stack(feats, dim=0)  # (B, 2*raw_dim)
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
        return self.proj(feats)            # (B, out_dim)


# -------------------------
# Losses
# -------------------------
def nt_xent(x: torch.Tensor, t=0.5) -> torch.Tensor:
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    sim = (x @ x.t()).clamp(min=1e-7)
    sim = sim / t
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e5

    targets = torch.arange(sim.size(0), device=sim.device)
    targets[::2] += 1
    targets[1::2] -= 1
    return F.cross_entropy(sim, targets.long())


def sim_matrix(x: torch.Tensor, t=0.5) -> torch.Tensor:
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    s = (x @ x.t()).clamp(min=1e-7)
    s = s / t
    s = s - torch.eye(s.size(0), device=s.device) * 1e5
    return s


def _soft_targets_from_sim(sim_logits: torch.Tensor, tau: float) -> torch.Tensor:
    """
    sim_logits: (N,N) higher means more similar. diagonal should already be masked/very negative.
    Returns row-stochastic soft target distribution.
    """
    N = sim_logits.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_logits.device)
    logits = (sim_logits / tau).masked_fill(~mask, -1e9)
    return F.softmax(logits, dim=1)


def ph_guided_contrastive(rep: torch.Tensor, ph_vec: torch.Tensor, tau_student: float, tau_ph: float) -> torch.Tensor:
    """
    PH defines soft neighbors; rep learns to match them via cross-entropy.
    rep: (N,d) differentiable
    ph_vec: (N,d') treated as constant target (no grad)
    """
    # Student logits (differentiable)
    S_rep = sim_matrix(rep, t=tau_student)

    # PH targets (no grad)
    with torch.no_grad():
        ph_vec = ph_vec / (ph_vec.norm(dim=1, keepdim=True) + 1e-8)
        S_ph = (ph_vec @ ph_vec.t()).clamp(min=1e-7)
        S_ph = S_ph - torch.eye(S_ph.size(0), device=S_ph.device) * 1e5
        P_ph = _soft_targets_from_sim(S_ph, tau=tau_ph)

    logQ = F.log_softmax(S_rep, dim=1)
    loss = -(P_ph * logQ).sum(dim=1).mean()
    return loss


# -------------------------
# Train
# -------------------------
@hydra.main(version_base=None, config_path=".", config_name="simclr_config")
def train(args: DictConfig) -> None:
    logger.info("Config:\n" + OmegaConf.to_yaml(args))

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[SimCLR] using device = {device}")
    if device == "cuda":
        cudnn.benchmark = True

    seed = int(getattr(args, "seed", 0))
    if seed:
        set_seed(seed)

    # Hydra run dir
    out_dir = os.getcwd()
    hist = HistoryLogger(out_dir)

    # Data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(s=float(args.aug.color_strength)),
        transforms.ToTensor()
    ])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10Pair(root=data_dir, train=True, transform=train_transform, download=True)

    if int(args.data.subset_size) > 0:
        train_set = Subset(train_set, range(int(args.data.subset_size)))

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True
    )

    # Model
    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = resnet18 if args.backbone == "resnet18" else resnet34

    model = SimCLR(
        base_encoder,
        projection_dim=int(args.projection_dim),
        proj_hidden_dim=int(args.model.proj_hidden_dim),
        reduce_channels=int(args.ph.reduce_channels),
        cifar_no_maxpool=True,   # IMPORTANT for PH on CIFAR
    ).to(device)

    # PH featurizer (PH is computed by ripser on CPU; projection head is learnable but PH path is detached)
    ph_featurizer = PHFeaturizer(
        out_dim=int(args.projection_dim),
        num_points=int(args.ph.num_points),
        pi_grid=int(args.ph.pi_grid),
        sigma=float(args.ph.pi_sigma),
        birth_range=(float(args.ph.birth_min), float(args.ph.birth_max)),
        pers_range=(float(args.ph.pers_min), float(args.ph.pers_max)),
    ).to(device)

    logger.info(f"Base model: {args.backbone}")
    logger.info(f"feature dim: {model.feature_dim}, projection dim: {args.projection_dim}")
    logger.info(f"method: {args.method}")

    # Optimizer: train backbone+projector AND the PH projection head
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(ph_featurizer.proj.parameters()),
        float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=True
    )

    # Scheduler
    total_steps = int(args.epochs) * len(train_loader)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, total_steps, float(args.learning_rate), float(args.optim.lr_min))
    )

    model.train()
    ph_featurizer.train()

    max_steps = int(args.train.max_steps) if int(args.train.max_steps) > 0 else None
    temperature = float(args.temperature)

    # For PH-guided: use these temps (still in config.loss for convenience)
    tau_ph = float(args.loss.teacher_temperature)     # PH soft-target sharpness
    tau_student = float(args.loss.student_temperature)  # rep logits temperature

    for epoch in range(1, int(args.epochs) + 1):
        loss_meter = AverageMeter("loss")
        bar = tqdm(train_loader, total=len(train_loader))

        for step, (x, _) in enumerate(bar):
            if max_steps is not None and step >= max_steps:
                break

            B = x.size(0)
            x = x.view(B * 2, x.size(2), x.size(3), x.size(4)).to(device, non_blocking=(device == "cuda"))

            optimizer.zero_grad()
            h_map_small, _, rep = model(x)

            method = str(args.method).lower()

            if method == "baseline":
                loss = nt_xent(rep, temperature)

            elif method == "phsim":
                # PH defines similarity; reps learn to match PH neighborhoods
                ph_vec = ph_featurizer(h_map_small)
                loss = ph_guided_contrastive(rep=rep, ph_vec=ph_vec, tau_student=tau_student, tau_ph=tau_ph)

            elif method == "hybrid":
                # baseline + PH-guided contrastive
                alpha = float(args.loss.alpha)
                loss_cos = nt_xent(rep, temperature)
                ph_vec = ph_featurizer(h_map_small)
                loss_phsim = ph_guided_contrastive(rep=rep, ph_vec=ph_vec, tau_student=tau_student, tau_ph=tau_ph)
                loss = alpha * loss_cos + (1.0 - alpha) * loss_phsim

            else:
                raise ValueError(f"Unknown method={args.method}. Use baseline|phsim|hybrid.")

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=x.size(0))
            bar.set_description(f"epoch {epoch} | loss {loss_meter.avg:.4f}")

        # Checkpoint
        if epoch >= int(args.log_interval) and epoch % int(args.log_interval) == 0:
            ckpt = {
                "model": model.state_dict(),
                "ph_featurizer": ph_featurizer.state_dict(),
                "config": OmegaConf.to_container(args, resolve=True),
            }
            path = f"simclr_{args.method}_{args.backbone}_epoch{epoch}.pt"
            logger.info(f"==> Save checkpoint: {path}")
            torch.save(ckpt, path)

        # End-of-epoch logging + plots
        current_lr = optimizer.param_groups[0]["lr"]
        hist.log_epoch(epoch, loss_meter.avg, current_lr)
        hist.plot()


if __name__ == "__main__":
    train()