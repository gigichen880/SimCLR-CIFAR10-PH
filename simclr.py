"""
simclr.py

Goal:
- Use Persistent Homology (PH) ONLY to define a similarity / neighborhood structure.
- Train the backbone with gradient descent by matching rep-similarities to PH-defined soft targets.
- We do NOT backprop through ripser / PH computation itself.

Methods (yaml / CLI):
  - method=baseline : standard SimCLR NT-Xent on rep
  - method=phsim    : PH-guided contrastive (PH defines soft positives; reps learn them)
  - method=hybrid   : alpha*baseline + (1-alpha)*phsim

Key stability improvements:
- Warmup: run baseline for first K epochs even if method=phsim/hybrid
- Top-k PH neighbors: sparsify PH targets per row (renormalize)
- Optional PH projection learning via a small similarity-alignment auxiliary loss
- Correct cosine LR scheduling with LambdaLR multiplier

Run examples:
  python simclr.py backbone=resnet18 method=baseline
  python simclr.py backbone=resnet18 method=phsim
  python simclr.py backbone=resnet18 method=hybrid loss.alpha=0.9
"""

import os
import csv
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Optional

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

import warnings
warnings.simplefilter("ignore", UserWarning)


# -------------------------
# Utilities
# -------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name: str):
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

    def plot(self, tag):
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
        plt.savefig(os.path.join(self.out_dir, f"loss_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, lrs)
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title("Learning Rate")
        plt.savefig(os.path.join(self.out_dir, f"loss_{tag}.png"), dpi=150)
        plt.close()


class CIFAR10Pair(CIFAR10):
    """Generate mini-batch pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # (2,C,H,W), y


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    """Cosine annealing schedule: returns ABSOLUTE lr."""
    if total_steps <= 1:
        return lr_min
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
        self.grid_size = int(grid_size)
        self.birth_min, self.birth_max = birth_range
        self.pers_min, self.pers_max = pers_range
        self.sigma = float(sigma)

        b = torch.linspace(self.birth_min, self.birth_max, self.grid_size)
        p = torch.linspace(self.pers_min, self.pers_max, self.grid_size)
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

        diff = pts[:, None, :] - self.centers[None, :, :]              # (M,G,2)
        dist2 = (diff ** 2).sum(dim=2)                                 # (M,G)
        bumps = torch.exp(-dist2 / (2 * self.sigma * self.sigma))      # (M,G)

        feat = (bumps * pers[:, None]).sum(dim=0)  # (G,)
        return feat


class PHFeaturizer(nn.Module):
    """
    feature map -> point cloud -> ripser diagrams -> persistence image -> projected PH vector

    NOTE: ripser is CPU + non-differentiable. We explicitly detach before ripser.
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
        self.out_dim = int(out_dim)
        self.num_points = int(num_points) if num_points is not None else None

        self.pi = SoftPersistenceImage(
            grid_size=int(pi_grid),
            birth_range=birth_range,
            pers_range=pers_range,
            sigma=float(sigma),
        )
        raw_dim = int(pi_grid) * int(pi_grid)
        self.proj = nn.Linear(2 * raw_dim, self.out_dim)  # concat(H0 PI, H1 PI)

    def _to_pointcloud(self, h_map_small: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,N,C) with N=H*W
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
        # Detach and move to CPU for ripser
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
            pts = pts_batch[b]  # (N,C)

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


def _soft_targets_from_sim(sim_logits: torch.Tensor, tau: float, topk: Optional[int] = None) -> torch.Tensor:
    """
    sim_logits: (N,N), higher means more similar. diagonal should already be masked/very negative.
    tau: temperature for targets
    topk: if not None, keep only top-k logits per row (excluding diagonal), renormalize.
    """
    N = sim_logits.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_logits.device)
    logits = (sim_logits / tau).masked_fill(~mask, -1e9)  # (N,N)

    if topk is not None and topk > 0 and topk < (N - 1):
        vals, idx = torch.topk(logits, k=topk, dim=1)
        sparse = logits.new_full(logits.shape, -1e9)
        sparse.scatter_(1, idx, vals)
        logits = sparse

    return F.softmax(logits, dim=1)


def mse_sim_alignment(rep: torch.Tensor, ph_vec: torch.Tensor) -> torch.Tensor:
    """
    Align rep similarity to PH similarity (grad flows into rep and ph_vec).
    This is the mechanism that actually trains ph_featurizer.proj without ever backproping through ripser.
    """
    rep_n = rep / (rep.norm(dim=1, keepdim=True) + 1e-8)
    ph_n  = ph_vec / (ph_vec.norm(dim=1, keepdim=True) + 1e-8)

    S_rep = rep_n @ rep_n.t()
    S_ph  = ph_n @ ph_n.t()

    N = rep.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=rep.device)
    return F.mse_loss(S_rep[mask], S_ph[mask])


def ph_guided_contrastive(
    rep: torch.Tensor,
    ph_vec: torch.Tensor,
    tau_student: float,
    tau_ph: float,
    topk: Optional[int] = None,
    learn_ph_proj: bool = True,
    align_w: float = 0.0,
) -> torch.Tensor:
    """
    PH defines soft neighbors; rep learns to match them via cross-entropy.

    rep: (N,d) differentiable
    ph_vec: (N,d') comes from PHFeaturizer; ripser part is detached by construction.
    topk: sparsify PH targets
    learn_ph_proj: if False, blocks gradients into ph_vec (and thus into ph_featurizer.proj)
    align_w: weight for similarity alignment auxiliary term (0 disables)
    """
    # Student logits (differentiable)
    S_rep = sim_matrix(rep, t=tau_student)

    if not learn_ph_proj:
        ph_vec = ph_vec.detach()

    # Targets from a detached copy (PH target neighborhoods fixed for student optimization)
    ph_vec_tgt = ph_vec.detach()
    ph_vec_tgt = ph_vec_tgt / (ph_vec_tgt.norm(dim=1, keepdim=True) + 1e-8)
    S_ph = (ph_vec_tgt @ ph_vec_tgt.t()).clamp(min=1e-7)
    S_ph = S_ph - torch.eye(S_ph.size(0), device=S_ph.device) * 1e5
    P_ph = _soft_targets_from_sim(S_ph, tau=tau_ph, topk=topk)

    logQ = F.log_softmax(S_rep, dim=1)
    loss = -(P_ph * logQ).sum(dim=1).mean()

    # Optional: make ph_featurizer.proj actually learn a meaningful geometry
    if learn_ph_proj and align_w > 0.0:
        loss = loss + float(align_w) * mse_sim_alignment(rep, ph_vec)

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

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    viz_dir = os.path.join(out_dir, "visuals")
    ensure_dir(viz_dir)
    hist = HistoryLogger(out_dir=viz_dir) 

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

    # Optimizer: train backbone+projector AND (optionally useful) PH projection head
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(ph_featurizer.proj.parameters()),
        float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=True
    )

    # Scheduler: correct LambdaLR multiplier behavior
    max_steps = int(args.train.max_steps) if int(args.train.max_steps) > 0 else None
    steps_per_epoch = min(len(train_loader), max_steps) if max_steps is not None else len(train_loader)
    total_steps = int(args.epochs) * int(steps_per_epoch)

    lr_max = float(args.learning_rate)
    lr_min = float(args.optim.lr_min)

    def lr_mult(step: int) -> float:
        lr_abs = get_lr(step, total_steps, lr_max, lr_min)
        return lr_abs / max(lr_max, 1e-12)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_mult)

    model.train()
    ph_featurizer.train()

    temperature = float(args.temperature)
    tau_ph = float(args.loss.teacher_temperature)
    tau_student = float(args.loss.student_temperature)

    warmup_epochs = int(getattr(args.train, "warmup_epochs", 0))
    topk_cfg = int(getattr(args.loss, "ph_topk", 0))
    ph_topk = topk_cfg if topk_cfg > 0 else None
    align_w = float(getattr(args.loss, "ph_proj_align", 0.0))

    for epoch in range(1, int(args.epochs) + 1):
        loss_meter = AverageMeter("loss")
        bar = tqdm(train_loader, total=steps_per_epoch)

        for step, (x, _) in enumerate(bar):
            if max_steps is not None and step >= max_steps:
                break

            B = x.size(0)
            x = x.view(B * 2, x.size(2), x.size(3), x.size(4)).to(
                device, non_blocking=(device == "cuda")
            )

            optimizer.zero_grad()
            h_map_small, _, rep = model(x)

            method = str(args.method).lower()
            # Warmup: force baseline early for stability
            if method in ["phsim", "hybrid"] and warmup_epochs > 0 and epoch <= warmup_epochs:
                method_eff = "baseline"
            else:
                method_eff = method

            if method_eff == "baseline":
                loss = nt_xent(rep, temperature)

            elif method_eff == "phsim":
                ph_vec = ph_featurizer(h_map_small)
                loss = ph_guided_contrastive(
                    rep=rep,
                    ph_vec=ph_vec,
                    tau_student=tau_student,
                    tau_ph=tau_ph,
                    topk=ph_topk,
                    learn_ph_proj=(align_w > 0.0),
                    align_w=align_w,
                )

            elif method_eff == "hybrid":
                alpha = float(args.loss.alpha)
                loss_cos = nt_xent(rep, temperature)

                ph_vec = ph_featurizer(h_map_small)
                loss_phsim = ph_guided_contrastive(
                    rep=rep,
                    ph_vec=ph_vec,
                    tau_student=tau_student,
                    tau_ph=tau_ph,
                    topk=ph_topk,
                    learn_ph_proj=(align_w > 0.0),
                    align_w=align_w,
                )
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
            ckpt_name = f"simclr_{args.method}_{args.backbone}_epoch{epoch}.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            logger.info(f"==> Save checkpoint: {ckpt_path}")
            torch.save(ckpt, ckpt_path)

        # End-of-epoch logging + plots
        current_lr = optimizer.param_groups[0]["lr"]
        tag = f"{args.method}_{args.backbone}"
        if int(args.seed) != 0:
            tag += f"_seed{args.seed}"
        hist.log_epoch(epoch, loss_meter.avg, current_lr)
        hist.plot(tag=tag)


if __name__ == "__main__":
    train()