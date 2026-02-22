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
from persim import wasserstein

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
class PHDiagramFeaturizer(nn.Module):
    """
    feature map -> point cloud -> ripser diagrams (H0/H1)
    Returns diagrams only (no vectorization).
    """
    def __init__(self, num_points=25, max_pts_per_dgm=64):
        super().__init__()
        self.num_points = int(num_points) if num_points is not None else None
        self.max_pts_per_dgm = int(max_pts_per_dgm)

    def _to_pointcloud(self, h_map_small: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,N,C)
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

    @staticmethod
    def _sanitize_dgm_np(dgm: np.ndarray) -> np.ndarray:
        """
        Keep only finite points with death > birth.
        """
        if dgm is None or dgm.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        dgm = dgm.astype(np.float32, copy=False)
        mask = np.isfinite(dgm).all(axis=1)
        dgm = dgm[mask]
        if dgm.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        # remove zero/negative persistence points (optional but usually stabilizes)
        pers = dgm[:, 1] - dgm[:, 0]
        dgm = dgm[pers > 1e-8]
        if dgm.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return dgm

    def _cap_points(self, dgm: np.ndarray) -> np.ndarray:
        """
        Keep top-K points by persistence to reduce Wasserstein cost.
        """
        if dgm.shape[0] <= self.max_pts_per_dgm:
            return dgm
        pers = dgm[:, 1] - dgm[:, 0]
        idx = np.argsort(-pers)[: self.max_pts_per_dgm]
        return dgm[idx]

    def _vr_persistence_np(self, pts: torch.Tensor, hom_dim: int) -> np.ndarray:
        # Detach -> CPU -> ripser
        pts_np = pts.detach().cpu().numpy().astype(np.float32, copy=False)
        pts_np = self._standardize_np(pts_np)

        dgms = ripser(pts_np, maxdim=1)["dgms"]
        dgm = dgms[hom_dim]
        dgm = self._sanitize_dgm_np(dgm)
        dgm = self._cap_points(dgm)
        return dgm

    def forward(self, h_map_small: torch.Tensor):
        """
        Returns:
          dgms0: list length B, each is (n_i,2) np.ndarray for H0
          dgms1: list length B, each is (m_i,2) np.ndarray for H1
        """
        pts_batch = self._to_pointcloud(h_map_small)  # (B,N,C)
        dgms0, dgms1 = [], []
        for b in range(pts_batch.size(0)):
            pts = pts_batch[b]  # (N,C)
            dgms0.append(self._vr_persistence_np(pts, hom_dim=0))
            dgms1.append(self._vr_persistence_np(pts, hom_dim=1))
        return dgms0, dgms1


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


def ph_wasserstein_logits(
    dgms0,
    dgms1,
    order=1,
    tau_logits=1.0,
    w_h0=0.5,
    w_h1=1.0,
):
    """
    Build teacher logits S_ph (N,N) from Wasserstein distances.
    Higher logits => more similar.

    logits[i,j] = -(w_h0 * W(d0_i,d0_j) + w_h1 * W(d1_i,d1_j)) / tau_logits
    """
    N = len(dgms0)
    S = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            d0 = wasserstein(dgms0[i], dgms0[j], matching=False)
            d1 = wasserstein(dgms1[i], dgms1[j], matching=False)
            d  = float(w_h0) * float(d0) + float(w_h1) * float(d1)
            sij = -d / max(float(tau_logits), 1e-12)
            S[i, j] = sij
            S[j, i] = sij

    # mask diagonal to be very negative (like your sim_matrix)
    np.fill_diagonal(S, -1e5)
    return torch.from_numpy(S)


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

def ph_guided_contrastive_from_logits(
    rep: torch.Tensor,
    S_ph_logits: torch.Tensor,
    tau_student: float,
    tau_ph: float,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """
    PH defines soft neighbors via teacher logits S_ph_logits (N,N).
    Student uses rep similarities, and matches teacher soft targets.

    rep: (N,d) differentiable
    S_ph_logits: (N,N) teacher logits (higher more similar), diagonal masked already
    """
    # Student logits
    S_rep = sim_matrix(rep, t=tau_student)

    # Teacher targets (detach!)
    S_ph_logits = S_ph_logits.to(device=rep.device, dtype=rep.dtype).detach()
    P_ph = _soft_targets_from_sim(S_ph_logits, tau=tau_ph, topk=topk)

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

    ph_featurizer = PHDiagramFeaturizer(
        num_points=int(args.ph.num_points),
        max_pts_per_dgm=int(getattr(args.ph, "max_pts_per_dgm", 64)),
    ).to(device)

    logger.info(f"Base model: {args.backbone}")
    logger.info(f"feature dim: {model.feature_dim}, projection dim: {args.projection_dim}")
    logger.info(f"method: {args.method}")

    # Optimizer: train backbone+projector AND (optionally useful) PH projection head
    optimizer = torch.optim.SGD(
        list(model.parameters()),
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
                dgms0, dgms1 = ph_featurizer(h_map_small)

                S_ph = ph_wasserstein_logits(
                    dgms0, dgms1,
                    order=int(getattr(args.ph, "wasserstein_p", 1)),
                    tau_logits=float(getattr(args.ph, "wasserstein_tau", 1.0)),
                    w_h0=float(getattr(args.ph, "w_h0", 0.5)),
                    w_h1=float(getattr(args.ph, "w_h1", 1.0)),
                )

                loss = ph_guided_contrastive_from_logits(
                    rep=rep,
                    S_ph_logits=S_ph,
                    tau_student=tau_student,
                    tau_ph=tau_ph,
                    topk=ph_topk,
                )

            elif method_eff == "hybrid":
                alpha = float(args.loss.alpha)
                loss_cos = nt_xent(rep, temperature)

                dgms0, dgms1 = ph_featurizer(h_map_small)

                S_ph = ph_wasserstein_logits(
                    dgms0, dgms1,
                    order=int(getattr(args.ph, "wasserstein_p", 1)),
                    tau_logits=float(getattr(args.ph, "wasserstein_tau", 1.0)),
                    w_h0=float(getattr(args.ph, "w_h0", 0.5)),
                    w_h1=float(getattr(args.ph, "w_h1", 1.0)),
                )

                loss_phsim = ph_guided_contrastive_from_logits(
                    rep=rep,
                    S_ph_logits=S_ph,
                    tau_student=tau_student,
                    tau_ph=tau_ph,
                    topk=ph_topk,
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