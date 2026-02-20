"""
simclr.py

Supports 3 training methods controlled by yaml / CLI:
  - method=baseline : standard SimCLR NT-Xent on rep
  - method=hybrid   : alpha*SimCLR + (1-alpha)*PH-NTXent (PH via ripser on CPU; stop-grad)
  - method=teacher  : PH-as-teacher similarity matching (KL) where PH sim is the target,
                      and rep sim is the student (backbone trains; PH computed on CPU)

Run examples:
  python simclr.py backbone=resnet18 method=baseline
  python simclr.py backbone=resnet18 method=hybrid loss.alpha=0.9
  python simclr.py backbone=resnet18 method=teacher

"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
from PIL import Image
from ripser import ripser
from torchph.pershom import vr_persistence_l1

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
# Model: ResNet backbone that exposes pre-pool feature map
# -------------------------
class SimCLR(nn.Module):
    """
    Returns:
      h_map_small: (B, c_small, H, W)   pre-global-pool map (channel reduced)
      h:          (B, C)               pooled backbone feature
      rep:        (B, proj_dim)        projection head output
    """
    def __init__(self, base_encoder_fn, projection_dim=128, proj_hidden_dim=512, reduce_channels=8):
        super().__init__()
        backbone = base_encoder_fn(weights=None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.feature_dim = backbone.fc.in_features  # e.g. 512 for resnet18

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, projection_dim),
        )

        self.ph_reduce = nn.Conv2d(self.feature_dim, reduce_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        h_map = self.layer4(x)               # (B, C, H, W)  <-- pre-global pooling
        h_map_small = self.ph_reduce(h_map)  # (B, c_small, H, W)

        h = self.avgpool(h_map)              # (B, C, 1, 1)
        h = torch.flatten(h, 1)              # (B, C)
        rep = self.projector(h)              # (B, proj_dim)
        return h_map_small, h, rep


# -------------------------
# PH featurizer (CPU ripser; stop-grad)
# -------------------------
class SoftPersistenceImage(nn.Module):
    """
    Diagram -> fixed vector via Gaussian bumps on (birth, persistence) grid.
    NOTE: If ripser yields death=inf, we filter those points.
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
        centers = torch.stack([bb.reshape(-1), pp.reshape(-1)], dim=1)  # (G, 2)
        self.register_buffer("centers", centers)

    def forward(self, diagram_bd: torch.Tensor) -> torch.Tensor:
        """
        diagram_bd: (M,2) (birth, death)
        returns: (G,) persistence image feature
        """

        # Drop inf deaths (essential classes) + any NaNs
        mask = torch.isfinite(diagram_bd).all(dim=1) & torch.isfinite(diagram_bd[:, 1])
        diagram_bd = diagram_bd[mask]
        if diagram_bd.numel() == 0:
            return diagram_bd.new_zeros(self.centers.shape[0])

        birth = diagram_bd[:, 0]
        death = diagram_bd[:, 1]
        pers = (death - birth).clamp(min=0.0)

        # Convert to (birth, persistence)
        pts = torch.stack([birth, pers], dim=1)  # (M,2)

        diff = pts[:, None, :] - self.centers[None, :, :]  # (M,G,2)
        dist2 = (diff ** 2).sum(dim=2)                     # (M,G)
        bumps = torch.exp(-dist2 / (2 * self.sigma * self.sigma))  # (M,G)

        weights = pers[:, None]  # weight by persistence
        feat = (bumps * weights).sum(dim=0)  # (G,)
        return feat


class PHFeaturizer(nn.Module):
    """
    Pre-pool map -> point cloud -> ripser diagrams -> persistence-image -> projected vector.

    ripser is CPU + non-differentiable, so this is stop-grad w.r.t the CNN.
    """
    def __init__(
        self,
        out_dim=128,
        num_points=16,
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
        # h_map_small: (B, c_small, H, W) -> (B, N, c_small)
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
        """
        pts: (N, C) torch tensor for ONE sample (CUDA)
        returns: (M, 2) torch tensor (birth, death) on pts.device
        """
        # IMPORTANT: torchph expects float32 on GPU
        pts = pts.float()

        # vr_persistence_l1 returns a nested structure; we index out the diagram tensor.
        # Pattern used in torchph examples: out[0][0] gives (M,2) diagram
        out = vr_persistence_l1(pts, hom_dim, 0)
        dgm = out[0][0]  # (M,2) birth/death

        # safety: empty handling
        if dgm.numel() == 0:
            return pts.new_zeros((0, 2))

        return dgm

    def forward(self, h_map_small: torch.Tensor) -> torch.Tensor:
        pts_batch = self._to_pointcloud(h_map_small)  # (B, N, C)
        feats = []
        for b in range(pts_batch.size(0)):
            pts = pts_batch[b]

            d0 = self._vr_persistence(pts, hom_dim=0)
            f0 = self.pi(d0)

            d1 = self._vr_persistence(pts, hom_dim=1)
            f1 = self.pi(d1)

            feats.append(torch.cat([f0, f1], dim=0))

        feats = torch.stack(feats, dim=0)  # (B, 2*raw_dim)
        # stabilize magnitude going into proj
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
        return self.proj(feats)  # (B, out_dim)


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


def ph_nt_xent(h_map_small: torch.Tensor, ph_featurizer: nn.Module, t=0.5) -> torch.Tensor:
    ph_vec = ph_featurizer(h_map_small)  # (2B, d)
    ph_vec = ph_vec / (ph_vec.norm(dim=1, keepdim=True) + 1e-8)

    sim = (ph_vec @ ph_vec.t()).clamp(min=1e-7)
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


def kl_match_loss(student_sim: torch.Tensor, teacher_sim: torch.Tensor) -> torch.Tensor:
    """
    KL( teacher || student ) over row-wise softmax distributions.
    teacher is detached (fixed target).
    """
    p = F.softmax(teacher_sim, dim=1).detach()
    logq = F.log_softmax(student_sim, dim=1)
    return F.kl_div(logq, p, reduction="batchmean")


# -------------------------
# Train
# -------------------------
@hydra.main(version_base=None, config_path=".", config_name="simclr_config")
def train(args: DictConfig) -> None:
    # Log config for reproducibility
    logger.info("Config:\n" + OmegaConf.to_yaml(args))

    # Device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[SimCLR] using device = {device}")
    if device == "cuda":
        cudnn.benchmark = True

    # Seed
    seed = int(getattr(args, "seed", 0))
    if seed:
        set_seed(seed)

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
    ).to(device)

    # PH featurizer
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

    # Optimizer
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(ph_featurizer.parameters()),
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

    # Training
    model.train()
    ph_featurizer.train()

    max_steps = int(args.train.max_steps) if int(args.train.max_steps) > 0 else None
    temperature = float(args.temperature)

    for epoch in range(1, int(args.epochs) + 1):
        loss_meter = AverageMeter("loss")
        bar = tqdm(train_loader, total=len(train_loader))

        for step, (x, _) in enumerate(bar):
            if max_steps is not None and step >= max_steps:
                break

            # x: (B,2,C,H,W) -> (2B,C,H,W)
            B = x.size(0)
            x = x.view(B * 2, x.size(2), x.size(3), x.size(4)).to(device, non_blocking=(device == "cuda"))

            optimizer.zero_grad()
            h_map_small, _, rep = model(x)

            method = str(args.method).lower()

            if method == "baseline":
                loss = nt_xent(rep, temperature)

            elif method == "hybrid":
                alpha = float(args.loss.alpha)
                loss_cos = nt_xent(rep, temperature)
                loss_ph = ph_nt_xent(h_map_small, ph_featurizer, temperature)
                loss = alpha * loss_cos + (1.0 - alpha) * loss_ph

            elif method == "teacher":
                # teacher sim from PH (CPU ripser inside featurizer), student sim from rep (differentiable)
                teacher_tau = float(args.loss.teacher_temperature)
                student_tau = float(args.loss.student_temperature)

                ph_vec = ph_featurizer(h_map_small)
                S_ph = sim_matrix(ph_vec, teacher_tau)

                S_rep = sim_matrix(rep, student_tau)
                loss = kl_match_loss(S_rep, S_ph)

            else:
                raise ValueError(f"Unknown method={args.method}. Use baseline|hybrid|teacher.")

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


if __name__ == "__main__":
    train()