"""
simclr_lin.py (updated to match your latest SimCLR + PH pipeline)

What this script does:
- Loads the checkpoint produced by simclr.py (baseline / phsim / hybrid)
- Builds a linear classifier on top of the *frozen encoder features*
- Trains only the linear layer (linear evaluation protocol)

Key updates vs your old version:
1) Matches checkpoint format from your updated simclr.py:
   - simclr.py saves: {"model": ..., "ph_featurizer": ..., "config": ...}
   - We now load state["model"] into the SimCLR model.

2) Uses the correct "encoder feature" output:
   - Your SimCLR class does NOT expose pre_model.enc.
   - Instead, SimCLR.forward returns (_, h, rep), where:
       h = pooled backbone feature (B, feature_dim)
   - For linear eval we train on h (NOT rep).

3) Keeps CIFAR stem consistent:
   - simclr.py used cifar_no_maxpool=True to avoid 1x1 maps.
   - We construct SimCLR the same way here.

4) Adds CSV + plots (loss/acc curves) in Hydra run dir, like simclr.py.

Run examples:
  python simclr_lin.py backbone=resnet18 method=baseline load_epoch=10
  python simclr_lin.py backbone=resnet18 method=phsim    load_epoch=10
  python simclr_lin.py backbone=resnet18 method=hybrid   load_epoch=10

Notes:
- For "linear evaluation" you typically use NO heavy augmentations; I keep it simple.
- Uses a small labeled subset by default (like your original): 10 samples per class.
"""

import os
import csv
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from tqdm import tqdm

from models import SimCLR 

logger = logging.getLogger(__name__)

# -------------------------
# small utils: history logging + plots
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class HistoryLogger:
    def __init__(self, out_dir: str, filename: str = "lin_history.csv"):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.csv_path = os.path.join(out_dir, filename)
        self.rows = []
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"])

    def log_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, lr):
        self.rows.append((epoch, train_loss, train_acc, test_loss, test_acc, lr))
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, train_acc, test_loss, test_acc, lr])

    def plot(self, tag):
        if not self.rows:
            return
        epochs = [r[0] for r in self.rows]
        train_losses = [r[1] for r in self.rows]
        train_accs = [r[2] for r in self.rows]
        test_losses = [r[3] for r in self.rows]
        test_accs = [r[4] for r in self.rows]

        plt.figure()
        plt.plot(epochs, train_losses, label="train")
        plt.plot(epochs, test_losses, label="test")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Linear Eval Loss ({tag})")
        plt.legend()
        plt.savefig(os.path.join(self.out_dir, f"lin_loss_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, train_accs, label="train")
        plt.plot(epochs, test_accs, label="test")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"Linear Eval Accuracy ({tag})")
        plt.legend()
        plt.savefig(os.path.join(self.out_dir, f"lin_acc_{tag}.png"), dpi=150)
        plt.close()


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
        v = float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def get_lr(step, total_steps, lr_max, lr_min):
    """Cosine annealing schedule (returns absolute LR)."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# -------------------------
# Linear eval head on pooled features h
# -------------------------
class LinearEvalModel(nn.Module):
    """
    Wraps the pretrained SimCLR model and trains a linear classifier on top of pooled feature h.

    SimCLR.forward(x) returns: (h_map_small, h, rep)
    We use h as feature for linear eval.

    Only self.fc is trainable.
    """
    def __init__(self, simclr_model: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.simclr = simclr_model
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        # We only want pooled features h
        _, h, _ = self.simclr(x)
        return self.fc(h)


@torch.no_grad()
def _set_encoder_eval_and_freeze(simclr_model: nn.Module):
    simclr_model.eval()
    for p in simclr_model.parameters():
        p.requires_grad = False


def run_epoch(model, dataloader, epoch, device, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    loader_bar = tqdm(dataloader)

    for x, y in loader_bar:
        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))

        phase = "Train" if is_train else "Test"
        loader_bar.set_description(
            f"{phase} epoch {epoch} | loss {loss_meter.avg:.4f} | acc {acc_meter.avg:.4f}"
        )

    return loss_meter.avg, acc_meter.avg


@hydra.main(version_base=None, config_path=".", config_name="simclr_config")
def finetune(args: DictConfig) -> None:
    logger.info("Config:\n" + OmegaConf.to_yaml(args))

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[SimCLR-LIN] using device = {device}")

    # Hydra run dir for outputs
    out_dir = os.getcwd()

    ckpt_dir = os.path.join(out_dir, "checkpoints", "downstream")
    viz_dir = os.path.join(out_dir, "visuals", "downstream")

    ensure_dir(ckpt_dir)
    ensure_dir(viz_dir)

    hist = HistoryLogger(viz_dir, filename=f"lin_history_{args.method}_{args.backbone}.csv")

    # Simple transforms for linear eval (common practice: light aug on train, standard on test)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    # labeled subset: 10 per class (like your original)
    n_classes = 10
    rng = np.random.default_rng(seed=int(getattr(args, "seed", 0) or 0))
    indices = rng.choice(len(train_set), 10 * n_classes, replace=False)
    sampler = SubsetRandomSampler(indices)

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        drop_last=False,
        sampler=sampler,
        num_workers=int(args.workers),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
    )

    # Build the same SimCLR backbone as training
    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = resnet18 if args.backbone == "resnet18" else resnet34

    pre_model = SimCLR(
        base_encoder_fn=base_encoder,
        projection_dim=int(args.projection_dim),
        proj_hidden_dim=int(args.model.proj_hidden_dim),
        reduce_channels=int(args.ph.reduce_channels),
        cifar_no_maxpool=True,  # must match simclr.py
    ).to(device)

    # Load checkpoint produced by simclr.py
    ckpt_path = f"checkpoints/upstream/simclr_{args.method}_{args.backbone}_epoch{int(args.load_epoch)}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Make sure you're running from the directory containing the checkpoint, "
            f"or pass an absolute path / adjust naming."
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # simclr.py saves ckpt["model"] = model.state_dict()
    if isinstance(ckpt, dict) and "model" in ckpt:
        pre_model.load_state_dict(ckpt["model"], strict=True)
    else:
        # fallback if you saved raw state_dict by mistake
        pre_model.load_state_dict(ckpt, strict=True)

    # Freeze encoder (SimCLR model)
    _set_encoder_eval_and_freeze(pre_model)

    # Linear eval model on pooled features h
    feature_dim = int(pre_model.feature_dim)  # 512 for resnet18/34
    model = LinearEvalModel(pre_model, feature_dim=feature_dim, n_classes=n_classes).to(device)

    # Optimizer only for linear layer
    parameters = [p for p in model.parameters() if p.requires_grad]
    lin_lr = 0.1 * float(args.batch_size) / 256.0  # SimCLR paper heuristic

    optimizer = torch.optim.SGD(
        parameters,
        lr=lin_lr,
        momentum=float(args.momentum),
        weight_decay=0.0,
        nesterov=True,
    )

    # Cosine annealing LR for finetuning epochs
    finetune_epochs = int(args.finetune_epochs)
    total_steps = finetune_epochs * len(train_loader)
    if total_steps <= 0:
        raise ValueError(
            f"Linear eval has 0 training steps. "
            f"Check labeled subset size vs batch_size. "
            f"(subset={len(indices)}, batch_size={int(args.batch_size)}, drop_last={False})"
        )
    lr_max = lin_lr
    lr_min = 1e-3

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, total_steps, lr_max, lr_min) / lr_max
    )

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, device, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch, device)

        current_lr = optimizer.param_groups[0]["lr"]
        tag = f"{args.method}_{args.backbone}_bs{args.batch_size}"
        if int(args.seed) != 0:
            tag += f"_seed{args.seed}"
        hist.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, current_lr)
        hist.plot(tag=tag)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            logger.info("==> New best test acc")
            best_name = f"simclr_lin_{tag}_best.pth"
            best_path = os.path.join(ckpt_dir, best_name)
            torch.save(model.state_dict(), best_path)

    logger.info(f"Best Test Acc: {best_test_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    finetune()