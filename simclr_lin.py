import hydra
from omegaconf import DictConfig
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from models import SimCLR
from tqdm import tqdm


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


def run_epoch(model, dataloader, epoch, device, optimizer=None, scheduler=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)

    for x, y in loader_bar:
        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))

        phase = "Train" if optimizer is not None else "Test"
        loader_bar.set_description(
            f"{phase} epoch {epoch}, loss: {loss_meter.avg:.4f}, acc: {acc_meter.avg:.4f}"
        )

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# @hydra.main(config_path='simclr_config.yml')
@hydra.main(version_base=None, config_path=".", config_name="simclr_config")
def finetune(args: DictConfig) -> None:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[SimCLR-LIN] using device = {device}")

    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    n_classes = 10
    indices = np.random.choice(len(train_set), 10*n_classes, replace=False)
    sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, sampler=sampler, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Prepare model
    base_encoder = eval(args.backbone)
    # pre_model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    pre_model = SimCLR(base_encoder, projection_dim=args.projection_dim).to(device)

    # pre_model.load_state_dict(torch.load('simclr_{}_epoch{}.pt'.format(args.backbone, args.load_epoch)))

    ckpt_path = 'simclr_{}_epoch{}.pt'.format(args.backbone, args.load_epoch)
    state = torch.load(ckpt_path, map_location=device)
    pre_model.load_state_dict(state)

    n_classes = 10
    # model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=len(train_set.targets))
    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=n_classes)
    # model = model.cuda()
    model = model.to(device)

    # Fix encoder
    # model.enc.requires_grad = False
    for p in model.enc.parameters():
      p.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.
    # optimizer = Adam(parameters, lr=0.001)
    lin_lr = 0.1 * args.batch_size / 256.0

    optimizer = torch.optim.SGD(
        parameters,
        lin_lr,   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
        momentum=args.momentum,
        weight_decay=0.,
        nesterov=True)

    # cosine annealing lr
    # scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
    #         step,
    #         # args.epochs * len(train_loader),
    #         args.finetune_epochs * len(train_loader),
    #         args.learning_rate,  # lr_lambda computes multiplicative factor
    #         1e-3))
    total_steps = args.finetune_epochs * len(train_loader)
    lr_max = lin_lr
    lr_min = 1e-3

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, total_steps, lr_max, lr_min) / lr_max
    )

    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, device, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch, device)

        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            torch.save(model.state_dict(), 'simclr_lin_{}_best.pth'.format(args.backbone))

    logger.info("Best Test Acc: {:.4f}".format(optimal_acc))


if __name__ == '__main__':
    finetune()


