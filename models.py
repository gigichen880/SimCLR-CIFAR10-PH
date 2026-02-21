import torch.nn as nn
import torch

# -------------------------
# Model
# -------------------------
class SimCLR(nn.Module):
    """
    Returns:
      h_map_small: (B, c_small, H, W)   feature map for PH (channel reduced)
      h:          (B, C)               pooled backbone feature
      rep:        (B, proj_dim)        projection head output
    """
    def __init__(
        self,
        base_encoder_fn,
        projection_dim=128,
        proj_hidden_dim=512,
        reduce_channels=8,
        cifar_no_maxpool: bool = True,
    ):
        super().__init__()
        backbone = base_encoder_fn(weights=None)

        if cifar_no_maxpool:
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        else:
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.feature_dim = backbone.fc.in_features  # 512 for resnet18/34

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, projection_dim),
        )

        # PH map will be taken from layer4 output (with no maxpool this is usually 2x2 on CIFAR)
        self.ph_reduce = nn.Conv2d(self.feature_dim, reduce_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        h_map = self.layer4(x)               # (B, 512, H, W)
        h_map_small = self.ph_reduce(h_map)  # (B, c_small, H, W)

        h = self.avgpool(h_map)              # (B, 512, 1, 1)
        h = torch.flatten(h, 1)              # (B, 512)
        rep = self.projector(h)              # (B, proj_dim)
        return h_map_small, h, rep



