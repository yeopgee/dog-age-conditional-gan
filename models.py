"""Generator, discriminator, encoder, and perceptual feature models."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN
from torchvision import models as tv_models

class Generator(nn.Module):
    def __init__(self, z_dim=100, c_dim=2):
        super().__init__()
        self.fc = nn.Linear(z_dim + c_dim, 1024 * 4 * 4)

        self.net = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 4×4 → 8×8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64×64 → 128×128
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)         # [B, z_dim + c_dim]
        x = self.fc(x)                       # [B, 1024 * 4 * 4]
        x = x.view(-1, 1024, 4, 4)           # [B, 1024, 4, 4]
        return self.net(x)                   # [B, 3, 128, 128]

class InjectionGenerator(nn.Module):
    def __init__(self, z_dim=128, c_dim=2):
        super().__init__()
        self.fc = nn.Linear(z_dim + c_dim, 1024 * 4 * 4)

        # condition inject layers (선택된 2개만)
        self.inject_c1 = nn.Linear(c_dim, 512)
        self.inject_c2 = nn.Linear(c_dim, 256)

        # deconv blocks
        self.net = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),   # 4×4 → 8×8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),    # 8×8 → 16×16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 16×16 → 32×32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # 32×32 → 64×64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),       # 64×64 → 128×128
            nn.Tanh()
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)             # [B, z_dim + c_dim]
        x = self.fc(x).view(-1, 1024, 4, 4)      # [B, 1024, 4, 4]

        # Initial batchnorm + relu
        x = self.net[0](x)
        x = self.net[1](x)

        # Block 1: 4x4 → 8x8 + c inject
        x = self.net[2](x)
        x += self.inject_c1(c).view(-1, 512, 1, 1)
        x = self.net[3](x)
        x = self.net[4](x)

        # Block 2: 8x8 → 16x16 + c inject
        x = self.net[5](x)
        x += self.inject_c2(c).view(-1, 256, 1, 1)
        x = self.net[6](x)
        x = self.net[7](x)

        # Block 3: 16x16 → 32x32
        x = self.net[8](x)
        x = self.net[9](x)
        x = self.net[10](x)

        # Block 4: 32x32 → 64x64
        x = self.net[11](x)
        x = self.net[12](x)
        x = self.net[13](x)

        # Output: 64x64 → 128x128 → Tanh
        x = self.net[14](x)
        x = self.net[15](x)

        return x

class ConcatGenerator(nn.Module):
    def __init__(self, z_dim=128, c_dim=2, c_emb_dim=16):
        super().__init__()
        self.c_emb = nn.Linear(c_dim, c_emb_dim)             # 🔹 2 → 16
        self.fc = nn.Linear(z_dim + c_emb_dim, 1024 * 4 * 4)  # z + c_emb

        # 🔹 condition injection (block 2만)
        self.inject_c = nn.Linear(c_emb_dim, 256)

        # 🔹 deconv blocks
        self.net = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),   # 4x4 → 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),    # 8x8 → 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 16x16 → 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # 32x32 → 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),       # 64x64 → 128x128
            nn.Tanh()
        )

    def forward(self, z, c):
        # 🔹 Embed condition
        c_emb = self.c_emb(c)  # [B, 16]

        # 🔹 Initial projection
        x = torch.cat([z, c_emb], dim=1)              # [B, z + c_emb]
        x = self.fc(x).view(-1, 1024, 4, 4)

        # 🔹 Block 1: 4x4 → 8x8
        x = self.net[0](x)
        x = self.net[1](x)

        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)

        # 🔹 Block 2: 8x8 → 16x16 + condition inject
        x = self.net[5](x)
        x += self.inject_c(c_emb).view(-1, 256, 1, 1)  # 💡 단 한 번만 주입
        x = self.net[6](x)
        x = self.net[7](x)

        # 🔹 Block 3: 16x16 → 32x32
        x = self.net[8](x)
        x = self.net[9](x)
        x = self.net[10](x)

        # 🔹 Block 4: 32x32 → 64x64
        x = self.net[11](x)
        x = self.net[12](x)
        x = self.net[13](x)

        # 🔹 Output: 64x64 → 128x128
        x = self.net[14](x)
        x = self.net[15](x)

        return x

class ProjectionDiscriminator(nn.Module):
    def __init__(self, c_dim=2, img_channels=3):
        super().__init__()
        self.c_dim = c_dim

        self.feature_extractor = nn.Sequential(
            SN(nn.Conv2d(img_channels, 64, 4, 2, 1)),   # 128 → 64
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(64, 128, 4, 2, 1)),             # 64 → 32
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(128, 256, 4, 2, 1)),            # 32 → 16
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(256, 512, 4, 2, 1)),            # 16 → 8
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(512, 512, 4, 2, 1)),            # 8 → 4
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(512, 512, 4, 2, 1)),            # 4 → 2
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(512, 512, 2, 1, 0)),            # 2 → 1
        )

        self.linear = SN(nn.Linear(512, 1))              # unconditional logit
        self.embed = SN(nn.Embedding(c_dim, 512))        # class embedding (same dim as features)

    def forward(self, x, c):
        """
        x: [B, 3, 128, 128]
        c: [B] (LongTensor class label)
        """
        features = self.feature_extractor(x)             # [B, 512, 1, 1]
        features = features.view(x.size(0), -1)          # [B, 512]

        out = self.linear(features)                      # [B, 1]
        emb = self.embed(c)                              # [B, 512]
        proj = torch.sum(features * emb, dim=1, keepdim=True)  # [B, 1]

        return out + proj                                # [B, 1]

class ConcatDiscriminator(nn.Module):
    def __init__(self, c_dim=2, img_channels=3, embed_dim=16):
        super().__init__()
        self.c_dim = c_dim

        # ⚙️ Feature extractor
        self.feature_extractor = nn.Sequential(
            SN(nn.Conv2d(img_channels, 64, 4, 2, 1)),   # 128 → 64
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(64, 128, 4, 2, 1)),             # 64 → 32
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(128, 256, 4, 2, 1)),            # 32 → 16
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(256, 512, 4, 2, 1)),            # 16 → 8
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(512, 512, 8, 1, 0)),            # 8×8 → 1×1
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ⚙️ Class embedding
        self.embed = nn.Linear(c_dim, embed_dim)         # e.g. 2 → 16

        # ⚙️ Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x, c):
        """
        x: [B, 3, 128, 128]
        c: [B, c_dim] (e.g. one-hot)
        """
        B = x.size(0)
        features = self.feature_extractor(x)             # [B, 512, 1, 1]
        features = features.view(B, -1)                  # [B, 512]

        c_emb = self.embed(c)                            # [B, embed_dim]
        concat = torch.cat([features, c_emb], dim=1)     # [B, 512 + embed_dim]

        out = self.classifier(concat)                    # [B, 1]
        return out

class Discriminator(nn.Module):
    def __init__(self, c_dim=2, img_channels=3):
        super().__init__()
        self.c_dim = c_dim  # for one-hot

        self.net = nn.Sequential(
            # Input: [B, 3 + c_dim, 128, 128]
            nn.Conv2d(img_channels + c_dim, 64, 4, 2, 1),   # 64×64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),                    # 32×32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),                   # 16×16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),                   # 8×8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 8),                           # 1×1
        )

    def forward(self, x, c):
        """
        x: [B, 3, 128, 128]
        c: [B] (LongTensor class label)
        """
        B, _, H, W = x.shape
        c_onehot = F.one_hot(c, num_classes=self.c_dim).float()  # [B, c_dim]
        c_map = c_onehot.view(B, self.c_dim, 1, 1).expand(B, self.c_dim, H, W)  # [B, c_dim, H, W]
        x_cond = torch.cat([x, c_map], dim=1)  # [B, 3 + c_dim, H, W]
        out = self.net(x_cond)                 # [B, 1, 1, 1]
        return out.view(-1, 1)                 # [B, 1]

class Encoder(nn.Module):
    def __init__(self, z_dim=128, c_dim=2, img_channels=3):
        super().__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels + c_dim, 64, 4, 2, 1),   # 128 → 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),                    # 64 → 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),                   # 32 → 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),                   # 16 → 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1),                  # 8 → 4
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                        # 4×4 → 1×1
        )
        self.fc = nn.Linear(1024, z_dim)

    def forward(self, x, c):
        # c: one-hot vector [B, c_dim]
        B, _, H, W = x.shape
        c = c.view(B, self.c_dim, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([x, c], dim=1)  # [B, 3 + c_dim, H, W]
        h = self.conv(x)              # [B, 1024, 1, 1]
        h = h.view(B, -1)             # [B, 1024]
        z = self.fc(h)                # [B, z_dim]
        return z

class ConcatEncoder(nn.Module):
    def __init__(self, z_dim=128, c_dim=2, c_emb_dim=16, img_channels=3):
        super().__init__()
        self.c_emb = nn.Linear(c_dim, c_emb_dim)  # 🔹 c: one-hot → embed
        self.z_dim = z_dim

        # 🔹 CNN Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),     # 128 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),              # 64 → 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),             # 32 → 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),             # 16 → 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1),             # 8 → 4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)                   # 4x4 → 1x1
        )

        # 🔹 FC: concat([h, c_emb]) → z
        self.fc = nn.Sequential(
            nn.Linear(512 + c_emb_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, z_dim)
        )

    def forward(self, x, c):
        B = x.size(0)
        h = self.conv(x).view(B, -1)          # [B, 512]
        c_emb = self.c_emb(c)                # [B, 16]
        x_cat = torch.cat([h, c_emb], dim=1) # [B, 512+16]
        z = self.fc(x_cat)                   # [B, z_dim]
        return z

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:36])  # conv5_2까지
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.slice(x)

class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss: 이미지의 노이즈를 줄이고 더 자연스럽게 만들기 위해
    인접한 픽셀 간의 차이를 최소화합니다.
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        loss_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))  # height 차이
        loss_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))  # width 차이
        return loss_h + loss_w
