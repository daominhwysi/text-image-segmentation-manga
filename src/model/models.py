import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForImageClassification
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, up=True) -> None:
        super().__init__()
        if up:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = SCSEModule(ch=out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2: Conv 3x3, Rate = 2
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 3: Conv 3x3, Rate = 4
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 4: Conv 3x3, Rate = 6
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 5: Global Average Pooling (Global View)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final Merge
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)

        feat4 = self.avg_pool(x)
        feat4 = self.b4(feat4)
        feat4 = F.interpolate(feat4, size=size, mode="bilinear", align_corners=True)

        # Concat
        x = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)
        x = self.project(x)
        return x


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        #  Channel SE
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // re, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // re, ch, 1),
            nn.Sigmoid(),
        )
        #  Spatial SE
        self.sSE = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Unet_B0(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Unet_B0, self).__init__()

        backbone = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        self.encoder = backbone.segformer.encoder

        self.encoder.patch_embeddings[0].proj.stride = (2, 2)
        self.encoder.patch_embeddings[1].proj.stride = (2, 2)
        self.encoder.patch_embeddings[2].proj.stride = (2, 2)
        self.encoder.patch_embeddings[3].proj.stride = (2, 2)

        self.aspp = ASPP(in_channels=256, out_channels=256)

        self.de_layer1 = DecoderBlock(
            in_channels=256, skip_channels=160, out_channels=160
        )

        self.de_layer2 = DecoderBlock(
            in_channels=160, skip_channels=64, out_channels=64
        )

        self.de_layer3 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        self.head1 = nn.Conv2d(160, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # hidden_states: [patch_embed, stage1, stage2, stage3, stage4]
        # s[0] = stage 1 (32 channels)
        # s[1] = stage 2 (64 channels)
        # s[2] = stage 3 (160 channels)
        # s[3] = stage 4 (256 channels)
        s = self.encoder(x, output_hidden_states=True).hidden_states

        bottleneck = self.aspp(s[3])
        d0 = self.de_layer1(bottleneck, s[2])
        d1 = self.de_layer2(d0, s[1])
        d2 = self.de_layer3(d1, s[0])

        main_out = self.final_up(d2)

        if self.training:
            out1 = F.interpolate(
                self.head1(d0), size=(256, 256), mode="bilinear", align_corners=True
            )
            out2 = F.interpolate(
                self.head2(d1), size=(256, 256), mode="bilinear", align_corners=True
            )
            out3 = F.interpolate(
                self.head3(d2), size=(256, 256), mode="bilinear", align_corners=True
            )
            return main_out, out3, out2, out1
        else:
            return main_out


class Unet_B1(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(Unet_B1, self).__init__()
        backbone = SegformerForImageClassification.from_pretrained("nvidia/mit-b1")
        self.encoder = backbone.segformer.encoder

        self.encoder.patch_embeddings[0].proj.stride = (2, 2)
        self.encoder.patch_embeddings[1].proj.stride = (2, 2)

        self.encoder.patch_embeddings[2].proj.stride = (2, 2)
        self.encoder.patch_embeddings[3].proj.stride = (2, 2)

        self.de_layer1 = DecoderBlock(
            in_channels=512, skip_channels=320, out_channels=320
        )
        self.de_layer2 = DecoderBlock(
            in_channels=320, skip_channels=128, out_channels=128
        )
        self.de_layer3 = DecoderBlock(
            in_channels=128, skip_channels=64, out_channels=64
        )
        self.aspp = ASPP(in_channels=512, out_channels=512)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        self.head1 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        s = self.encoder(x, output_hidden_states=True).hidden_states
        bottleneck = self.aspp(s[3])
        d0 = self.de_layer1(bottleneck, s[2])
        d1 = self.de_layer2(d0, s[1])
        d2 = self.de_layer3(d1, s[0])

        main_out = self.final_up(d2)
        if self.training:
            out1 = F.interpolate(
                self.head1(d0), size=(256, 256), mode="bilinear", align_corners=True
            )
            out2 = F.interpolate(
                self.head2(d1), size=(256, 256), mode="bilinear", align_corners=True
            )
            out3 = F.interpolate(
                self.head3(d2), size=(256, 256), mode="bilinear", align_corners=True
            )
            return main_out, out3, out2, out1
        else:
            return main_out
