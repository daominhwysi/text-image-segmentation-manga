import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ultralytics import YOLO # Ensure ultralytics is installed

# ==========================================
# COMMON COMPONENTS
# ==========================================

class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // re, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // re, ch, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, up=True) -> None:
        super().__init__()
        if up:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        else:
            self.upsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels + skip_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
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
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.b4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        f0, f1, f2, f3 = self.b0(x), self.b1(x), self.b2(x), self.b3(x)
        f4 = F.interpolate(self.b4(self.avg_pool(x)), size=size, mode='bilinear', align_corners=True)
        return self.project(torch.cat([f0, f1, f2, f3, f4], dim=1))

# ==========================================
# MODELS
# ==========================================

class Unet_EfficientViT_B2(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Unet_EfficientViT_B2, self).__init__()
        backbone = timm.create_model('efficientvit_b2.r224_in1k', pretrained=pretrained, features_only=False)

        self.stem = backbone.stem
        self.stage0 = backbone.stages[0] # 1/4
        self.stage1 = backbone.stages[1] # 1/8
        self.stage2 = backbone.stages[2] # 1/16
        self.stage3 = backbone.stages[3] # 1/16 (modified)

        # Modify Stride for 1/16 bottleneck
        target_layer = self.stage3.blocks[0].main.depth_conv.conv
        target_layer.stride, target_layer.dilation, target_layer.padding = (1, 1), (2, 2), (2, 2)

        self.aspp = ASPP(384, 384)
        self.de_layer1 = DecoderBlock(384, 192, 192, up=False)
        self.de_layer2 = DecoderBlock(192, 96, 96, up=True)
        self.de_layer3 = DecoderBlock(96, 48, 48, up=True)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, num_classes, kernel_size=1)
        )
        self.head1 = nn.Conv2d(192, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(96, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(48, num_classes, kernel_size=1)

    def freeze_backbone(self):
        for m in [self.stem, self.stage0, self.stage1, self.stage2, self.stage3]:
            for param in m.parameters():
                param.requires_grad = False
        print("Freezed Backbone")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        print("unFreezed Backbone")
    def forward(self, x):
        input_size = x.shape[-2:]
        s0 = self.stage0(self.stem(x))
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        d0 = self.de_layer1(self.aspp(s3), s2)
        d1 = self.de_layer2(d0, s1)
        d2 = self.de_layer3(d1, s0)
        main_out = self.final_up(d2)

        if self.training:
            return main_out, \
                   F.interpolate(self.head3(d2), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.head2(d1), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.head1(d0), size=input_size, mode='bilinear', align_corners=True)
        return main_out

class Unet_MobileNetV4(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Unet_MobileNetV4, self).__init__()
        backbone = timm.create_model('mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k', pretrained=pretrained, features_only=False)

        self.stem = nn.Sequential(backbone.conv_stem, backbone.bn1) # 1/2
        self.stage0 = backbone.blocks[0] # 1/4
        self.stage1 = backbone.blocks[1] # 1/8
        self.stage2 = backbone.blocks[2] # 1/16
        self.stage3 = backbone.blocks[3] # 1/16 (modified)

        # # Modify Stride for 1/16 bottleneck
        # target_layer = self.stage3[0].dw_mid.conv
        # target_layer.stride, target_layer.dilation, target_layer.padding = (1, 1), (2, 2), (4, 4)

        self.aspp = ASPP(256, 256)
        self.de_layer1 = DecoderBlock(256, 160, 160, up=True)
        self.de_layer2 = DecoderBlock(160, 80, 80, up=True)
        self.de_layer3 = DecoderBlock(80, 48, 48, up=True)
        self.de_layer4 = DecoderBlock(48, 32, 24, up=True)


        self.seg_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1) # Logits for CrossEntropy
        )

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.head1 = nn.Conv2d(160, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(80, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(48, num_classes, kernel_size=1)

    def freeze_backbone(self):
        for m in [self.stem, self.stage0, self.stage1, self.stage2, self.stage3]:
            for param in m.parameters():
                param.requires_grad = False
        print("Freezed Backbone")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        print("UnFreezed Backbone")


    def forward(self, x):
        input_size = x.shape[-2:]
        stem_out = self.stem(x)
        s0 = self.stage0(stem_out)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        d0 = self.de_layer1(self.aspp(s3), s2)
        d1 = self.de_layer2(d0, s1)
        d2 = self.de_layer3(d1, s0)
        d3 = self.de_layer4(d2, stem_out)


        # Final scale up
        seg_out = F.interpolate(self.seg_head(d3), size=input_size, mode='bilinear', align_corners=True)


        if self.training:
            # Matches CombinedDSLoss: main_out, ds3, ds2, ds1
            return seg_out, \
                   F.interpolate(self.head3(d2), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.head2(d1), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.head1(d0), size=input_size, mode='bilinear', align_corners=True)
        return seg_out



class Unet_YOLO(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 1. Load Backbone (YOLOv11n structure assumed based on your layers)
        yolo_base = YOLO('yolo26n.pt').model.model

        self.stage1 = yolo_base[0:3]   # P2 (1/4)
        self.stage2 = yolo_base[3:5]   # P3 (1/8)
        self.stage3 = yolo_base[5:7]   # P4 (1/16)
        self.stage4 = yolo_base[7:11]  # P5 (1/32)

        self.aspp = ASPP(in_channels=256, out_channels=256)

        # 2. Decoder Layers
        self.up1 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128) # -> 1/16
        self.up2 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)  # -> 1/8
        self.up3 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)    # -> 1/4

        # 3. Final Main Head
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

        # 4. Deep Supervision Heads (Auxiliary)
        self.ds_head1 = nn.Conv2d(128, num_classes, kernel_size=1) # From 1/16
        self.ds_head2 = nn.Conv2d(64, num_classes, kernel_size=1)  # From 1/8
        self.ds_head3 = nn.Conv2d(32, num_classes, kernel_size=1)  # From 1/4

    def freeze_backbone(self):
        for m in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for param in m.parameters():
                param.requires_grad = False
        print("Backbone Freezed")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        print("Model Fully Unfreezed")

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        # Decoder
        d1 = self.up1(self.aspp(p5), p4)
        d2 = self.up2(d1, p3)
        d3 = self.up3(d2, p2)

        main_out = self.final_up(d3)

        if self.training:
            # Return main output + auxiliary outputs upsampled to input size
            return main_out, \
                   F.interpolate(self.ds_head3(d3), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.ds_head2(d2), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.ds_head1(d1), size=input_size, mode='bilinear', align_corners=True)
        return main_out

class Unet_YOLO_Medium(nn.Module):
    def __init__(self, model_variant='yolo26m.pt', num_classes=2):
        super().__init__()

        yolo_base = YOLO(model_variant).model.model

        self.stage1 = yolo_base[0:3]
        self.stage2 = yolo_base[3:5]
        self.stage3 = yolo_base[5:7]
        self.stage4 = yolo_base[7:11]

        # Dynamic channel detection for Medium version
        ch_p2 = self._get_out_channels(self.stage1, 3)
        ch_p3 = self._get_out_channels(self.stage2, ch_p2)
        ch_p4 = self._get_out_channels(self.stage3, ch_p3)
        ch_p5 = self._get_out_channels(self.stage4, ch_p4)

        self.aspp = ASPP(in_channels=ch_p5, out_channels=ch_p5)

        # Decoder
        self.up1 = DecoderBlock(in_channels=ch_p5, skip_channels=ch_p4, out_channels=ch_p4)
        self.up2 = DecoderBlock(in_channels=ch_p4, skip_channels=ch_p3, out_channels=ch_p3 // 2)
        self.up3 = DecoderBlock(in_channels=ch_p3 // 2, skip_channels=ch_p2, out_channels=ch_p2 // 2)

        # Main Head
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_p2 // 2, ch_p2 // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_p2 // 4, num_classes, kernel_size=1)
        )

        # Deep Supervision Heads
        self.ds_head1 = nn.Conv2d(ch_p4, num_classes, kernel_size=1)
        self.ds_head2 = nn.Conv2d(ch_p3 // 2, num_classes, kernel_size=1)
        self.ds_head3 = nn.Conv2d(ch_p2 // 2, num_classes, kernel_size=1)

    def _get_out_channels(self, layer, in_ch):
        dummy_input = torch.zeros(1, in_ch, 64, 64)
        with torch.no_grad():
            output = layer(dummy_input)
        return output.shape[1]

    def freeze_backbone(self):
        for m in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for param in m.parameters():
                param.requires_grad = False
        print("Medium Backbone Freezed")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        input_size = x.shape[-2:]

        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        d1 = self.up1(self.aspp(p5), p4)
        d2 = self.up2(d1, p3)
        d3 = self.up3(d2, p2)

        main_out = self.final_up(d3)

        if self.training:
            return main_out, \
                   F.interpolate(self.ds_head3(d3), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.ds_head2(d2), size=input_size, mode='bilinear', align_corners=True), \
                   F.interpolate(self.ds_head1(d1), size=input_size, mode='bilinear', align_corners=True)
        return main_out
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet_YOLO_Medium(num_classes=2).to(device)

    # Standard input size for this backbone
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [1, 2, 256, 256]
