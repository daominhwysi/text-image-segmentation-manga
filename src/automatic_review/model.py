import torch
import torch.nn as nn
import timm

class ReviewClassifier(nn.Module):
    def __init__(self, backbone_name="mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k", pretrained=True, freeze_backbone=False):
        super(ReviewClassifier, self).__init__()

        # 1. Load the MobileNetV4 Hybrid model
        # num_classes=0 + global_pool='avg' gives us a pooled feature vector
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        # 2. Modify the first layer (conv_stem) to accept 4 channels
        # MobileNetV4 uses a standard Conv2d at the start
        original_conv = self.backbone.conv_stem

        self.backbone.conv_stem = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

        # Copy pretrained weights and initialize the 4th channel (Mask)
        with torch.no_grad():
            self.backbone.conv_stem.weight[:, :3, :, :] = original_conv.weight
            # Average of RGB weights for the mask channel initialization
            self.backbone.conv_stem.weight[:, 3, :, :] = torch.mean(original_conv.weight, dim=1)

        # 3. Dynamic Feature Detection
        # MobileNetV4 Hybrid Medium typically outputs 960 or 1024 features,
        # but we detect it automatically to be safe.
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 256, 256) # MNv4 Medium is often trained at 256
            dummy_output = self.backbone(dummy_input)
            feature_dim = dummy_output.view(1, -1).size(1)
        self.backbone.train()

        print(f"Detected MobileNetV4 feature dimension: {feature_dim}")

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone (MobileNetV4) frozen.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone (MobileNetV4) unfrozen.")

    def forward(self, x):
        # x shape: [Batch, 4, H, W]
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
