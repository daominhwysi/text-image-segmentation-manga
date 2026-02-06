import torch
import torch.nn as nn
import torchvision.models as models

class ReviewClassifier(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super(ReviewClassifier, self).__init__()
        if backbone == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            # Modify the first layer to accept 4 channels (RGB + Mask)
            original_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                4,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            # Copy weights from RGB to the first 3 channels, initialize 4th channel
            with torch.no_grad():
                self.model.conv1.weight[:, :3, :, :] = original_conv.weight
                self.model.conv1.weight[:, 3, :, :] = torch.mean(original_conv.weight, dim=1)

            # Binary classification output
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        # x: [B, 4, H, W]
        return self.model(x)
