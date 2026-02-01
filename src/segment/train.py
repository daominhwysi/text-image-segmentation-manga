import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# UPDATE: New AMP API
from torch.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torchmetrics.classification import BinaryJaccardIndex
from src.segment.models import Unet_EfficientViT_B2, Unet_MobileNetV4, Unet_YOLO, Unet_YOLO_Medium

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS_FREEZED = 5  # Warm-up phase
NUM_EPOCHS_TOTAL = 30
LR_HEAD = 1e-3         # Faster LR for new decoder
LR_FINETUNE = 1e-5     # Slow LR for backbone


def get_transforms(split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),

            # 1. Geometric Augmentations (Affine replaces ShiftScaleRotate)
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),           # Replaces scale_limit=0.2
                rotate=(-15, 15),           # Replaces rotate_limit=15
                translate_percent=(-0.1, 0.1), # Replaces shift_limit=0.1
                p=0.7
            ),

            # 2. Distortion (Parameters updated for newer Albumentations)
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                # Removed 'alpha_affine', strictly using elastic params
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                # Removed 'shift_limit', only distort_limit is needed
                A.OpticalDistortion(distort_limit=0.2, p=1.0),
            ], p=0.3),

            # 3. Pixel-level Augmentations
            A.OneOf([
                # Switched to ISONoise (more stable API than GaussNoise across versions)
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                A.MotionBlur(p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.3),

            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# ==========================================
# 1. DATASET
# ==========================================
class TextSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        self.image_names = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_dir, os.path.splitext(self.image_names[idx])[0] + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        else:
            mask = mask.float()

        # Add channel dimension
        mask = mask.unsqueeze(0)

        return image, mask


# ==========================================
# 3. LOSS & TRAINING
# ==========================================
class SoftDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(SoftDiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def soft_dice_loss(self, logits, targets):
        smooth = 1e-6
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs + targets)
        dice_score = (2. * intersection + smooth) / (cardinality + smooth)
        return 1 - dice_score

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.soft_dice_loss(logits, targets)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

class CombinedDSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_loss = SoftDiceBCELoss()
        self.ds_loss = SoftDiceBCELoss()

    def forward(self, outputs, targets):
        if isinstance(outputs, (list, tuple)):
            main_out, ds3, ds2, ds1 = outputs
            l_main = self.main_loss(main_out, targets)
            l_ds = (self.ds_loss(ds3, targets) +
                    self.ds_loss(ds2, targets) +
                    self.ds_loss(ds1, targets)) / 3.0
            return l_main + 0.4 * l_ds
        else:
            return self.main_loss(outputs, targets)
def train_model():
    # 1. Setup Data
    train_ds = TextSegmentationDataset('manga_data', 'train', transform=get_transforms("train"))
    val_ds = TextSegmentationDataset('manga_data', 'test', transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model (Starts Frozen)
    model = Unet_MobileNetV4(num_classes=1).to(DEVICE)
    model.freeze_backbone()

    # 3. Optimizer & Loss
    # Only optimize parameters that require_grad
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    criterion = CombinedDSLoss()
    scaler = GradScaler("cuda")
    metric = BinaryJaccardIndex().to(DEVICE)

    best_miou = 0.0

    for epoch in range(NUM_EPOCHS_TOTAL):
        # --- Phase Switching Logic ---
        if epoch == NUM_EPOCHS_FREEZED:
            print("\n🔥 Unfreezing Backbone for Fine-tuning...")
            model.unfreeze_backbone()
            # Re-initialize optimizer with lower LR for the whole network
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE)

        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_TOTAL}")

        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation ---
        model.eval()
        metric.reset()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                # model.eval() returns ONLY the main output, not a tuple
                val_output = model(imgs)
                # For binary segmentation, we use threshold 0.5 on sigmoid (or 0 on logits)
                preds = (val_output > 0).float()
                metric.update(preds, masks)

        miou = metric.compute().item()
        print(f"📊 Epoch {epoch+1} | Avg Loss: {epoch_loss/len(train_loader):.4f} | Val mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "best_manga_unet.pth")
            print("💾 Model Saved!")

if __name__ == "__main__":
    train_model()
