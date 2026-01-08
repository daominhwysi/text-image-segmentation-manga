import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex
import torch.nn.functional as F
from src.model.models import Unet_B0, Unet_B1
from torch.cuda.amp import GradScaler

# --- INITIALIZATION ---
scaler = GradScaler()
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

# Create checkpoints directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


# --- DATASET CLASS ---
class MySegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_side_len=256, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform
        self.img_side_len = img_side_len

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_name = self.images[idx].replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # Resize mask to target side length
        mask = mask.resize(
            (self.img_side_len, self.img_side_len), resample=Image.NEAREST
        )
        mask_np = np.array(mask)

        # Thresholding: 255 -> 1, 0 -> 0
        mask_final = np.zeros_like(mask_np, dtype=np.int64)
        mask_final[mask_np > 128] = 1

        mask_tensor = torch.from_numpy(mask_final).long()
        return image, mask_tensor


# --- LOSS FUNCTIONS ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        probs = probs[:, 1, :, :]
        targets = targets.float()
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class ComboLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(ComboLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        return self.ce_weight * self.ce(
            outputs, targets
        ) + self.dice_weight * self.dice(outputs, targets)


# --- CONFIGURATION & DATA ---
img_side_len = 256
transform = transforms.Compose(
    [
        transforms.Resize((img_side_len, img_side_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = MySegmentationDataset(
    "dataset/train/images",
    "dataset/train/masks",
    transform=transform,
    img_side_len=img_side_len,
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = MySegmentationDataset(
    "dataset/val/images",
    "dataset/val/masks",
    transform=transform,
    img_side_len=img_side_len,
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --- MODEL & TRAINING SETUP ---
num_classes = 2
model = Unet_B1(num_classes=num_classes, in_channels=3).to(device)
miou_metric = MulticlassJaccardIndex(num_classes=2).to(device)

criterion = ComboLoss(ce_weight=0.5, dice_weight=0.5).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.5, patience=2
)

num_epochs = 20
best_miou = 0
val_interval = 500
global_step = 0
save_path_best = "best_unet_256_b1.pth"

# --- TRAINING LOOP ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Train")

    for images, masks in train_loop:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type):
            outputs = model(images)
            # Deep supervision loss
            l_main = criterion(outputs[0], masks)
            l3 = criterion(outputs[1], masks)
            l2 = criterion(outputs[2], masks)
            l1 = criterion(outputs[3], masks)
            total_loss = l_main + 0.4 * l3 + 0.2 * l2 + 0.1 * l1

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        global_step += 1
        train_loop.set_postfix(loss=total_loss.item(), step=global_step)

        # --- VALIDATION PHASE ---
        if global_step % val_interval == 0:
            model.eval()
            val_loss = 0.0
            miou_metric.reset()

            with torch.no_grad():
                with torch.autocast(device_type=device_type):
                    for v_images, v_masks in val_loader:
                        v_images, v_masks = v_images.to(device), v_masks.to(device)

                        v_outputs = model(v_images)
                        v_main_output = (
                            v_outputs[0]
                            if isinstance(v_outputs, (list, tuple))
                            else v_outputs
                        )

                        v_loss = criterion(v_main_output, v_masks)
                        val_loss += v_loss.item()

                        preds = torch.argmax(v_main_output, dim=1)
                        miou_metric.update(preds, v_masks)

            avg_val_loss = val_loss / len(val_loader)
            current_miou = miou_metric.compute().item()

            print(
                f"\n[Step {global_step}] Val Loss: {avg_val_loss:.4f}, mIoU: {current_miou:.4f}"
            )

            # 1. SAVE WEIGHTS FOR EACH INTERVAL (Numbered by global step)
            interval_filename = os.path.join(
                checkpoint_dir, f"unet_b1_step{global_step}_miou{current_miou:.4f}.pth"
            )
            torch.save(
                {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "miou": current_miou,
                },
                interval_filename,
            )
            print(f"💾 Interval weight saved: {interval_filename}")

            # 2. SAVE BEST MODEL
            scheduler.step(avg_val_loss)
            if current_miou > best_miou:
                best_miou = current_miou
                torch.save(model.state_dict(), save_path_best)
                print(f"Saved New Best Model! (mIoU: {best_miou:.4f})")

            model.train()

print("\ Training Complete!")
