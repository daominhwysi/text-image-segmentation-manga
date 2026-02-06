import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ReviewDataset(Dataset):
    def __init__(self, dataset_root, status_json, split="train", transform=None, mask_transform=None):
        self.dataset_root = dataset_root
        self.split = "test" if split == "valid" else split
        self.images_dir = os.path.join(dataset_root, self.split, "images")
        self.masks_dir = os.path.join(dataset_root, self.split, "masks")

        with open(status_json, "r") as f:
            self.status_data = json.load(f)

        # Filter only approved and rejected samples
        self.samples = [
            (name, 1 if status == "approved" else 0)
            for name, status in self.status_data.items()
            if status in ["approved", "rejected"]
        ]

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default mask processing: resize and to tensor
            mask = mask.resize((224, 224), Image.BILINEAR)
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0

        # Combine image and mask?
        # For now let's just use image, but maybe concatenate them?
        # A 4-channel input (RGB + Mask) could be very effective.

        return image, mask, label
