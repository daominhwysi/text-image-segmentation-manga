import os
import random
import cv2
import numpy as np
from collections import defaultdict

class PatchManager:
    def __init__(self, patch_dir="output/bg_patches", split_ratio=0.8, seed=42):
        self.patch_dir = patch_dir
        self.split_ratio = split_ratio

        all_patch_files = []
        if os.path.exists(patch_dir):
            all_patch_files = [f for f in os.listdir(patch_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not all_patch_files:
            print(f"Warning: No patches found in {patch_dir}. Please run src/data/v1/cache_bg_patches.py first.")
            self.train_patches = []
            self.test_patches = []
            return

        # Group patches by source image to avoid leakage
        source_to_patches = defaultdict(list)
        for f in all_patch_files:
            # Format: {img_basename}_p{patch_idx}.jpg
            # We use rpartition to get everything before the last '_p'
            source_name = f.rpartition('_p')[0]
            if not source_name: # fallback if partition failed
                source_name = f
            source_to_patches[source_name].append(os.path.join(patch_dir, f))

        source_images = sorted(list(source_to_patches.keys()))
        random.Random(seed).shuffle(source_images)

        split_idx = int(len(source_images) * split_ratio)
        train_sources = set(source_images[:split_idx])

        self.train_patches = []
        self.test_patches = []

        for source, patches in source_to_patches.items():
            if source in train_sources:
                self.train_patches.extend(patches)
            else:
                self.test_patches.extend(patches)

        print(f"PatchManager: Loaded {len(self.train_patches)} train patches and {len(self.test_patches)} test patches from {len(source_images)} source images.")

    def get_random_patch(self, roi_size=(256, 256), split="train"):
        pool = self.train_patches if split == "train" else self.test_patches
        if not pool:
            return None, None

        # Pick a random patch path
        patch_path = random.choice(pool)
        patch = cv2.imread(patch_path)

        if patch is None:
            return None, None

        h, w, _ = patch.shape
        rw, rh = roi_size

        if rw > w or rh > h:
            patch = cv2.resize(patch, (max(rw, w), max(rh, h)))
            h, w, _ = patch.shape

        # Random crop from the patch
        rx1 = random.randint(0, w - rw)
        ry1 = random.randint(0, h - rh)
        roi = patch[ry1:ry1+rh, rx1:rx1+rw]

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return roi_rgb, patch_path

# Legacy function for compatibility
_global_patch_manager = None

def get_random_non_overlapping_roi(
    dataset_path=None, roi_size=(200, 200), max_attempts=100, detector=None, split="train"
):
    global _global_patch_manager
    if _global_patch_manager is None:
        _global_patch_manager = PatchManager()

    return _global_patch_manager.get_random_patch(roi_size, split=split)
