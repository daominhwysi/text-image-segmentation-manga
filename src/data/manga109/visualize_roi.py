import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm

def visualize_dataset(
    dataset_root='output/manga109_roi_text_seg',
    output_dir='output/manga109_roi_text_seg_viz',
    num_samples=50,
    split='train'
):
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(dataset_root, split, 'images')
    mask_dir = os.path.join(dataset_root, split, 'masks')

    img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    # Shuffle and pick samples
    random.seed(42)
    random.shuffle(img_paths)
    samples = img_paths[:num_samples]

    print(f"Visualizing {len(samples)} samples to {output_dir}...")

    for img_path in tqdm(samples):
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, base_name.replace('.jpg', '.png'))

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # Create overlay
        # Mask is grayscale, color it red for visibility
        color_mask = np.zeros_like(img)
        color_mask[:, :, 2] = mask # Red channel

        # Blend
        alpha = 0.5
        overlay = cv2.addWeighted(img, 1.0, color_mask, alpha, 0)

        # Side by side
        # Convert mask to BGR for concatenation
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([img, mask_bgr, overlay])

        # Save
        cv2.imwrite(os.path.join(output_dir, base_name), combined)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='output/manga109_roi_text_seg')
    parser.add_argument("--output", type=str, default='output/manga109_roi_text_seg_viz')
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    visualize_dataset(dataset_root=args.dataset, output_dir=args.output, num_samples=args.n)
