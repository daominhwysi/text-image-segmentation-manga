import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

def get_masks(mask_img):
    """
    Extracts text (black) and SFX (pink) from the Manga109 mask image.
    Pink: [100, 0, 150] to [255, 180, 255] (In BGR: [150, 0, 100] to [255, 180, 255])
    Black: [0, 0, 0] to [60, 60, 60]
    """
    # OpenCV reads in BGR.
    # Black (Text) - usually [0,0,0]
    text_mask = cv2.inRange(mask_img, np.array([0, 0, 0]), np.array([60, 60, 60]))

    # Pink (SFX) - In RGB it's Pink, in BGR we need to be careful.
    # The range in generator.py was [100, 0, 150] to [255, 180, 255]
    # Let's assume that range was meant for BGR if it was used with cv2.imread directly.
    sfx_mask = cv2.inRange(mask_img, np.array([100, 0, 150]), np.array([255, 180, 255]))

    return text_mask, sfx_mask

def visualize_masks(img, mask_img, alpha=0.5):
    """
    Overlays masks on the image.
    Text: Blue
    SFX: Green
    """
    text_mask, sfx_mask = get_masks(mask_img)

    overlay = img.copy()

    # Text in Blue (BGR: 255, 0, 0)
    overlay[text_mask > 0] = [255, 0, 0]

    # SFX in Green (BGR: 0, 255, 0)
    overlay[sfx_mask > 0] = [0, 255, 0]

    # Blend
    viz = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    return viz

def main():
    parser = argparse.ArgumentParser(description="Visualize Manga109 original images with mask overlays")
    parser.add_argument("--img_dir", type=str, default="resources/manga109/images", help="Path to images")
    parser.add_argument("--mask_dir", type=str, default="resources/manga109/masks", help="Path to masks")
    parser.add_argument("--output_dir", type=str, default="resources/manga109/viz_mask", help="Path to save visualizations")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to visualize")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha for blending")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img_paths = sorted(glob(os.path.join(args.img_dir, "*.jpg")))
    if not img_paths:
        print(f"No images found in {args.img_dir}")
        return

    if args.limit:
        img_paths = img_paths[:args.limit]
        print(f"Visualizing first {args.limit} images")
    else:
        print(f"Visualizing all {len(img_paths)} images")

    for img_path in tqdm(img_paths):
        base_name = os.path.basename(img_path)
        mask_filename = os.path.splitext(base_name)[0] + ".png"
        mask_path = os.path.join(args.mask_dir, mask_filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path)
            if mask_img is not None:
                # Resize mask to image size if they differ (shouldn't happen in Manga109 but good for robustness)
                if mask_img.shape[:2] != img.shape[:2]:
                    mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                viz = visualize_masks(img, mask_img, alpha=args.alpha)
            else:
                print(f"Failed to read mask: {mask_path}")
                viz = img
        else:
            cv2.putText(img, "NO MASK FOUND", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            viz = img

        output_path = os.path.join(args.output_dir, base_name)
        cv2.imwrite(output_path, viz)

    print(f"\nVisualization complete. Results saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
