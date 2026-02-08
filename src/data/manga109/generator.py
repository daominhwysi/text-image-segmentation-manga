import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm

def get_binary_mask(mask_img):
    """
    Extracts text (black) and SFX (pink) from the Manga109 mask image.
    Pink: [100, 0, 150] to [255, 180, 255]
    Black: [0, 0, 0] to [60, 60, 60]
    """
    # Black (Text)
    text_mask = cv2.inRange(mask_img, np.array([0, 0, 0]), np.array([60, 60, 60]))
    # Pink (SFX)
    sfx_mask = cv2.inRange(mask_img, np.array([100, 0, 150]), np.array([255, 180, 255]))

    # Combine
    combined = cv2.bitwise_or(text_mask, sfx_mask)
    return combined

def extract_rois(
    image_dir='resources/manga109/images',
    mask_dir='resources/manga109/masks',
    label_dir='resources/manga109/labels',
    output_root='output/manga109_roi_text_seg',
    train_ratio=0.9,
    target_size=256,
    max_original_crop_size=640,
    zoom_range=(1.2, 2.0),
    num_pages=None
):
    os.makedirs(output_root, exist_ok=True)

    label_paths = sorted(glob(os.path.join(label_dir, '*.txt')))
    if not label_paths:
        print(f"No labels found in {label_dir}")
        return

    # Extract base names (page names)
    page_names = [os.path.splitext(os.path.basename(p))[0] for p in label_paths]

    if num_pages:
        page_names = page_names[:num_pages]
        print(f"Testing with {len(page_names)} pages")

    # Split page-level
    temp_page_names = page_names.copy()
    random.Random(42).shuffle(temp_page_names)
    split_idx = int(len(temp_page_names) * train_ratio)
    train_pages = temp_page_names[:split_idx]
    test_pages = temp_page_names[split_idx:]

    splits = {'train': train_pages, 'test': test_pages}

    for split_name, pages in splits.items():
        img_out = os.path.join(output_root, split_name, 'images')
        mask_out = os.path.join(output_root, split_name, 'masks')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        print(f"Processing {split_name} split ({len(pages)} pages)...")

        sample_idx = 0
        for page in tqdm(pages):
            img_path = os.path.join(image_dir, f"{page}.jpg")
            mask_path = os.path.join(mask_dir, f"{page}.png")
            label_path = os.path.join(label_dir, f"{page}.txt")

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Missing image or mask for page {page}")
                continue

            img = cv2.imread(img_path)
            mask_img = cv2.imread(mask_path)

            if img is None or mask_img is None:
                continue

            h_orig, w_orig = img.shape[:2]

            # Extract combined binary mask once for the whole page
            binary_mask = get_binary_mask(mask_img)

            # Read labels (YOLO format: class_id x_center y_center width height)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts: continue

                # class_id = int(parts[0]) # We merge all classes to 0
                x_center, y_center, w_norm, h_norm = map(float, parts[1:])

                # Convert to pixel coords
                cx = int(x_center * w_orig)
                cy = int(y_center * h_orig)
                bw = int(w_norm * w_orig)
                bh = int(h_norm * h_orig)

                # Determine crop size (square, centered on bbox)
                zoom = random.uniform(*zoom_range)
                crop_size = int(max(bw, bh) * zoom)

                # Constraint: max 640px
                if crop_size > max_original_crop_size:
                    crop_size = max_original_crop_size

                # Calculate crop bounds
                x1 = cx - crop_size // 2
                y1 = cy - crop_size // 2
                x2 = x1 + crop_size
                y2 = y1 + crop_size

                # Padding handling (if crop goes outside image)
                # We'll use reflection padding for the image and zero padding for the mask
                # But it's easier to just adjust the bounds and crop, then pad if needed

                def safe_crop(data, bounds, pad_value=0):
                    x1, y1, x2, y2 = bounds
                    h, w = data.shape[:2]

                    # Target crop area
                    crop = np.full((y2-y1, x2-x1) + data.shape[2:], pad_value, dtype=data.dtype)

                    # Source area within image
                    src_x1 = max(0, x1)
                    src_y1 = max(0, y1)
                    src_x2 = min(w, x2)
                    src_y2 = min(h, y2)

                    # Destination area within crop
                    dst_x1 = src_x1 - x1
                    dst_y1 = src_y1 - y1
                    dst_x2 = dst_x1 + (src_x2 - src_x1)
                    dst_y2 = dst_y1 + (src_y2 - src_y1)

                    if src_x2 > src_x1 and src_y2 > src_y1:
                        crop[dst_y1:dst_y2, dst_x1:dst_x2] = data[src_y1:src_y2, src_x1:src_x2]

                    return crop

                # For image, we can use reflect padding if we want more natural backgrounds
                # For now let's stick to black padding as it's safer for segmentation training
                roi_img = safe_crop(img, (x1, y1, x2, y2), pad_value=0)
                roi_mask = safe_crop(binary_mask, (x1, y1, x2, y2), pad_value=0)

                # Resize to target size (256x256)
                roi_img_resized = cv2.resize(roi_img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

                # Resize mask with BILINEAR to get soft edges (soft mask)
                roi_mask_resized = cv2.resize(roi_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

                # Save
                file_id = f"{page}_{sample_idx:03d}"
                cv2.imwrite(os.path.join(img_out, f"{file_id}.jpg"), roi_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(mask_out, f"{file_id}.png"), roi_mask_resized)

                sample_idx += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pages", type=int, default=None)
    parser.add_argument("--output", type=str, default='output/manga109_roi_text_seg')
    args = parser.parse_args()

    extract_rois(num_pages=args.num_pages, output_root=args.output)
