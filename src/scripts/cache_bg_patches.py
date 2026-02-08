import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def extract_patches_from_image(img_path, label_path, output_dir, patch_size=512, max_patches_per_img=20, max_attempts=100):
    image = cv2.imread(img_path)
    if image is None:
        return 0

    h, w, _ = image.shape
    if h < patch_size or w < patch_size:
        return 0

    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    _, xc, yc, bw, bh = map(float, parts)
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    bboxes.append((x1, y1, x2, y2))
                except ValueError:
                    continue

    extracted_count = 0
    img_basename = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(max_attempts):
        if extracted_count >= max_patches_per_img:
            break

        rx1 = random.randint(0, w - patch_size)
        ry1 = random.randint(0, h - patch_size)
        rx2, ry2 = rx1 + patch_size, ry1 + patch_size

        collision = False
        for bx1, by1, bx2, by2 in bboxes:
            # Check overlap
            if not (rx2 <= bx1 or rx1 >= bx2 or ry2 <= by1 or ry1 >= by2):
                collision = True
                break

        if not collision:
            patch = image[ry1:ry2, rx1:rx2]
            # Simple check for "entropy" or local variance to avoid pure empty patches if possible?
            # For now, just save it.
            patch_name = f"{img_basename}_p{extracted_count:02d}.jpg"
            cv2.imwrite(os.path.join(output_dir, patch_name), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            extracted_count += 1

    return extracted_count

def main():
    dataset_root = "resources/Speech-bubble-6/train"
    img_dir = os.path.join(dataset_root, "images")
    lbl_dir = os.path.join(dataset_root, "labels")
    output_dir = "resources/bg_patches"

    os.makedirs(output_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(img_files)} images. Extracting patches...")

    tasks = []
    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(lbl_dir, label_name)
        tasks.append((img_path, label_path, output_dir))

    total_patches = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(extract_patches_from_image, *task) for task in tasks]
        for result in tqdm(futures):
            total_patches += result.result()

    print(f"Finished. Total patches extracted: {total_patches}")

if __name__ == "__main__":
    main()
