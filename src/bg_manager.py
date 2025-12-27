import glob
import os
import random

import cv2


def get_random_non_overlapping_roi(dataset_path, roi_size=(200, 200), max_attempts=100):
    img_dir = os.path.join(dataset_path, "images")
    lbl_dir = os.path.join(dataset_path, "labels")

    img_paths = glob.glob(os.path.join(img_dir, "*"))
    if not img_paths:
        return None, None

    img_path = random.choice(img_paths)
    image = cv2.imread(img_path)
    if image is None:
        return None, img_path

    h, w, _ = image.shape
    rw, rh = roi_size

    # Ensure ROI isn't larger than the source image
    rw, rh = min(rw, w), min(rh, h)

    label_path = os.path.join(
        lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    )
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                bboxes.append((x1, y1, x2, y2))

    # [X logic]: Try to find a random ROI that doesn't collide with bboxes
    for _ in range(max_attempts):
        rx1 = random.randint(0, w - rw)
        ry1 = random.randint(0, h - rh)
        rx2, ry2 = rx1 + rw, ry1 + rh

        collision = False
        for bx1, by1, bx2, by2 in bboxes:
            # Check overlap
            if not (rx2 <= bx1 or rx1 >= bx2 or ry2 <= by1 or ry1 >= by2):
                collision = True
                break

        if not collision:
            roi = image[ry1:ry2, rx1:rx2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            return roi_rgb, img_path

    return None, img_path
