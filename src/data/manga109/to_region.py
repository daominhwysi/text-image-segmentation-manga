import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_robust_boxes(image, color_type='text'):
    # 1. Isolate the color mask
    if color_type == 'text':
        # Black (Text)
        mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([60, 60, 60]))
    else:
        # Pink (SFX) - Broadened range for robustness
        mask = cv2.inRange(image, np.array([100, 0, 150]), np.array([255, 180, 255]))

    # 2. Find individual character/stroke sizes to determine scale
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask

    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.contourArea(c) > 5]
    if not heights:
        return [], mask

    # Calculate median height (the 'scale' of the text)
    median_h = np.median(heights)

    # 3. Define dynamic kernel size based on the scale of the text
    # We want to bridge gaps roughly 150% of the character height
    gap_size = int(median_h * 1.5)
    if gap_size < 1: gap_size = 1

    kernel = np.ones((gap_size, gap_size), np.uint8)

    # 4. Dilate to merge
    # We use 'morphologyEx' with CLOSE to fill gaps without growing the outer boundary too much
    dilated = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Further expand slightly to ensure grouping
    dilated = cv2.dilate(dilated, kernel, iterations=1)

    # 5. Get final paragraph contours
    final_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in final_contours:
        if cv2.contourArea(cnt) > (median_h * median_h * 0.5): # Filter tiny artifacts
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    return boxes, dilated

def to_yolo(box, img_w, img_h):
    x, y, w, h = box
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

if __name__ == "__main__":
    import os
    from glob import glob
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    mask_dir = 'resources/manga109/masks'
    output_dir = 'resources/manga109/visualization'
    label_dir = 'resources/manga109/labels'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))

    print(f"Found {len(mask_paths)} images in {mask_dir}")

    for input_path in tqdm(mask_paths):
        img = cv2.imread(input_path)

        if img is None:
            print(f"Error: Could not load image at {input_path}")
            continue

        h_img, w_img = img.shape[:2]

        # Get text paragraphs
        text_boxes, text_mask = get_robust_boxes(img, 'text')

        # Get SFX groups
        sfx_boxes, sfx_mask = get_robust_boxes(img, 'sfx')

        # --- YOLO Labels ---
        filename = os.path.basename(input_path)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        with open(label_path, 'w') as f:
            for box in text_boxes:
                # Class 0: Text
                f.write(f"0 {to_yolo(box, w_img, h_img)}\n")
            for box in sfx_boxes:
                # Class 1: SFX
                f.write(f"1 {to_yolo(box, w_img, h_img)}\n")

        # --- Visualization ---
        viz = img.copy()
        for (x, y, w, h) in text_boxes:
            cv2.rectangle(viz, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue Text
        for (x, y, w, h) in sfx_boxes:
            cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green SFX

        output_path = os.path.join(output_dir, filename)

        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)) # Convert to RGB for matplotlib
        plt.title(f"Merged Detection - {filename}")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close() # Close to free memory

    print(f"All visualizations saved to {output_dir}")
    print(f"All labels saved to {label_dir}")
