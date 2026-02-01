import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models import Unet_MobileNetV4 # Ensure this matches your file structure

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_manga_unet.pth"
INPUT_FOLDER = "representative_sample_folder"       # Folder containing your raw images
OUTPUT_FOLDER = "inference_output" # Where results will be saved
IMG_SIZE = 256
NUM_CLASSES = 2

# Define colors for visualization (BGR format for OpenCV)
# Class 0 (Background): Black
# Class 1 (Text): Green
COLORS = np.array([
    [0, 0, 0],    # Background
    [0, 255, 0],  # Text
], dtype=np.uint8)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. PREPROCESSING & UTILS
# ==========================================
def get_inference_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def decode_mask(mask_index):
    """Converts class indices to an RGB color mask."""
    return COLORS[mask_index]

# ==========================================
# 3. INFERENCE FUNCTION
# ==========================================
def run_inference():
    # 1. Load Model
    print(f"📂 Loading model from {CHECKPOINT_PATH}...")
    # Make sure 'Unet_MobileNetV4' matches the class name in your src.models
    model = Unet_MobileNetV4(num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    transforms = get_inference_transforms()

    # 2. Get Images
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(image_extensions)]

    print(f"🚀 Found {len(image_files)} images. Starting inference...")

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(INPUT_FOLDER, img_name)

            # Load original image
            orig_bgr = cv2.imread(img_path)
            if orig_bgr is None: continue

            h_orig, w_orig = orig_bgr.shape[:2]
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

            # Preprocess
            input_tensor = transforms(image=orig_rgb)["image"].unsqueeze(0).to(DEVICE)

            # Predict
            logits = model(input_tensor)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # [H, W]

            # Post-process mask
            # Resize mask back to original image size for better visualization
            mask_vis = decode_mask(preds)
            mask_vis_resized = cv2.resize(mask_vis, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # Create an overlay (optional)
            # Mix 70% original image and 30% mask for areas where text is detected
            overlay = orig_bgr.copy()
            text_mask = (preds == 1).astype(np.uint8)
            text_mask_resized = cv2.resize(text_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            overlay[text_mask_resized > 0] = cv2.addWeighted(orig_bgr, 0.5, mask_vis_resized, 0.5, 0)[text_mask_resized > 0]

            # Stack for visual comparison: [Original | Mask | Overlay]
            combined = np.hstack([orig_bgr, mask_vis_resized, overlay])

            # Save
            save_path = os.path.join(OUTPUT_FOLDER, f"res_{img_name}")
            cv2.imwrite(save_path, combined)

    print(f"✅ Done! Results saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    # Ensure the input folder exists before running
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Input folder '{INPUT_FOLDER}' not found.")
    else:
        run_inference()
