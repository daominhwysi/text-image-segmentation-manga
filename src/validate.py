import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

from src.segment.models import Unet_MobileNetV4

def validate():
    # Configuration
    MODEL_PATH = "checkpoints/best_manga_unet.pth"
    IMAGE_DIR = "representative_sample_folder"
    OUTPUT_DIR = "visualization"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = (256, 256)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = Unet_MobileNetV4(num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 2. Preprocessing
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Process Images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {IMAGE_DIR}. Starting inference...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)

        try:
            # Load image
            original_img_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = original_img_pil.size

            # Prepare input
            input_tensor = transform(original_img_pil).unsqueeze(0).to(DEVICE)

            # Inference
            with torch.no_grad():
                output = model(input_tensor)
                # Output shape: [1, 2, H, W]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Resize pred back to original size for visualization
            pred_resized = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Create visualization
            img_np = np.array(original_img_pil)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            mask_colored = np.zeros_like(img_cv)
            mask_colored[pred_resized == 1] = [0, 0, 255] # Red in BGR

            # Overlay
            overlay = cv2.addWeighted(img_cv, 0.7, mask_colored, 0.3, 0)

            # Save
            save_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(save_path, overlay)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print(f"\nValidation complete. Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    validate()
