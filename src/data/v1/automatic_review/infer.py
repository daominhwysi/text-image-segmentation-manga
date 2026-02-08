import os
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from src.automatic_review.model import ReviewClassifier

class Reviewer:
    def __init__(self, checkpoint_path="checkpoints/mb4_automatic_review.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        if not os.path.exists(self.checkpoint_path):
            self.model = None
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}. Reviewer disabled.")
        else:
            self.model = ReviewClassifier(pretrained=False).to(self.device)
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            self.model.eval()

        # Transforms for MobileNetV4 (usually 256x256)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def predict(self, images, masks):
        """
        images: list of PIL Images (RGB)
        masks: list of PIL Images (L)
        returns: list of probabilities (float)
        """
        if self.model is None:
            return [1.0] * len(images) # Default to approved if no model

        batch_inputs = []
        for img, mask in zip(images, masks):
            img_tensor = self.transform(img.convert("RGB"))
            mask_tensor = self.mask_transform(mask.convert("L"))
            input_tensor = torch.cat([img_tensor, mask_tensor], dim=0)
            batch_inputs.append(input_tensor)

        if not batch_inputs:
            return []

        inputs = torch.stack(batch_inputs).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        return probs.tolist()

def infer():
    # Config
    DATASET_ROOT = "/srv/shared/text-image-segmentation/output/synthetic_dataset"
    STATUS_JSON = os.path.join(DATASET_ROOT, "review_status_train.json")
    SPLIT = "train"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "checkpoints/mb4_automatic_review.pth"

    # Reviewer
    reviewer = Reviewer(CHECKPOINT, DEVICE)
    if reviewer.model is None:
        return

    # Load existing status
    if os.path.exists(STATUS_JSON):
        with open(STATUS_JSON, "r") as f:
            status_data = json.load(f)
    else:
        status_data = {}

    images_dir = os.path.join(DATASET_ROOT, SPLIT, "images")
    masks_dir = os.path.join(DATASET_ROOT, SPLIT, "masks")

    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    pending_images = [img for img in all_images if img not in status_data]

    print(f"Found {len(pending_images)} pending images to auto-review.")

    if not pending_images:
        return

    # Process in batches for speed
    BATCH_SIZE = 64
    for i in tqdm(range(0, len(pending_images), BATCH_SIZE)):
        batch_names = pending_images[i:i + BATCH_SIZE]

        images = []
        masks = []
        for name in batch_names:
            img_path = os.path.join(images_dir, name)
            mask_path = os.path.join(masks_dir, os.path.splitext(name)[0] + ".png")
            images.append(Image.open(img_path))
            masks.append(Image.open(mask_path))

        probs = reviewer.predict(images, masks)

        for name, prob in zip(batch_names, probs):
            status = "approved" if prob > 0.5 else "rejected"
            status_data[name] = status

    # Save results
    with open(STATUS_JSON, "w") as f:
        json.dump(status_data, f, indent=4)

    print(f"Auto-review complete. Updated {STATUS_JSON}")

if __name__ == "__main__":
    infer()
