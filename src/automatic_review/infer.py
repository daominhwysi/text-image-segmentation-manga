import os
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from src.automatic_review.model import ReviewClassifier

def infer():
    # Config
    DATASET_ROOT = "/srv/shared/text-image-segmentation/output/synthetic_dataset"
    STATUS_JSON = os.path.join(DATASET_ROOT, "review_status_train.json")
    SPLIT = "train"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "checkpoints/automatic_review/best_reviewer.pth"

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found at {CHECKPOINT}. Please train first.")
        return

    # Model
    model = ReviewClassifier(backbone="resnet18", pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load existing status
    if os.path.exists(STATUS_JSON):
        with open(STATUS_JSON, "r") as f:
            status_data = json.load(f)
    else:
        status_data = {}

    images_dir = os.path.join(DATASET_ROOT, SPLIT, "images")
    masks_dir = os.path.join(DATASET_ROOT, SPLIT, "masks")

    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    pending_images = [img for img in all_images if img not in status_data]

    print(f"Found {len(pending_images)} pending images to auto-review.")

    if not pending_images:
        return

    # Process in batches for speed
    BATCH_SIZE = 64
    for i in tqdm(range(0, len(pending_images), BATCH_SIZE)):
        batch_names = pending_images[i:i + BATCH_SIZE]
        batch_inputs = []

        for name in batch_names:
            img_path = os.path.join(images_dir, name)
            mask_path = os.path.join(masks_dir, os.path.splitext(name)[0] + ".png")

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            img_tensor = transform(image)
            mask_tensor = mask_transform(mask)

            input_tensor = torch.cat([img_tensor, mask_tensor], dim=0)
            batch_inputs.append(input_tensor)

        inputs = torch.stack(batch_inputs).to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()

        for name, prob in zip(batch_names, probs):
            status = "approved" if prob > 0.5 else "rejected"
            status_data[name] = status

    # Save results
    with open(STATUS_JSON, "w") as f:
        json.dump(status_data, f, indent=4)

    print(f"Auto-review complete. Updated {STATUS_JSON}")

if __name__ == "__main__":
    infer()
