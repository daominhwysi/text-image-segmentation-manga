import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from src.automatic_review.dataset import ReviewDataset
from src.automatic_review.model import ReviewClassifier

def train():
    # Config
    DATASET_ROOT = "/srv/shared/text-image-segmentation/output/synthetic_dataset"
    STATUS_JSON = os.path.join(DATASET_ROOT, "review_status_train.json")
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = "checkpoints/automatic_review"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

    # Dataset
    full_dataset = ReviewDataset(
        DATASET_ROOT,
        STATUS_JSON,
        split="train",
        transform=transform,
        mask_transform=mask_transform
    )

    if len(full_dataset) == 0:
        print("No labeled samples found in status JSON.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = ReviewClassifier(backbone="resnet18", pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    print(f"Starting training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            # Concatenate image and mask
            inputs = torch.cat([images, masks], dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)
                inputs = torch.cat([images, masks], dim=1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_reviewer.pth"))
            print(f"New best model saved with accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    train()
