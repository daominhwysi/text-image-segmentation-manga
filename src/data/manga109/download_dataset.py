from datasets import load_dataset
from pathlib import Path
import os
from PIL import Image
from tqdm.auto import tqdm
repo_id = "Daominhwysi/manga-109-text-seg"
OUTPUT_FOLDER = "resources/manga109"

print(f"Downloading dataset from {repo_id}...")
downloaded_dataset = load_dataset(repo_id)

print("Dataset downloaded successfully!")
print(downloaded_dataset)

output_root = Path(OUTPUT_FOLDER)

target_img_dir = output_root / "images"
target_mask_dir = output_root / "masks"

target_img_dir.mkdir(parents=True, exist_ok=True)
target_mask_dir.mkdir(parents=True, exist_ok=True)

print(f"Starting flattened reconstruction to {output_root.absolute()}...")

for item in tqdm(downloaded_dataset['train']):
    manga_name = item['manga_name']
    page_id = item['page_id']

    file_name = f"{manga_name}_{page_id}"

    image_path = target_img_dir / f"{file_name}.jpg"
    item['image'].save(image_path)

    # Save mask directly to the masks folder
    mask_path = target_mask_dir / f"{file_name}.png"
    item['mask'].save(mask_path)

print(f"Reconstruction complete! Total files: {len(os.listdir(target_img_dir))}")
