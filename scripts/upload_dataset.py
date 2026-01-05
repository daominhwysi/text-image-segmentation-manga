import os
from glob import glob
from datasets import Dataset, DatasetDict, Image

# 1. Define paths (adjust folder name if necessary)
base_path = "synthetic_dataset"


def create_split(split_name):
    # Get sorted lists of file paths to ensure they match
    image_paths = sorted(glob(os.path.join(base_path, split_name, "images", "*")))
    mask_paths = sorted(glob(os.path.join(base_path, split_name, "masks", "*")))

    # Check if counts match
    if len(image_paths) != len(mask_paths):
        print(f"Warning: Split '{split_name}' has mismatching counts!")

    # Create a dictionary of the data
    return (
        Dataset.from_dict({"image": image_paths, "label": mask_paths})
        .cast_column("image", Image())
        .cast_column("label", Image())
    )


# 2. Build the DatasetDict (Train and Test)
dataset = DatasetDict({"train": create_split("train"), "test": create_split("test")})

# 3. Push to Hub
# Replace 'your-username/your-dataset-name' with your info
dataset.push_to_hub("Daominhwysi/manga_text_seg")
