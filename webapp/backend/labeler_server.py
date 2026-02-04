import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_ROOT = "/srv/shared/text-image-segmentation/synthetic_dataset"
STATUS_FILE = os.path.join(DATASET_ROOT, "review_status.json")

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=4)

class ReviewUpdate(BaseModel):
    sample_name: str
    status: str  # "approved", "rejected", "pending"

@app.get("/samples")
def get_samples():
    train_images = os.path.join(DATASET_ROOT, "train", "images")
    if not os.path.exists(train_images):
        return []

    images = [f for f in os.listdir(train_images) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    statuses = load_status()

    samples = []
    for img in images:
        samples.append({
            "name": img,
            "status": statuses.get(img, "pending")
        })

    return samples

@app.get("/image/{sample_name}")
def get_image(sample_name: str):
    path = os.path.join(DATASET_ROOT, "train", "images", sample_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/mask/{sample_name}")
def get_mask(sample_name: str):
    # Try common mask extensions if the image extension doesn't match
    base_name = os.path.splitext(sample_name)[0]
    mask_path = os.path.join(DATASET_ROOT, "train", "masks", base_name + ".png")

    if not os.path.exists(mask_path):
        # Retry with original extension if needed
        mask_path = os.path.join(DATASET_ROOT, "train", "masks", sample_name)

    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Mask not found")
    return FileResponse(mask_path)

@app.post("/review")
def update_review(update: ReviewUpdate):
    statuses = load_status()
    statuses[update.sample_name] = update.status
    save_status(statuses)
    return {"message": "Status updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
