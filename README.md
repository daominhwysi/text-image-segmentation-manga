# Text-Image Segmentation for Manga
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/collections/Daominhwysi/manga-text-seg)

<!-- [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg)](https://huggingface.co/models) -->

This project provides a robust pipeline for **text segmentation** and **OCR** in images, specifically tailored for manga. It includes tools for generating high-quality synthetic datasets, training state-of-the-art segmentation models (U-Net variants with modern backbones), and performing inference with OpenOCR.

## 🚀 Key Features

*   **Synthetic Data Generation (v1)**: Create large-scale datasets by compositing realistic text onto background images with advanced augmentations and background ROI management.
    * **YOLO format bounding box refinement**: Use trained segmentation models to refine YOLO format bounding boxes.
    * **Manual Review Webapp**: A modern React + FastAPI tool for manual dataset review, featuring rapid keyboard-driven approval/rejection.
    * **review_sample_classifier**: A classifier model trained on a small subset of manually reviewed data to assist in the review process.
*   **Modern Segmentation Architectures**: Support for high-performance U-Net backbones:
    *   **EfficientViT-B2** for state-of-the-art accuracy.
    *   **MobileNetV4** for a balance of speed and performance.
*   **Advanced Training Techniques**:
    *   **Deep Supervision**: Multi-scale loss integration for stable convergence.
    *   **Mixed Precision**: Cuda-optimized training using `torch.amp`.
    *   **Warm-up Phase**: Frozen backbone training followed by full fine-tuning.

---

## 🛠️ Installation

### Using Conda
We provide environment files for both GPU and CPU:

**For GPU (NVIDIA CUDA):**
```bash
conda env create -f environment_cuda.yml
conda activate text-image-seg
```

**For CPU:**
```bash
conda env create -f environment_cpu.yml
conda activate text-image-seg
```

---

## 📖 Usage
### 0. Run the webapp to review the samples
```bash
# Backend
uvicorn webapp.backend.server:app --host 0.0.0.0 --port 8000 --reload
# Frontend
cd webapp/frontend && npm run dev
```
### 1. Generate Synthetic Dataset
Before training, generate a synthetic dataset using background images and fonts.
1.  download the font and background with `python -m src.scripts.download_data`.
2.  Refine the bounding box with `python -m src.scripts.refine_labels`.
3.  Cache the background with `python -m src.scripts.cache_bg_patches`.
4.  Run the generator: `python -m src.data.v1.generator --total-samples 50000`.


## 🧠 Model Architectures

Available models in `src.segment.models`:

| Model | Backbone | Best For |
| :--- | :--- | :--- |
| `Unet_EfficientViT_B2` | EfficientViT-B2 | Maximum Accuracy |
| `Unet_MobileNetV4` | MobileNetV4 | Ideal for production |

---

## 📄 License
This project is for educational and research purposes.
