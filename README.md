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

## 📁 Project Structure

```text
.
├── src/
│   ├── data/
│   │   ├── v1/                # Synthetic data generation logic
│   │   │   ├── generator.py   # Main entry for dataset generation
│   │   │   ├── text_render.py # Text rendering utilities
│   │   │   ├── bg_manager.py  # Background ROI management
│   │   │   └── augment_text.py # Text-specific augmentations
│   │   └── openocr/           # OpenOCR inference (ONNX)
│   │       ├── det_infer.py   # Text detection
│   │       ├── rec_infer.py   # Text recognition
│   │       └── e2e_infer.py   # End-to-end OCR pipeline
│   ├── segment/               # Segmentation model logic
│   │   ├── models.py          # U-Net architecture definitions
│   │   ├── train.py           # Training script
│   │   └── test.py            # Inference and evaluation
│   └── validate.py            # Validation utilities

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

1.  Place background images in `resource/your_bg_data/`
2.  Place fonts in `resource/fonts/`
3.  Run the generator:
    ```bash
    python -m src.data.v1.generator --total-samples 50000
    ```

### 2. Training the Model
To train the segmentation model (default uses `Unet_MobileNetV4`):

```bash
python -m src.segment.train
```
*   **Phase 1**: Warm-up with frozen backbone (5 epochs).
*   **Phase 2**: Full fine-tuning (25 epochs).
*   **Checkpoints**: Best model saved as `best_manga_unet.pth`.

### 3. Segmentation Inference
Run inference on raw images using a trained checkpoint:

```bash
python -m src.segment.test
```
*   Input: `representative_sample_folder/`
*   Output: `inference_output/` (Original | Mask | Overlay)

### 4. OCR Inference (OpenOCR)
Use pre-trained OpenOCR models for detection and recognition:

```bash
# Detection only
python -m src.data.openocr.det_infer

# End-to-end OCR
python -m src.data.openocr.e2e_infer
```

---

## 🧠 Model Architectures

Available models in `src.segment.models`:

| Model | Backbone | Best For |
| :--- | :--- | :--- |
| `Unet_EfficientViT_B2` | EfficientViT-B2 | Maximum Accuracy |
| `Unet_MobileNetV4` | MobileNetV4 | Balanced Speed/Accuracy |

---

## 📄 License
This project is for educational and research purposes.
