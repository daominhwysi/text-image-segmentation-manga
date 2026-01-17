# Text-Image Segmentation for Manga

This project provides a robust pipeline for **text segmentation** in images, specifically tailored for manga. It includes tools for generating high-quality synthetic datasets and training state-of-the-art segmentation models (U-Net variants with modern backbones).

## 🚀 Key Features

*   **Synthetic Data Generation**: Create large-scale datasets by compositing realistic text onto background images with various augmentations.
*   **Diverse Architectures**: Support for multiple U-Net backbones:
    *   **Segformer** (MIT-B0, MIT-B1) for high accuracy.
    *   **MobileNetV3** (Small, Large) for efficient, lightweight inference.
*   **Advanced Modules**: Integrated **ASPP** (Atrous Spatial Pyramid Pooling) and **SCSE** (Spatial/Channel Squeeze & Excitation) for better context awareness.
*   **Deep Supervision**: Multi-scale loss integration for stable and faster convergence.
*   **Cuda Optimized**: Full support for mixed-precision training (`torch.cuda.amp`).

---

## 📁 Project Structure

```text
.
├── src/
│   ├── data/
│   │   ├── v1/                # Synthetic data generation logic
│   │   │   ├── generator.py   # Main entry for dataset generation
│   │   │   ├── text_render.py # Text rendering utilities
│   │   │   └── bg_manager.py  # Background ROI management
│   │   └── v2/                # OCR integration (experimental)
│   └── model/
│       ├── models.py          # U-Net architecture definitions
│       ├── train.py           # Training script
│       └── test.py            # Evaluation and inference testing
├── scripts/                   # Utility scripts (download/upload)
├── resource/                  # Base images and fonts
├── synthetic_dataset/         # Default output for generated data
└── environment_cuda.yml       # Conda environment config
```

---

## 🛠️ Installation

### Using Conda
We provide two environment files based on your hardware:

**For GPU (NVIDIA CUDA 12.1):**
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

### 1. Generate Synthetic Dataset
Before training, you need to generate a synthetic dataset using your background images and fonts.

1.  Place background images in `resource/your_bg_data/`
2.  Place fonts in `resource/fonts/`
3.  Configure and run:
    ```bash
    python -m src.data.v1.generator
    ```
    *This will generate 100,000 samples by default in `synthetic_dataset/`.*

### 2. Training the Model
To start training the segmentation model (default uses `Unet_B1`):

```bash
python -m src.model.train
```

*   **Checkpoints**: Saved in the `checkpoints/` directory.
*   **Best Model**: The best performing model (mIoU) is saved as `best_unet_256_b1.pth`.

### 3. Inference / Testing
You can run a quick test or evaluate on your test set:

```bash
python -m src.model.test
```

---

## 🧠 Model Architectures

The project supports several configurations in `src/model/models.py`:

| Model | Backbone | Target |
| :--- | :--- | :--- |
| `Unet_B0` | Segformer MIT-B0 | Balanced Performance |
| `Unet_B1` | Segformer MIT-B1 | High Accuracy |
| `Unet_MobileNet_small` | MobileNetV3 Small | Ultra Lightweight |
| `Unet_MobileNet_large` | MobileNetV3 Large | Mobile-Efficient |

---

## 📄 License
This project is for educational and research purposes.
