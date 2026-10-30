# Adaptive Deep Learning and Machine Learning Model Selection in Sketch-to-3D Face Reconstruction for Forensic Applications

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.style=for-the-badge)](LICENSE)

---

## 📌 Project Overview

This repository is a research prototype for sketch-to-image and approximate 3D face-surface experiments. Its five candidate implementations are three PyTorch generators (Pix2Pix, residual U-Net, ResNet-LSGAN) and two classical methods (PCA/Ridge and patch K-NN). The model-selection code ranks available candidate images with hand-designed image heuristics; those scores are not calibrated confidence or validated forensic measurements.

The 3D code builds a closed parametric head shell and modulates its front relief using image intensity and heuristic landmarks. It exports real OBJ, PLY, and GLB mesh files, but it does not infer identity-accurate hidden facial geometry from a sketch.

**Current workspace status:** the checked project copy has a bundled frontend sample image but no `checkpoints/`, `data/raw/`, or `data/processed/` directory. The FastAPI dashboard starts on CPU. Valid uploads are decoded, resized and normalized with OpenCV and returned as `preprocessing_only`; this preview is not a face reconstruction. Trained inference and 3D export remain unavailable until compatible checkpoints are supplied. Invalid or unsupported uploads return HTTP 400. Benchmark values previously listed for this project are not verified by the files currently present.

---

## 🚀 Key Technical Features

- **Candidate ranking:** Scores generated images using visual heuristics only. Scores are request-relative and are not probabilities, ground-truth metrics, or calibrated confidence.
- **CPU-capable inference/training code:** Device selection falls back to CPU when CUDA is unavailable. Training requires real paired data and is never started by the app.
- **Approximate textured 3D surface:** The mesh generator creates a closed parametric head shell with image-conditioned front relief. It is an approximation, not an identity-accurate scan.
- **Interactive dashboard:** FastAPI serves the existing page and Three.js viewer; exports are available as OBJ, PLY, and GLB after a successful reconstruction.

---

## 🛠️ Tech Stack

| Layer | Framework / Library |
| :--- | :--- |
| **Deep Learning** | PyTorch 2.x (CPU or CUDA when available) |
| **Machine Learning** | Scikit-Learn (GradientBoostingRegressor, PCA, Ridge), Joblib |
| **Computer Vision** | OpenCV (`cv2`), Scikit-Image (`skimage.metrics.ssim`), Pillow (PIL) |
| **Backend API** | FastAPI, Uvicorn, PyYAML |
| **Frontend & 3D** | HTML5, CSS3, JavaScript (ES6), Three.js WebGL, OrbitControls |
| **Documentation & Evaluation** | Matplotlib, Pandas, NumPy, ReportLab |

---

## 📁 Repository Structure

```text
Adaptive-Model-Selection-for-Forensic-Sketch-to-3D-Reconstruction/
├── backend/
│   └── app.py                  # FastAPI REST Server Entrypoint
├── inference/
│   └── pipeline.py             # Checkpoint-backed inference orchestrator
├── model_selection/
│   ├── meta_selector.py        # Candidate-image heuristic scorer
│   └── feature_extractor.py    # Sketch Structural Feature Extractor
├── models/
│   ├── model_1_pix2pix.py      # Pix2Pix Conditional GAN
│   ├── model_2_unet.py         # Deep U-Net (Perceptual Loss)
│   ├── model_3_resnet_gan.py   # ResNet Bottleneck LSGAN
│   ├── model_4_pca_ridge.py    # Eigenface Dual-PCA + Ridge
│   └── model_5_knn_patch.py    # Multi-Scale Patch K-NN Regressor
├── face_3d/
│   ├── mesh_generator.py       # 3DMM Depth Map Z(x,y) & Surface Mesh
│   └── exporter.py             # Wavefront .OBJ, .PLY, .GLB Exporter
├── frontend/
│   ├── index.html              # Responsive Dark Dashboard UI
│   ├── app.js                  # Three.js WebGL OrbitControls Script
│   └── styles.css              # Custom Styling
├── configs/
│   └── config.yaml             # Master Configuration File
├── scripts/
│   └── run_paper_experiments.py # Evaluation experiment runner
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- Python with a compatible PyTorch wheel (the tested workspace used Python 3.14 and CPU Torch)
- Node.js is not required; the FastAPI page uses the existing static HTML and CDN-hosted Three.js scripts

### 2. Clone the Repository
```bash
git clone https://github.com/dhanushri432/Adaptive-Model-Selection-for-Forensic-Sketch-to-3D-Reconstruction.git
cd Adaptive-Model-Selection-for-Forensic-Sketch-to-3D-Reconstruction
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
py -3 -m venv .venv
```

In PowerShell, activate and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If script activation is blocked, use the environment interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the Local Web Server
```bash
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```
Open `http://127.0.0.1:8001`; API documentation is at `/docs`. The included `start_demo.ps1` starts the same API on port 8001.

The API serves the main frontend, so Node.js/npm and a separate frontend build are not required. The optional Streamlit frontend can be started in a second PowerShell window:

```powershell
$env:FORENSIC_BACKEND_URL = "http://127.0.0.1:8001"
.\.venv\Scripts\python.exe -m streamlit run app_frontend.py --server.address 127.0.0.1 --server.port 8501
```

### Smoke Checks and Training

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -k "paired_dataset_identity_matching_and_normalization or adaptive_model_selector or sketch_feature_extractor" -q
.\.venv\Scripts\python.exe training\train_models.py --model unet --device cpu --smoke-test
.\.venv\Scripts\python.exe training\train_selector.py --device cpu --smoke-test
```

Training and inference require genuine paired images under `data/processed/{train,val,test}/{sketches,photos}` and compatible checkpoints in `checkpoints/`. The current checked workspace contains neither; these commands therefore stop with explicit missing-data/checkpoint errors until the real assets are supplied. Do not start `--model all` training just to test application startup.

---

## 📊 Benchmark Results

The benchmark figures previously included here are not reproducible from the current workspace: the paired dataset and checkpoints are absent. Do not treat those historical numbers as verified results. Run the evaluation scripts only after installing the real dataset and training/loading compatible checkpoints.

---

## 📜 Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
