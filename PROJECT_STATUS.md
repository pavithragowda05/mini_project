# Project Status: Verified Demonstration State

**Project**: Adaptive Multi-Model Forensic Sketch-to-2D & 3D Face Reconstruction System  
**Workspace**: `C:\Users\user\Downloads\forensic_project-main\forensic_project-main`  
**Runtime**: Python 3.14.7, PyTorch 2.14.1+cpu, OpenCV 5.0.0; CUDA unavailable  
**Dashboard**: `http://127.0.0.1:8003`  
**Backend/API docs**: `http://127.0.0.1:8001` / `http://127.0.0.1:8001/docs`  
**Optional Streamlit UI**: `http://127.0.0.1:8501`  

## Verified Working

- FastAPI health, OpenAPI, static dashboard, and cross-port CORS are working.
- OpenCV imports in the active `.venv`; CPU-only startup succeeds.
- Valid uploads are decoded, resized to 256x256 RGB, converted to model-range values, and returned as `preprocessing_only` when no trained pipeline is available.
- The dashboard displays both the original sketch and the preprocessed preview, labels the preview as not a reconstruction, and does not display candidate scores or 3D downloads when inference is unavailable.
- Unsupported file extensions and corrupt image content return HTTP 400.
- Test result: 8 passed, 1 skipped. The skip is the dataset-presence test because research data is not included.
- Mesh generation/export code is exercised by tests using the bundled sample. Its output is a parametric head shell with heuristic image relief, not trained, identity-specific 3D inference.

## Not Available

- No trained checkpoints are present in `checkpoints/`; all five expected model files are reported missing by `/health`.
- No paired raw or processed sketch/photo dataset is present. Model training and held-out evaluation cannot be performed.
- The preprocessing preview is not a generated face. Trained 2D reconstruction, model selection, and identity-specific 3D reconstruction remain unavailable.
- Historical benchmark values and GPU-completion claims in older project material are not verified by this workspace.
- No project-local `.git`, project branches/history, `.gitmodules`, or `.gitattributes` are present. Git resolves upward to `C:\Users\user` and reports zero commits. The existing `backups/` snapshots contain no model weights or archive files, and the README provides no checkpoint download source.

## Expected Checkpoint Formats

- `model_1_pix2pix_best.pth`: PyTorch dictionary with `netG` and `netD` state dictionaries for the current Pix2Pix architecture.
- `model_2_unet_best.pth`: PyTorch generator state dictionary for the current residual U-Net architecture.
- `model_3_resnet_gan_best.pth`: PyTorch dictionary with `netG` and `netD` state dictionaries for the current ResNet-LSGAN architecture.
- `model_4_pca_ridge_best.pkl`: joblib dictionary containing `pca_sketch`, `pca_photo`, `regressor`, `img_shape`, and `is_fitted`.
- `model_5_knn_patch_best.pkl`: joblib dictionary containing `nn_model`, paired sketch/photo patches, patch parameters, `img_shape`, and `is_fitted`.

The neural state dictionaries must match the architectures and layer dimensions in the corresponding `models/model_*.py` files. A matching filename alone does not establish compatibility; each supplied checkpoint must load and pass a finite 256x256 RGB inference smoke test before enabling reconstruction.

## Required Assets

Place compatible trained weights at:

- `checkpoints/model_1_pix2pix_best.pth`
- `checkpoints/model_2_unet_best.pth`
- `checkpoints/model_3_resnet_gan_best.pth`
- `checkpoints/model_4_pca_ridge_best.pkl`
- `checkpoints/model_5_knn_patch_best.pkl`

The preprocessing script expects paired raw files named `{identity}_sketch.png` under `data/raw/sketches/` and `{identity}_photo.png` under `data/raw/photos/`. It writes identity-based splits under `data/processed/{train,val,test}/{sketches,photos}/`. These files must come from the genuine paired dataset; no sample training data is included.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

The main dashboard is served separately on port 8003 and is configured to call the API on port 8001. Start the optional Streamlit UI with `FORENSIC_BACKEND_URL=http://127.0.0.1:8001`.
