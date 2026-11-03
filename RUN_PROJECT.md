# Run the Forensic Sketch Reconstruction Project

## 1. Requirements
- Windows 10/11 or Linux/macOS
- Python 3.10 or newer
- At least 8 GB RAM recommended
- GPU is optional; CPU inference works but may be slower

## 2. Install (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install the CUDA-enabled PyTorch build appropriate for your driver using the official PyTorch installation selector. The default installation may be CPU-only.

## 3. Start the API and dashboard
```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8002 --workers 1
```

Open:
- Dashboard: http://127.0.0.1:8002/
- Health/model availability: http://127.0.0.1:8002/health
- API documentation: http://127.0.0.1:8002/docs

Use one worker because the model pipeline is loaded into process memory.

## 4. Run tests
```powershell
python -m pytest tests/ -v
```

## 5. Train missing custom neural models
The separately supplied project ZIP contains paired raw/processed data, but the three custom PyTorch checkpoint files are absent. Train them separately:
```powershell
python training/train_models.py --model pix2pix --device auto
python training/train_models.py --model unet --device auto
python training/train_models.py --model resnet_gan --device auto
```

Training may take a long time, especially on CPU. Restart the API after training and inspect /health to confirm each checkpoint loads. Do not assume a checkpoint exists merely because model source or a training script exists.

## 6. Important interpretation note
A sketch does not contain enough information to recover a unique real person's face, especially when only the eyes or a partial face are drawn. Reconstructed faces are plausible model estimates, not verified identities or ground-truth biometric records.
