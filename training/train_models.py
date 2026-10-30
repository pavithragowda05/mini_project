import os
import sys
import glob
import time
import argparse
import shutil
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_1_pix2pix import Pix2PixModel
from models.model_2_unet import PerceptualUNetModel
from models.model_3_resnet_gan import ResNetLSGANModel
from models.model_4_pca_ridge import PCARidgeModel
from models.model_5_knn_patch import PatchKNNModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SketchPhotoDataset(Dataset):
    """PyTorch Dataset for paired sketch-photo images."""
    def __init__(self, split_dir="data/processed/train", img_size=256):
        sketches = {
            Path(path).stem: path
            for path in glob.glob(os.path.join(split_dir, "sketches", "*.png"))
        }
        photos = {
            Path(path).stem: path
            for path in glob.glob(os.path.join(split_dir, "photos", "*.png"))
        }
        if not sketches or not photos:
            raise FileNotFoundError(
                f"No paired PNG data found in '{split_dir}/sketches' and '{split_dir}/photos'."
            )
        if sketches.keys() != photos.keys():
            raise ValueError(
                f"Unpaired dataset identities in '{split_dir}': "
                f"{len(sketches.keys() - photos.keys())} missing photos, "
                f"{len(photos.keys() - sketches.keys())} missing sketches."
            )
        self.pairs = [(sketches[key], photos[key]) for key in sorted(sketches)]
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sketch_path, photo_path = self.pairs[idx]
        with Image.open(sketch_path) as image:
            s_img = image.convert("RGB")
        with Image.open(photo_path) as image:
            p_img = image.convert("RGB")
        target_size = (self.img_size, self.img_size)
        if s_img.size != target_size:
            s_img = s_img.resize(target_size, Image.Resampling.BILINEAR)
        if p_img.size != target_size:
            p_img = p_img.resize(target_size, Image.Resampling.BILINEAR)

        # Convert PIL to Tensor in range [-1, 1]
        s_arr = (np.array(s_img, dtype=np.float32) / 127.5) - 1.0
        p_arr = (np.array(p_img, dtype=np.float32) / 127.5) - 1.0

        s_tensor = torch.from_numpy(s_arr.transpose(2, 0, 1)).float()
        p_tensor = torch.from_numpy(p_arr.transpose(2, 0, 1)).float()

        return s_tensor, p_tensor

def get_numpy_data(split_dir="data/processed/train"):
    """Loads dataset into memory as numpy arrays for Scikit-Learn models."""
    dataset = SketchPhotoDataset(split_dir)
    if not len(dataset):
        raise ValueError(f"No paired training images found in '{split_dir}'.")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for s, p in loader:
        # Convert (B, C, H, W) -> (B, H, W, C)
        sketch_np = s.numpy().transpose(0, 2, 3, 1)
        photo_np = p.numpy().transpose(0, 2, 3, 1)
        return sketch_np, photo_np
    return None, None

def train_dl_model(model_key, model_obj, train_loader, epochs, checkpoints_dir, device):
    print(f"\n==========================================")
    print(f" TRAINING MODEL: {model_key.upper()}")
    print(f"==========================================")
    
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    
    best_loss = float("inf")
    start_time = time.time()
    save_path = os.path.join(checkpoints_dir, f"{model_key}_best.pth")
    if os.path.isfile(save_path):
        backup_path = f"{save_path}.{time.strftime('%Y%m%d%H%M%S')}.bak"
        shutil.copy2(save_path, backup_path)
        print(f"[!] Existing checkpoint preserved at '{backup_path}'.")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        
        for sketch, photo in train_loader:
            metrics = model_obj.train_step(sketch, photo, scaler=scaler)
            loss_val = metrics.get("loss_G", metrics.get("loss_total", 0.0))
            epoch_loss += loss_val
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        elapsed = time.time() - start_time
        
        print(f"[{model_key.upper()}] Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - Elapsed: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            temporary_path = f"{save_path}.tmp"
            model_obj.save(temporary_path)
            os.replace(temporary_path, save_path)

    print(f"[OK] Training completed for {model_key}. Best loss: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for 5 Forensic Reconstruction Models")
    parser.add_argument("--model", type=str, default="all", choices=["all", "pix2pix", "unet", "resnet_gan", "pca_ridge", "knn_patch"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoints-dir")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints_dir = args.checkpoints_dir or config["paths"]["checkpoints_dir"]
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = os.path.join(PROJECT_ROOT, checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)

    processed_dir = config["dataset"]["processed_dir"]
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(PROJECT_ROOT, processed_dir)
    train_dataset = SketchPhotoDataset(
        os.path.join(processed_dir, "train"), config["dataset"]["img_size"]
    )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    print(f"[*] Training on {device}; {len(train_dataset)} paired training identities.")

    if args.smoke_test:
        sample_sketch, sample_photo = train_dataset[0]
        sample_sketch = sample_sketch.unsqueeze(0).to(device)
        expected_shape = (1, 3, config["dataset"]["img_size"], config["dataset"]["img_size"])
        smoke_specs = [
            ("pix2pix", "model_1_pix2pix_best.pth", lambda: Pix2PixModel(device=device)),
            ("unet", "model_2_unet_best.pth", lambda: PerceptualUNetModel(device=device)),
            ("resnet_gan", "model_3_resnet_gan_best.pth", lambda: ResNetLSGANModel(device=device)),
            ("pca_ridge", "model_4_pca_ridge_best.pkl", PCARidgeModel),
            ("knn_patch", "model_5_knn_patch_best.pkl", PatchKNNModel),
        ]
        selected_specs = [
            spec for spec in smoke_specs if args.model in ("all", spec[0])
        ]
        for model_name, checkpoint_name, create_model in selected_specs:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
            model = create_model()
            if os.path.isfile(checkpoint_path):
                model.load(checkpoint_path)
            elif model_name in {"pca_ridge", "knn_patch"}:
                print(f"[!] Skipping {model_name}: checkpoint not found.")
                continue
            else:
                print(f"[!] Testing {model_name} architecture with untrained weights; checkpoint missing.")
            with torch.inference_mode():
                generated = model.generate(sample_sketch)
            if tuple(generated.shape) != expected_shape or not torch.isfinite(generated).all():
                raise ValueError(f"{model_name} returned invalid output shape or non-finite values.")
            print(f"[OK] {model_name}: output shape {tuple(generated.shape)}.")
        return
    
    # ------------------
    # 1. Pix2Pix cGAN
    # ------------------
    if args.model in ["all", "pix2pix"]:
        settings = config["models"]["model_1_pix2pix"]
        loader = DataLoader(train_dataset, batch_size=args.batch_size or settings["batch_size"], shuffle=True)
        m1 = Pix2PixModel(device=device, lr=args.learning_rate or settings["lr"])
        train_dl_model("model_1_pix2pix", m1, loader, args.epochs or settings["epochs"], checkpoints_dir, device)

    # ------------------
    # 2. Deep U-Net
    # ------------------
    if args.model in ["all", "unet"]:
        settings = config["models"]["model_2_unet"]
        loader = DataLoader(train_dataset, batch_size=args.batch_size or settings["batch_size"], shuffle=True)
        m2 = PerceptualUNetModel(device=device, lr=args.learning_rate or settings["lr"])
        train_dl_model("model_2_unet", m2, loader, args.epochs or settings["epochs"], checkpoints_dir, device)

    # ------------------
    # 3. ResNet LSGAN
    # ------------------
    if args.model in ["all", "resnet_gan"]:
        settings = config["models"]["model_3_resnet_gan"]
        loader = DataLoader(train_dataset, batch_size=args.batch_size or settings["batch_size"], shuffle=True)
        m3 = ResNetLSGANModel(device=device, lr=args.learning_rate or settings["lr"])
        train_dl_model("model_3_resnet_gan", m3, loader, args.epochs or settings["epochs"], checkpoints_dir, device)

    # ------------------
    # 4. Eigenface PCA-Ridge
    # ------------------
    if args.model in ["all", "pca_ridge"]:
        print(f"\n==========================================")
        print(f" TRAINING MODEL: MODEL_4_PCA_RIDGE")
        print(f"==========================================")
        s_np, p_np = get_numpy_data(os.path.join(processed_dir, "train"))
        m4 = PCARidgeModel(n_components=config["models"]["model_4_pca_ridge"]["n_components"], alpha=config["models"]["model_4_pca_ridge"]["alpha"])
        m4.fit(s_np, p_np)
        checkpoint_path = os.path.join(checkpoints_dir, "model_4_pca_ridge_best.pkl")
        if os.path.isfile(checkpoint_path):
            shutil.copy2(checkpoint_path, f"{checkpoint_path}.{time.strftime('%Y%m%d%H%M%S')}.bak")
        temporary_path = f"{checkpoint_path}.tmp"
        m4.save(temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(f"[OK] Training completed for Model 4 PCA-Ridge.")

    # ------------------
    # 5. Patch K-NN Regressor
    # ------------------
    if args.model in ["all", "knn_patch"]:
        print(f"\n==========================================")
        print(f" TRAINING MODEL: MODEL_5_KNN_PATCH")
        print(f"==========================================")
        s_np, p_np = get_numpy_data(os.path.join(processed_dir, "train"))
        m5 = PatchKNNModel(
            patch_size=config["models"]["model_5_knn_patch"]["patch_size"],
            stride=config["models"]["model_5_knn_patch"]["stride"],
            k_neighbors=config["models"]["model_5_knn_patch"]["k_neighbors"]
        )
        m5.fit(s_np, p_np)
        checkpoint_path = os.path.join(checkpoints_dir, "model_5_knn_patch_best.pkl")
        if os.path.isfile(checkpoint_path):
            shutil.copy2(checkpoint_path, f"{checkpoint_path}.{time.strftime('%Y%m%d%H%M%S')}.bak")
        temporary_path = f"{checkpoint_path}.tmp"
        m5.save(temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(f"[OK] Training completed for Model 5 Patch K-NN.")

if __name__ == "__main__":
    main()
