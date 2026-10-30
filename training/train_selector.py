import os
import sys
import glob
import argparse
import shutil
import time
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from models.model_1_pix2pix import Pix2PixModel
from models.model_2_unet import PerceptualUNetModel
from models.model_3_resnet_gan import ResNetLSGANModel
from models.model_4_pca_ridge import PCARidgeModel
from models.model_5_knn_patch import PatchKNNModel
from model_selection.meta_selector import AdaptiveModelSelector, MODEL_KEYS
from training.train_models import SketchPhotoDataset

def calculate_quality_score(gen_tensor, gt_tensor):
    """
    Computes metric quality score (combined SSIM + normalized PSNR) 
    between generated face and ground truth photo.
    """
    gen_np = ((gen_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    gt_np = ((gt_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    score_ssim = ssim(gen_np, gt_np, channel_axis=2, data_range=255)
    score_psnr = psnr(gt_np, gen_np, data_range=255)
    
    # Combined score
    composite_score = 0.7 * score_ssim + 0.3 * (score_psnr / 40.0)
    return float(composite_score)

def train_adaptive_selector(device="auto", smoke_test=False, checkpoints_override=None):
    with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints_dir = checkpoints_override or config["paths"]["checkpoints_dir"]
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = os.path.join(PROJECT_ROOT, checkpoints_dir)
    processed_dir = config["dataset"]["processed_dir"]
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(PROJECT_ROOT, processed_dir)
    val_dir = os.path.join(processed_dir, "val")
    val_dataset = SketchPhotoDataset(val_dir, config["dataset"]["img_size"])
    if not smoke_test and len(val_dataset) < 2:
        raise ValueError("Selector training needs at least two held-out paired validation samples.")
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("\n==========================================")
    print(" TRAINING ADAPTIVE META-MODEL SELECTOR")
    print("==========================================")
    print(f"[*] Device: {device}; validation pairs: {len(val_dataset)}")

    # 1. Load Trained Candidate Models
    print("[*] Loading trained candidate models from checkpoints...")
    models = {}
    
    model_specs = [
        ("model_1_pix2pix", "model_1_pix2pix_best.pth", lambda: Pix2PixModel(device=device)),
        ("model_2_unet", "model_2_unet_best.pth", lambda: PerceptualUNetModel(device=device)),
        ("model_3_resnet_gan", "model_3_resnet_gan_best.pth", lambda: ResNetLSGANModel(device=device)),
        ("model_4_pca_ridge", "model_4_pca_ridge_best.pkl", PCARidgeModel),
        ("model_5_knn_patch", "model_5_knn_patch_best.pkl", PatchKNNModel),
    ]
    missing = [name for _, name, _ in model_specs if not os.path.isfile(os.path.join(checkpoints_dir, name))]
    if missing:
        raise FileNotFoundError(
            "Selector training requires all five candidate checkpoints; missing: "
            + ", ".join(missing)
        )
    for key, filename, create_model in model_specs:
        model = create_model()
        model.load(os.path.join(checkpoints_dir, filename))
        models[key] = model

    # 2. Extract Features & Evaluate Candidate Quality on Validation Set
    selector = AdaptiveModelSelector()
    X_features = []
    Y_scores = []

    print(f"[*] Evaluating candidate models across {len(val_dataset)} validation sketches...")

    for sketch_tensor, real_photo_tensor in loader:
        # Extract features strictly from sketch
        feat = selector.feature_extractor.extract_features(sketch_tensor)
        X_features.append(feat)

        sample_scores = []
        for key in MODEL_KEYS:
            model_obj = models[key]
            with torch.inference_mode():
                gen_tensor = model_obj.generate(sketch_tensor)
            if gen_tensor.shape != real_photo_tensor.shape or not torch.isfinite(gen_tensor).all():
                raise ValueError(f"{key} returned invalid output shape or non-finite values.")
            score = calculate_quality_score(gen_tensor, real_photo_tensor)
            sample_scores.append(score)
        
        Y_scores.append(sample_scores)
        if smoke_test:
            print("[OK] All candidate checkpoints produced finite outputs with expected shapes.")
            return

    X_features = np.array(X_features)
    Y_scores = np.array(Y_scores)

    # 3. Fit Meta-Selector
    selector.fit(X_features, Y_scores)
    save_path = os.path.join(checkpoints_dir, "adaptive_selector_best.pkl")
    if os.path.isfile(save_path):
        backup_path = f"{save_path}.{time.strftime('%Y%m%d%H%M%S')}.bak"
        shutil.copy2(save_path, backup_path)
        print(f"[!] Existing selector checkpoint preserved at '{backup_path}'.")
    temporary_path = f"{save_path}.tmp"
    selector.save(temporary_path)
    os.replace(temporary_path, save_path)
    print(f"[OK] Adaptive Meta-Selector successfully trained & saved to '{save_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or smoke-test the adaptive selector.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoints-dir")
    parser.add_argument("--smoke-test", action="store_true")
    arguments = parser.parse_args()
    train_adaptive_selector(arguments.device, arguments.smoke_test, arguments.checkpoints_dir)
