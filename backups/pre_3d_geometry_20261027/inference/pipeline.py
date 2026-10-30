
import os
import sys
import time
import base64
import io
import gc

import yaml
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from models.model_1_pix2pix import Pix2PixModel
from models.model_2_unet import PerceptualUNetModel
from models.model_3_resnet_gan import ResNetLSGANModel
from models.model_4_pca_ridge import PCARidgeModel
from models.model_5_knn_patch import PatchKNNModel

from model_selection.meta_selector import (
    AdaptiveModelSelector,
    MODEL_KEYS,
    MODEL_DISPLAY_NAMES,
)

from face_3d.mesh_generator import Face3DMeshGenerator
from face_3d.exporter import Face3DExporter

# GFPGAN is optional.
try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None


def pil_to_base64(pil_img):
    """Convert a PIL image to a base64-encoded PNG."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def tensor_to_pil(tensor):
    """Convert a model output tensor in [-1, 1] to an RGB image."""
    arr = tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    arr = np.clip(
        (arr + 1.0) / 2.0 * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    return Image.fromarray(arr)


class ForensicInferencePipeline:
    """
    Sketch-to-image-to-3D inference pipeline.

    GFPGAN enhancement is optional. The pipeline falls back to the
    selected model's generated image when GFPGAN is unavailable.
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoints_dir = self.config["paths"]["checkpoints_dir"]

        print(
            f"[*] Initializing Forensic Pipeline on device: "
            f"'{self.device}'..."
        )

        # Load candidate models.
        self.models = {}

        m1 = Pix2PixModel(device=self.device)
        p1 = os.path.join(
            checkpoints_dir, "model_1_pix2pix_best.pth"
        )
        if os.path.exists(p1):
            m1.load(p1)
        self.models["model_1_pix2pix"] = m1

        m2 = PerceptualUNetModel(device=self.device)
        p2 = os.path.join(
            checkpoints_dir, "model_2_unet_best.pth"
        )
        if os.path.exists(p2):
            m2.load(p2)
        self.models["model_2_unet"] = m2

        m3 = ResNetLSGANModel(device=self.device)
        p3 = os.path.join(
            checkpoints_dir, "model_3_resnet_gan_best.pth"
        )
        if os.path.exists(p3):
            m3.load(p3)
        self.models["model_3_resnet_gan"] = m3

        m4 = PCARidgeModel()
        p4 = os.path.join(
            checkpoints_dir, "model_4_pca_ridge_best.pkl"
        )
        if os.path.exists(p4):
            m4.load(p4)
        self.models["model_4_pca_ridge"] = m4

        m5 = PatchKNNModel()
        p5 = os.path.join(
            checkpoints_dir, "model_5_knn_patch_best.pkl"
        )
        if os.path.exists(p5):
            try:
                m5.load(p5)
            except Exception as e:
                print(
                    f"[!] Optional Model 5 loading skipped: {e}"
                )
        self.models["model_5_knn_patch"] = m5

        # Load the adaptive selector if its checkpoint exists.
        self.selector = AdaptiveModelSelector()
        selector_path = os.path.join(
            checkpoints_dir, "adaptive_selector_best.pkl"
        )
        if os.path.exists(selector_path):
            self.selector.load(selector_path)

        # Initialize the 3D engine.
        self.mesh_generator = Face3DMeshGenerator(
            img_size=(256, 256)
        )
        self.exporter = Face3DExporter(
            output_dir=os.path.join(
                self.config["paths"]["output_dir"], "3d"
            )
        )

        self.gfpgan_path = (
            "https://github.com/TencentARC/GFPGAN/releases/"
            "download/v1.3.0/GFPGANv1.3.pth"
        )

    def process_sketch(
        self, sketch_input, output_basename="reconstructed_face"
    ):
        """Generate a 2D face image and export a 3D mesh."""
        start_time = time.time()

        # Read the input sketch.
        if isinstance(sketch_input, str):
            sketch_pil = Image.open(sketch_input).convert("RGB")
        elif isinstance(sketch_input, Image.Image):
            sketch_pil = sketch_input.convert("RGB")
        else:
            sketch_pil = Image.fromarray(sketch_input).convert("RGB")

        sketch_resized = sketch_pil.resize(
            (256, 256), Image.Resampling.BILINEAR
        )
        sketch_np = np.array(sketch_resized, dtype=np.float32)

        sketch_arr = (sketch_np / 127.5) - 1.0
        sketch_tensor = torch.from_numpy(
            sketch_arr.transpose(2, 0, 1)
        ).unsqueeze(0).float()

        # Generate candidate outputs.
        candidate_previews = {}
        generated_pil_dict = {}

        for key in MODEL_KEYS:
            if key not in self.models:
                raise KeyError(
                    f"Model key '{key}' is missing from self.models."
                )

            gen_tensor = self.models[key].generate(sketch_tensor)
            gen_pil = tensor_to_pil(gen_tensor)

            candidate_previews[key] = pil_to_base64(gen_pil)
            generated_pil_dict[key] = gen_pil

        # Select the best candidate.
        selection_result = self.selector.predict_suitability(
            sketch_resized,
            generated_pil_dict=generated_pil_dict,
        )

        selected_key = selection_result["selected_model_key"]
        selected_name = selection_result["selected_model_name"]
        raw_best_photo = generated_pil_dict[selected_key]

        del sketch_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Optional GFPGAN enhancement. Keep the original generated
        # image if GFPGAN is not installed or enhancement fails.
        best_2d_photo = raw_best_photo

        if GFPGANer is not None:
            restorer = None
            try:
                restorer = GFPGANer(
                    model_path=self.gfpgan_path,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device,
                )

                raw_best_np = np.array(
                    raw_best_photo.convert("RGB")
                )
                raw_best_bgr = cv2.cvtColor(
                    raw_best_np, cv2.COLOR_RGB2BGR
                )

                _, _, enhanced_bgr = restorer.enhance(
                    raw_best_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )

                if enhanced_bgr is not None:
                    enhanced_rgb = cv2.cvtColor(
                        enhanced_bgr, cv2.COLOR_BGR2RGB
                    )
                    best_2d_photo = Image.fromarray(enhanced_rgb)

            except Exception as e:
                print(
                    f"[!] GFPGAN enhancement skipped: {e}"
                )

            finally:
                if restorer is not None:
                    del restorer
        else:
            print(
                "[!] GFPGAN is not installed; "
                "using the generated image without enhancement."
            )

        best_2d_photo = best_2d_photo.convert("RGB")
        best_2d_np = np.array(best_2d_photo)
        candidate_previews[selected_key] = pil_to_base64(
            best_2d_photo
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Save the generated 2D image.
        output_2d_dir = os.path.join(
            self.config["paths"]["output_dir"], "2d"
        )
        os.makedirs(output_2d_dir, exist_ok=True)

        photo_2d_path = os.path.join(
            output_2d_dir, f"{output_basename}_2d.png"
        )
        best_2d_photo.save(photo_2d_path)

        # Create and export the 3D mesh.
        vertices, faces, uvs, landmarks = (
            self.mesh_generator.generate_mesh(best_2d_np)
        )
        paths_3d = self.exporter.export_all(
            vertices,
            faces,
            uvs,
            best_2d_photo,
            base_filename=output_basename,
        )

        elapsed = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "execution_time_sec": elapsed,
            "sketch_b64": pil_to_base64(sketch_resized),
            "generated_2d_b64": pil_to_base64(best_2d_photo),
            "selected_model_key": selected_key,
            "selected_model_name": selected_name,
            "confidence_percent": selection_result[
                "confidence_percent"
            ],
            "model_scores": selection_result["scores"],
            "model_probabilities": selection_result[
                "probabilities"
            ],
            "candidate_previews": candidate_previews,
            "paths": {
                "2d_photo": os.path.abspath(photo_2d_path),
                "3d_obj": paths_3d["obj"],
                "3d_ply": paths_3d["ply"],
                "3d_glb": paths_3d["glb"],
            },
        }
