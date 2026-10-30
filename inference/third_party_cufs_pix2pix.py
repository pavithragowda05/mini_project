"""Optional third-party Pix2Pix model trained for CUFS face sketch-to-photo."""

import hashlib
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch
from PIL import Image


MODEL_FILENAME = "third_party_cufs_pix2pix.keras"
MODEL_SHA256 = "76befbcc6f0dc5e5f63451c3ce237921f014306c6f81e618631a2925fa697cad"
SOURCE_URL = "https://github.com/goldteaa/pix2pix-face-sketch-to-photo"
SOURCE_LICENSE = "MIT"
TRAINING_DATA = "CUFS (188 paired face sketch/photo samples, as documented upstream)"


class CufsPix2PixBackend:
    """CPU-capable optional photo generator; does not replace project models."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Optional CUFS Pix2Pix weights not found: {self.model_path}")

        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError(
                "Optional CUFS Pix2Pix weight hash mismatch; refusing to load the file."
            )

        self.model = keras.saving.load_model(
            self.model_path, compile=False, safe_mode=True
        )
        if self.model.input_shape != (None, 256, 256, 3):
            raise ValueError(f"Unexpected model input shape: {self.model.input_shape}")
        if self.model.output_shape != (None, 256, 256, 3):
            raise ValueError(f"Unexpected model output shape: {self.model.output_shape}")
        self.model.trainable = False

    def generate(self, sketch_image):
        """Return the model-generated RGB photo; input/output use Pix2Pix [-1, 1]."""
        if not isinstance(sketch_image, Image.Image):
            sketch_image = Image.fromarray(np.asarray(sketch_image))
        rgb = sketch_image.convert("RGB").resize(
            (256, 256), Image.Resampling.BILINEAR
        )
        image_array = np.asarray(rgb, dtype=np.float32)
        normalized = image_array / 127.5 - 1.0
        input_tensor = torch.from_numpy(normalized[None, ...]).to(dtype=torch.float32)

        with torch.inference_mode():
            output = self.model(input_tensor, training=False)
        if not torch.is_tensor(output):
            output = torch.as_tensor(output)
        if tuple(output.shape) != (1, 256, 256, 3):
            raise ValueError(f"Optional model returned unexpected shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise ValueError("Optional model returned non-finite pixel values.")

        output_array = output[0].detach().cpu().numpy()
        output_rgb = np.rint((np.clip(output_array, -1.0, 1.0) + 1.0) * 127.5)
        return Image.fromarray(output_rgb.astype(np.uint8), mode="RGB")