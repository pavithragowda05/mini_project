
import os
import joblib
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class PatchKNNModel:
    """Patch-based K-NN model."""

    def __init__(self, patch_size=32, stride=32, k_neighbors=3,
                 img_shape=(256, 256, 3)):
        self.patch_size = patch_size
        self.stride = stride
        self.k_neighbors = k_neighbors
        self.img_shape = img_shape
        self.nn_model = NearestNeighbors(
            n_neighbors=k_neighbors, algorithm="kd_tree"
        )
        self.sketch_patches = None
        self.photo_patches = None
        self.is_fitted = False

    def _extract_patches(self, images):
        N, H, W, C = images.shape
        patches = []
        for i in range(N):
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    patches.append(
                        images[i, y:y+self.patch_size,
                               x:x+self.patch_size, :]
                    )
        return np.array(patches)

    def fit(self, sketch_images, photo_images):
        s_patches = self._extract_patches(sketch_images)
        p_patches = self._extract_patches(photo_images)

        if len(s_patches) == 0 or len(p_patches) == 0:
            raise ValueError("No patches found for K-NN training.")

        if len(s_patches) != len(p_patches):
            raise ValueError("Sketch and photo patch counts differ.")

        self.nn_model.fit(s_patches.reshape(len(s_patches), -1))
        self.sketch_patches = s_patches
        self.photo_patches = p_patches
        self.is_fitted = True
        return self

    def predict_single(self, sketch_img):
        # Safe fallback if the saved K-NN model was not fitted.
        if (
            not self.is_fitted
            or self.sketch_patches is None
            or self.photo_patches is None
        ):
            return sketch_img.copy()

        H, W, C = sketch_img.shape
        canvas = np.zeros((H, W, C), dtype=np.float32)
        weight_map = np.zeros((H, W, C), dtype=np.float32)

        query_patches = []
        coords = []

        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                patch = sketch_img[
                    y:y+self.patch_size, x:x+self.patch_size, :
                ]
                query_patches.append(patch.reshape(-1))
                coords.append((y, x))

        if not query_patches:
            return sketch_img.copy()

        query_feats = np.array(query_patches)
        distances, indices = self.nn_model.kneighbors(query_feats)

        for idx, (y, x) in enumerate(coords):
            idx_list = indices[idx]
            dist_list = distances[idx]

            weights = 1.0 / (dist_list + 1e-5)
            weights /= np.sum(weights)

            matched = self.photo_patches[idx_list]
            blended = np.tensordot(weights, matched, axes=(0, 0))

            canvas[y:y+self.patch_size, x:x+self.patch_size, :] += blended
            weight_map[y:y+self.patch_size, x:x+self.patch_size, :] += 1.0

        weight_map = np.maximum(weight_map, 1.0)
        return np.clip(canvas / weight_map, -1.0, 1.0)

    def generate(self, sketch_tensor):
        sketch_np = (
            sketch_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
        )
        preds = [self.predict_single(img) for img in sketch_np]
        preds_np = np.array(preds).transpose(0, 3, 1, 2)
        return torch.from_numpy(preds_np).float().to(sketch_tensor.device)

    def save(self, filepath):
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        joblib.dump({
            "nn_model": self.nn_model,
            "sketch_patches": self.sketch_patches,
            "photo_patches": self.photo_patches,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "k_neighbors": self.k_neighbors,
            "img_shape": self.img_shape,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.nn_model = data["nn_model"]
        self.sketch_patches = data["sketch_patches"]
        self.photo_patches = data["photo_patches"]
        self.patch_size = data["patch_size"]
        self.stride = data["stride"]
        self.k_neighbors = data["k_neighbors"]
        self.img_shape = data["img_shape"]
        self.is_fitted = data["is_fitted"]