import os
import sys
import pytest
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_1_pix2pix import Pix2PixGenerator
from models.model_2_unet import DeepUNetGenerator
from models.model_3_resnet_gan import ResNetGenerator
from models.model_4_pca_ridge import PCARidgeModel
from models.model_5_knn_patch import PatchKNNModel
from model_selection.feature_extractor import SketchFeatureExtractor
from model_selection.meta_selector import AdaptiveModelSelector
from face_3d.mesh_generator import Face3DMeshGenerator
from face_3d.exporter import Face3DExporter

def test_dataset_structure():
    processed_dir = "data/processed"
    assert os.path.exists(os.path.join(processed_dir, "train", "sketches"))
    assert os.path.exists(os.path.join(processed_dir, "train", "photos"))
    assert os.path.exists(os.path.join(processed_dir, "PREPROCESSING_REPORT.txt"))

def test_deep_learning_forward_passes():
    x = torch.randn(1, 3, 256, 256)
    
    # Model 1
    g1 = Pix2PixGenerator()
    out1 = g1(x)
    assert out1.shape == (1, 3, 256, 256)
    del g1, out1

    # Model 2
    g2 = DeepUNetGenerator()
    out2 = g2(x)
    assert out2.shape == (1, 3, 256, 256)
    del g2, out2

    # Model 3
    g3 = ResNetGenerator()
    out3 = g3(x)
    assert out3.shape == (1, 3, 256, 256)
    del g3, out3

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_sketch_feature_extractor():
    extractor = SketchFeatureExtractor()
    dummy_img = Image.new("RGB", (256, 256), (200, 200, 200))
    feats = extractor.extract_features(dummy_img)
    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 1
    assert len(feats) > 10

def test_adaptive_model_selector():
    selector = AdaptiveModelSelector()
    dummy_img = Image.new("RGB", (256, 256), (200, 200, 200))
    res = selector.predict_suitability(dummy_img)
    
    assert "selected_model_key" in res
    assert "scores" in res
    assert "probabilities" in res
    
    probs_sum = sum(res["probabilities"].values())
    assert abs(probs_sum - 1.0) < 1e-4

def test_3d_face_mesh_generation_and_export():
    generator = Face3DMeshGenerator(img_size=(256, 256), grid_res=32)
    dummy_face = np.ones((256, 256, 3), dtype=np.uint8) * 180
    
    vertices, faces, uvs, landmarks = generator.generate_mesh(dummy_face)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert uvs.shape[1] == 2

    exporter = Face3DExporter(output_dir="output/test_3d")
    paths = exporter.export_all(vertices, faces, uvs, dummy_face, base_filename="test_mesh")
    
    assert os.path.exists(paths["obj"])
    assert os.path.exists(paths["ply"])
    assert os.path.exists(paths["glb"])
