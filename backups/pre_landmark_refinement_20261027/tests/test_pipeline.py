import os
import sys
import pytest
import numpy as np
import trimesh
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dataset_structure():
    processed_dir = "data/processed"
    assert os.path.exists(os.path.join(processed_dir, "train", "sketches"))
    assert os.path.exists(os.path.join(processed_dir, "train", "photos"))
    assert os.path.exists(os.path.join(processed_dir, "PREPROCESSING_REPORT.txt"))

def test_deep_learning_forward_passes():
    import torch

    from models.model_1_pix2pix import Pix2PixGenerator
    from models.model_2_unet import DeepUNetGenerator
    from models.model_3_resnet_gan import ResNetGenerator

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
    from model_selection.feature_extractor import SketchFeatureExtractor

    extractor = SketchFeatureExtractor()
    dummy_img = Image.new("RGB", (256, 256), (200, 200, 200))
    feats = extractor.extract_features(dummy_img)
    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 1
    assert len(feats) > 10

def test_adaptive_model_selector():
    from model_selection.meta_selector import AdaptiveModelSelector

    selector = AdaptiveModelSelector()
    dummy_img = Image.new("RGB", (256, 256), (200, 200, 200))
    res = selector.predict_suitability(dummy_img)
    
    assert "selected_model_key" in res
    assert "scores" in res
    assert "probabilities" in res
    
    probs_sum = sum(res["probabilities"].values())
    assert abs(probs_sum - 1.0) < 1e-4

def test_3d_face_mesh_generation_and_export(tmp_path):
    from face_3d.mesh_generator import Face3DMeshGenerator
    from face_3d.exporter import Face3DExporter

    generator = Face3DMeshGenerator(img_size=(256, 256), grid_res=32)
    sketch_dir = os.path.join("data", "processed", "test", "sketches")
    sketches = [
        np.array(Image.open(os.path.join(sketch_dir, filename)).convert("RGB"))
        for filename in ("id_0004.png", "id_0007.png")
    ]

    vertices, faces, uvs, landmarks = generator.generate_mesh(sketches[0])
    other_vertices, _, _, _ = generator.generate_mesh(sketches[1])
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert uvs.shape[1] == 2
    assert len(vertices) == len(uvs)
    assert faces.min() >= 0 and faces.max() < len(vertices)
    assert np.ptp(vertices[:, 2]) > 0.5
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert np.mean(np.abs(vertices - other_vertices)) > 0.001
    assert landmarks["landmark_source"] in {
        "opencv_haar_face", "opencv_haar_face_and_eyes", "centered_face_fallback",
        "centered_face_fallback_and_eyes",
    }

    exporter = Face3DExporter(output_dir=str(tmp_path))
    paths = exporter.export_all(vertices, faces, uvs, sketches[0], base_filename="test_mesh")
    
    assert os.path.exists(paths["obj"])
    assert os.path.exists(paths["ply"])
    assert os.path.exists(paths["glb"])
    ply_mesh = trimesh.load(paths["ply"], force="mesh")
    obj_scene = trimesh.load(paths["obj"], force="scene")
    assert len(ply_mesh.faces) > 0
    assert len(ply_mesh.visual.vertex_colors) == len(ply_mesh.vertices)
    assert len(obj_scene.geometry) > 0
    assert os.path.exists(os.path.join(str(tmp_path), "material.mtl"))
    assert os.path.exists(os.path.join(str(tmp_path), "material_0.png"))
    glb_scene = trimesh.load(paths["glb"], force="scene")
    assert len(glb_scene.geometry) > 0
    assert sum(len(mesh.faces) for mesh in glb_scene.geometry.values()) > 0
    assert any(
        getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None) is not None
        for mesh in glb_scene.geometry.values()
    )
