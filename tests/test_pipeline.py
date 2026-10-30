import os
import sys
import pytest
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SKETCH = PROJECT_ROOT / "frontend" / "sample_sketch.png"

def test_dataset_structure():
    processed_dir = "data/processed"
    if not os.path.exists(os.path.join(processed_dir, "train", "sketches")):
        pytest.skip("Processed paired dataset is not included in this workspace.")
    assert os.path.exists(os.path.join(processed_dir, "train", "sketches"))
    assert os.path.exists(os.path.join(processed_dir, "train", "photos"))
    assert os.path.exists(os.path.join(processed_dir, "PREPROCESSING_REPORT.txt"))

def test_paired_dataset_identity_matching_and_normalization(tmp_path):
    import torch
    from training.train_models import SketchPhotoDataset

    sketch_dir = tmp_path / "train" / "sketches"
    photo_dir = tmp_path / "train" / "photos"
    sketch_dir.mkdir(parents=True)
    photo_dir.mkdir(parents=True)
    Image.new("RGB", (8, 12), (255, 0, 0)).save(sketch_dir / "id_2.png")
    Image.new("RGB", (16, 8), (0, 0, 255)).save(photo_dir / "id_2.png")
    Image.new("RGB", (10, 10), (0, 255, 0)).save(sketch_dir / "id_1.png")
    Image.new("RGB", (12, 16), (255, 255, 0)).save(photo_dir / "id_1.png")

    dataset = SketchPhotoDataset(str(tmp_path / "train"), img_size=16)
    sketch, photo = dataset[0]
    assert len(dataset) == 2
    assert sketch.shape == photo.shape == (3, 16, 16)
    assert torch.allclose(sketch[:, 0, 0], torch.tensor([-1.0, 1.0, -1.0]))
    assert torch.allclose(photo[:, 0, 0], torch.tensor([1.0, 1.0, -1.0]))

    (photo_dir / "id_2.png").unlink()
    with pytest.raises(ValueError, match="Unpaired dataset identities"):
        SketchPhotoDataset(str(tmp_path / "train"), img_size=16)

def test_upload_preprocesses_without_claiming_reconstruction(monkeypatch):
    import asyncio
    import base64
    import json
    from io import BytesIO
    from fastapi import HTTPException
    from starlette.datastructures import UploadFile
    import backend.app as backend_app

    monkeypatch.setattr(backend_app, "pipeline", None)
    monkeypatch.setattr(backend_app, "inference_error", "No usable trained model checkpoints found.")
    sample = SAMPLE_SKETCH.read_bytes()
    upload = UploadFile(filename="sample_sketch.png", file=BytesIO(sample))
    response = asyncio.run(backend_app.process_sketch(upload))

    assert response.status_code == 200
    result = json.loads(response.body)
    assert result["status"] == "preprocessing_only"
    assert result["reconstruction_available"] is False
    assert result["preprocessing"]["is_reconstruction"] is False
    preview = Image.open(BytesIO(base64.b64decode(result["preprocessed_input_b64"])))
    assert preview.size == (256, 256)
    assert preview.mode == "RGB"

    invalid = UploadFile(filename="invalid.png", file=BytesIO(b"not an image"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(backend_app.process_sketch(invalid))
    assert getattr(error.value, "status_code", None) == 400

    unsupported = UploadFile(filename="sample.gif", file=BytesIO(sample))
    with pytest.raises(HTTPException) as error:
        asyncio.run(backend_app.process_sketch(unsupported))
    assert error.value.status_code == 400

def test_health_reports_expected_checkpoint_status(monkeypatch, tmp_path):
    import backend.app as backend_app

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "paths:\n  checkpoints_dir: checkpoints\n", encoding="utf-8"
    )
    monkeypatch.setattr(backend_app, "project_dir", str(tmp_path))
    monkeypatch.setattr(backend_app, "pipeline", None)
    monkeypatch.setattr(backend_app, "inference_error", "weights unavailable")
    health = backend_app.health()

    assert health["status"] == "ok"
    assert health["pipeline_ready"] is False
    assert health["reconstruction_ready"] is False
    assert health["inference_error"] == "weights unavailable"
    assert set(health["checkpoints"]) == {
        "model_1_pix2pix", "model_2_unet", "model_3_resnet_gan",
        "model_4_pca_ridge", "model_5_knn_patch",
    }
    assert all(status["available"] is False for status in health["checkpoints"].values())
    assert all(status["loaded"] is False for status in health["checkpoints"].values())

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
    with pytest.raises(RuntimeError, match="Selector is not trained"):
        selector.predict_suitability(dummy_img)

    single_candidate = selector.predict_suitability(
        dummy_img,
        generated_pil_dict={"model_4_pca_ridge": dummy_img},
    )
    assert single_candidate["selected_model_key"] == "model_4_pca_ridge"
    assert list(single_candidate["scores"]) == ["model_4_pca_ridge"]
    assert single_candidate["probabilities"] == {}
    assert single_candidate["confidence_percent"] is None
    assert single_candidate["score_semantics"] == "heuristic_composite_rank_not_probability"

def test_3d_face_mesh_generation_and_export(tmp_path):
    from face_3d.mesh_generator import Face3DMeshGenerator
    from face_3d.exporter import Face3DExporter
    from PIL import ImageDraw

    generator = Face3DMeshGenerator(img_size=(256, 256), grid_res=32)
    sample = np.array(Image.open(SAMPLE_SKETCH).convert("RGB"))
    sketches = [
        sample,
        np.fliplr(sample).copy(),
    ]

    vertices, faces, uvs, landmarks = generator.generate_mesh(sketches[0])
    other_vertices, _, _, _ = generator.generate_mesh(sketches[1])
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert uvs.shape[1] == 2
    assert len(vertices) == len(uvs)
    assert faces.min() >= 0 and faces.max() < len(vertices)
    assert vertices[-1, 1] > 0.99 and uvs[-1, 1] == 1.0
    assert vertices[-2, 1] < -0.99 and uvs[-2, 1] == 0.0
    assert landmarks["left_eye"][1] < landmarks["nose_tip"][1]
    assert landmarks["nose_tip"][1] < landmarks["mouth_center"][1] < landmarks["chin"][1]
    assert np.ptp(vertices[:, 2]) > 0.5
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert np.mean(np.abs(vertices - other_vertices)) > 0.0001
    assert landmarks["landmark_source"] in {
        "opencv_haar_face", "opencv_haar_face_and_eyes", "centered_face_fallback",
        "centered_face_fallback_and_eyes", "centered_face_fallback_contrast_eyes",
        "centered_face_fallback_contrast_eyes_contrast_mouth",
    }

    landmark_image = Image.new("RGB", (256, 256), (245, 245, 245))
    draw = ImageDraw.Draw(landmark_image)
    draw.ellipse((48, 18, 208, 238), fill=(205, 180, 160))
    draw.ellipse((69, 84, 104, 101), fill=(35, 30, 28))
    draw.ellipse((152, 84, 187, 101), fill=(35, 30, 28))
    draw.ellipse((98, 169, 158, 186), fill=(90, 35, 40))
    detected = generator.extract_landmarks(np.asarray(landmark_image))
    assert detected["eye_confidence"] >= 0.8
    assert detected["mouth_confidence"] >= 0.8
    assert abs(detected["left_eye"][0] - 86.5) < 10
    assert abs(detected["right_eye"][0] - 169.5) < 10
    assert abs(detected["mouth_center"][1] - 177.5) < 10

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


def test_3d_download_routes_with_temporary_exports(tmp_path, monkeypatch):
    import zipfile
    from fastapi import HTTPException

    import backend.app as backend_app
    from face_3d.exporter import Face3DExporter
    from face_3d.mesh_generator import Face3DMeshGenerator

    image = np.array(Image.open(SAMPLE_SKETCH).convert("RGB"))
    output_dir = tmp_path / "output"
    output_3d_dir = output_dir / "3d"
    monkeypatch.setattr(backend_app, "output_dir", str(output_dir))
    monkeypatch.setattr(backend_app, "reconstruction_ready", False)
    for file_type in ("obj", "ply", "glb"):
        with pytest.raises(HTTPException) as error:
            backend_app.download_3d(file_type)
        assert error.value.status_code == 404
    with pytest.raises(HTTPException) as error:
        backend_app.download_3d_obj_bundle()
    assert error.value.status_code == 404

    vertices, faces, uvs, _ = Face3DMeshGenerator(grid_res=24).generate_mesh(image)
    exporter = Face3DExporter(output_dir=str(output_3d_dir))
    paths = exporter.export_all(
        vertices, faces, uvs, image, base_filename="sketch_result"
    )
    monkeypatch.setattr(backend_app, "reconstruction_ready", True)

    for file_type in ("obj", "ply", "glb"):
        response = backend_app.download_3d(file_type)
        assert os.path.abspath(response.path) == paths[file_type]
        assert os.path.getsize(response.path) > 0

    bundle = backend_app.download_3d_obj_bundle()
    try:
        with zipfile.ZipFile(bundle.path) as archive:
            assert sorted(archive.namelist()) == [
                "material.mtl", "material_0.png", "sketch_result.obj"
            ]
    finally:
        os.remove(bundle.path)
