import os
import sys
import shutil
import tempfile
import zipfile
import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from PIL import Image, UnidentifiedImageError

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

pipeline = None
inference_error = None
reconstruction_ready = False
third_party_pix2pix_backend = None
EXPECTED_CHECKPOINTS = {
    "model_1_pix2pix": "model_1_pix2pix_best.pth",
    "model_2_unet": "model_2_unet_best.pth",
    "model_3_resnet_gan": "model_3_resnet_gan_best.pth",
    "model_4_pca_ridge": "model_4_pca_ridge_best.pkl",
    "model_5_knn_patch": "model_5_knn_patch_best.pkl",
}


@asynccontextmanager
async def lifespan(app):
    global pipeline, inference_error, reconstruction_ready
    reconstruction_ready = False
    print("[*] Starting FastAPI Server & Loading Forensic Pipeline...")
    try:
        from inference.pipeline import ForensicInferencePipeline

        config_path = os.path.join(project_dir, "configs", "config.yaml")
        pipeline = ForensicInferencePipeline(config_path=config_path)
        inference_error = None
        print("[OK] Forensic Pipeline initialized successfully.")
    except Exception as exc:
        pipeline = None
        inference_error = str(exc)
        print(f"[!] Live inference unavailable: {inference_error}")
    try:
        yield
    finally:
        pipeline = None
        reconstruction_ready = False


app = FastAPI(
    title="Adaptive Multi-Model Forensic Sketch-to-2D and 3D Face Reconstruction System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Frontend Files
frontend_dir = os.path.join(project_dir, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

output_dir = os.path.join(project_dir, "output")
os.makedirs(output_dir, exist_ok=True)
app.mount("/output", StaticFiles(directory=output_dir), name="output")

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Frontend index.html not found.</h2>", status_code=404)

@app.get("/health")
def health():
    config_path = os.path.join(project_dir, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as config_file:
        checkpoints_dir = yaml.safe_load(config_file)["paths"]["checkpoints_dir"]
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = os.path.join(project_dir, checkpoints_dir)
    loaded_keys = set(getattr(pipeline, "available_model_keys", []))
    checkpoint_status = {
        key: {
            "filename": filename,
            "available": os.path.isfile(os.path.join(checkpoints_dir, filename)),
            "loaded": key in loaded_keys,
        }
        for key, filename in EXPECTED_CHECKPOINTS.items()
    }
    optional_model_path = os.path.join(
        project_dir, "models", "third_party_cufs_pix2pix.keras"
    )
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "inference_error": inference_error,
        "device": pipeline.device if pipeline is not None else None,
        "reconstruction_ready": reconstruction_ready,
        "checkpoints": checkpoint_status,
        "optional_models": {
            "third_party_cufs_pix2pix": {
                "available": os.path.isfile(optional_model_path),
                "loaded": third_party_pix2pix_backend is not None,
                "source": "goldteaa/pix2pix-face-sketch-to-photo",
                "license": "MIT",
                "task": "CUFS face sketch to photo, 2D only",
            }
        },
    }


def _encode_png_base64(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/api/download_3d/obj-bundle")
def download_3d_obj_bundle():
    if not reconstruction_ready:
        raise HTTPException(
            status_code=404,
            detail="No reconstruction has been generated in this server session.",
        )
    output_3d_dir = os.path.join(output_dir, "3d")
    sidecars = ["sketch_result.obj", "material.mtl", "material_0.png"]
    missing = [
        filename for filename in sidecars
        if not os.path.isfile(os.path.join(output_3d_dir, filename))
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail="OBJ material package is incomplete. Process a sketch first.",
        )

    archive_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as archive:
            archive_path = archive.name
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for filename in sidecars:
                bundle.write(
                    os.path.join(output_3d_dir, filename), arcname=filename
                )
    except Exception:
        if archive_path and os.path.exists(archive_path):
            os.remove(archive_path)
        raise

    return FileResponse(
        archive_path,
        filename="forensic_3d_face_obj_bundle.zip",
        media_type="application/zip",
        background=BackgroundTask(os.remove, archive_path),
    )

@app.post("/api/process_sketch")
async def process_sketch(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Invalid image file format. Supported: PNG, JPG, JPEG, BMP, WEBP.")

    temp_path = None
    try:
        temp_dir = os.path.join(project_dir, "data", "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="upload_", suffix=suffix,
            dir=temp_dir, delete=False
        ) as buffer:
            temp_path = buffer.name
            shutil.copyfileobj(file.file, buffer)

        try:
            with Image.open(temp_path) as image:
                image.verify()
            with Image.open(temp_path) as image:
                sketch_pil = image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

        if pipeline is None:
            config_path = os.path.join(project_dir, "configs", "config.yaml")
            with open(config_path, "r", encoding="utf-8") as config_file:
                image_size = int(yaml.safe_load(config_file)["dataset"]["img_size"])
            original_rgb = np.asarray(sketch_pil, dtype=np.uint8)
            resized_rgb = cv2.resize(
                original_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR
            )
            normalized = resized_rgb.astype(np.float32) / 127.5 - 1.0
            preview_rgb = np.rint((normalized + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            return JSONResponse(content={
                "status": "preprocessing_only",
                "reconstruction_available": False,
                "message": (
                    inference_error
                    or "Trained checkpoints are required for model reconstruction."
                ),
                "preprocessed_input_b64": _encode_png_base64(preview_rgb),
                "preprocessing": {
                    "method": "OpenCV RGB resize and model-range normalization",
                    "original_size": [sketch_pil.width, sketch_pil.height],
                    "model_input_size": [image_size, image_size],
                    "channels": "RGB",
                    "normalized_range": [-1.0, 1.0],
                    "is_reconstruction": False,
                },
            })

        res = pipeline.process_sketch(sketch_pil, output_basename="sketch_result")
        required_outputs = ("2d_photo", "3d_obj", "3d_ply", "3d_glb")
        paths = res.get("paths", {})
        missing_outputs = [
            key for key in required_outputs
            if not paths.get(key) or not os.path.isfile(paths[key])
        ]
        if missing_outputs:
            raise RuntimeError(
                "Inference did not produce required files: " + ", ".join(missing_outputs)
            )
        global reconstruction_ready
        reconstruction_ready = True
        return JSONResponse(content=res)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Sketch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/process_sketch/third-party-cufs")
async def process_sketch_third_party_cufs(file: UploadFile = File(...)):
    """Use a separate, documented third-party CUFS Pix2Pix model for 2D only."""
    global third_party_pix2pix_backend

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file format. Supported: PNG, JPG, JPEG, BMP, WEBP.",
        )

    weights_path = os.path.join(
        project_dir, "models", "third_party_cufs_pix2pix.keras"
    )
    if not os.path.isfile(weights_path):
        raise HTTPException(
            status_code=503,
            detail="The optional third-party CUFS Pix2Pix weights are not installed.",
        )

    try:
        contents = await file.read()
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
        with Image.open(io.BytesIO(contents)) as image:
            sketch = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        if third_party_pix2pix_backend is None:
            from inference.third_party_cufs_pix2pix import CufsPix2PixBackend

            third_party_pix2pix_backend = CufsPix2PixBackend(weights_path)
        generated = third_party_pix2pix_backend.generate(sketch)
    except Exception as exc:
        print(f"[ERROR] Optional CUFS Pix2Pix inference failed: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Optional CUFS Pix2Pix inference failed: {exc}"
        ) from exc

    model_info = {
        "key": "third_party_cufs_pix2pix",
        "name": "Third-party CUFS Pix2Pix (2D only)",
        "source": "https://github.com/goldteaa/pix2pix-face-sketch-to-photo",
        "license": "MIT",
        "training_data": "CUFS (188 paired examples, per upstream README)",
        "limitations": [
            "Not validated for forensic identification or identity accuracy.",
            "Training data is limited to frontal CUFS sketches/photos.",
            "This alternative is separate from the five missing project models.",
            "This model does not generate 3D geometry.",
        ],
    }
    return JSONResponse(content={
        "status": "third_party_model_output",
        "reconstruction_available": True,
        "model": model_info,
        "original_sketch_b64": pil_to_base64(sketch),
        "generated_2d_b64": pil_to_base64(generated),
        "output_size": list(generated.size),
    })

@app.get("/api/download_3d/{file_type}")
def download_3d(file_type: str):
    if not reconstruction_ready:
        raise HTTPException(
            status_code=404,
            detail="No reconstruction has been generated in this server session.",
        )
    file_type = file_type.lower()
    if file_type not in ["obj", "ply", "glb"]:
        raise HTTPException(status_code=400, detail="Invalid 3D format requested. Use obj, ply, or glb.")

    target_file = os.path.join(output_dir, "3d", f"sketch_result.{file_type}")
    if os.path.exists(target_file):
        return FileResponse(path=target_file, filename=f"forensic_3d_face.{file_type}", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Requested 3D face file not found. Process a sketch first.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=False)
