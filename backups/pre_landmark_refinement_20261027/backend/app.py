import os
import sys
import shutil
import tempfile
import zipfile
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import ForensicInferencePipeline

app = FastAPI(
    title="Adaptive Multi-Model Forensic Sketch-to-2D and 3D Face Reconstruction System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Pipeline
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    print("[*] Starting FastAPI Server & Loading Forensic Pipeline...")
    pipeline = ForensicInferencePipeline()
    print("[OK] Forensic Pipeline initialized successfully.")

# Serve Static Frontend Files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

output_dir = os.path.abspath("output")
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
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.get("/api/download_3d/obj-bundle")
def download_3d_obj_bundle():
    output_3d_dir = os.path.join("output", "3d")
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
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="Invalid image file format. Supported: PNG, JPG, JPEG, BMP, WEBP.")

    try:
        temp_dir = "data/temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        # Use sanitized filename to prevent path traversal / space issues
        safe_filename = "upload_" + os.path.basename(file.filename).replace(" ", "_")
        temp_path = os.path.join(temp_dir, safe_filename)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load PIL image object directly
        sketch_pil = Image.open(temp_path).convert("RGB")

        # Run pipeline with PIL Image
        res = pipeline.process_sketch(sketch_pil, output_basename="sketch_result")

        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return JSONResponse(content=res)

    except Exception as e:
        print(f"[ERROR] Sketch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")

@app.get("/api/download_3d/{file_type}")
def download_3d(file_type: str):
    file_type = file_type.lower()
    if file_type not in ["obj", "ply", "glb"]:
        raise HTTPException(status_code=400, detail="Invalid 3D format requested. Use obj, ply, or glb.")

    target_file = os.path.join("output", "3d", f"sketch_result.{file_type}")
    if os.path.exists(target_file):
        return FileResponse(path=target_file, filename=f"forensic_3d_face.{file_type}", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Requested 3D face file not found. Process a sketch first.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
