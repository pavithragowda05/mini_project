import base64
import html
import os
import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = os.environ.get(
    "FORENSIC_BACKEND_URL", "http://127.0.0.1:8001"
).rstrip("/")
PROCESS_URL = f"{BACKEND_URL}/api/process_sketch"
MODELS = [
    "Pix2Pix Conditional GAN",
    "Deep U-Net (Perceptual Loss)",
    "ResNet Bottleneck LSGAN",
    "Eigenface Dual-PCA + Ridge",
    "Multi-Scale Patch K-NN",
]

st.set_page_config(page_title="Adaptive Forensic Sketch-to-3D Face System", layout="wide")

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#07111f,#101d31); color:#e8eef8}
.block-container {padding-top:1rem; max-width:1500px}
h1,h2,h3,p,label {color:#e8eef8}
.panel {background:rgba(15,28,47,.92); border:1px solid #233b59; border-radius:14px;
        padding:16px; min-height:120px; margin-bottom:12px}
.panel-title {font-size:17px;font-weight:700;margin-bottom:12px;color:#f1f5fb}
.subtle {color:#91a9c4;font-size:12px}
.selected {border:1px solid #46d9cf;background:#102e3c}
.score {background:#21334a;border-radius:8px;height:10px;overflow:hidden;margin:6px 0 12px}
.fill {height:10px;background:linear-gradient(90deg,#39c6d9,#6c8cff)}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="margin-bottom:0;color:#56b6ee">Adaptive Multi-Model Forensic Sketch-to-2D &amp; 3D Face System</h2>
<p class="subtle">Automatic Model Selection &amp; 3D Face Reconstruction</p>
""", unsafe_allow_html=True)

# Upload first so the sketch is visible in the first panel just like the reference.
uploaded = st.file_uploader("Upload Forensic Sketch", type=["png", "jpg", "jpeg"], key="forensic_sketch")

result = None
image_bytes = None
selected_model = "Waiting for sketch"
scores = {}
candidate_previews = {}
error_message = None
preprocessing_only = False
if uploaded is not None:
    try:
        response = requests.post(
            PROCESS_URL,
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/png")},
            timeout=180,
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise requests.HTTPError(
                f"HTTP {response.status_code}: {detail}", response=response
            )
        result = response.json()
        preprocessing_only = result.get("status") == "preprocessing_only"
        selected_model = (
            "OpenCV preprocessing only (not a reconstruction)"
            if preprocessing_only else
            result.get("selected_model_name") or result.get("selected_model_key")
            or result.get("selected_model") or "Model not specified"
        )
        scores = result.get("model_scores") or result.get("all_scores") or result.get("model_probabilities") or {}
        candidate_previews = result.get("candidate_previews") or {}
        image_b64 = (
            result.get("preprocessed_input_b64") if preprocessing_only else
            result.get("generated_2d_b64") or result.get("image_base64")
        )
        if isinstance(image_b64, str) and image_b64:
            if image_b64.startswith("data:") and "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception:
                image_bytes = None
    except requests.exceptions.ConnectionError:
        error_message = f"Cannot connect to backend at {BACKEND_URL}. Start the FastAPI server first."
    except requests.exceptions.Timeout:
        error_message = "Backend request timed out. Try again after checking the backend terminal."
    except Exception as exc:
        error_message = f"Backend error: {exc}"

# TOP ROW: three panels as in the user's screenshot.
upload_col, face_col, viewer_col = st.columns([1.05, 1.05, 1.05], gap="medium")
with upload_col:
    st.markdown('<div class="panel"><div class="panel-title">1. Upload Forensic Sketch</div>', unsafe_allow_html=True)
    if uploaded:
        st.success(f"Selected: {uploaded.name} ({uploaded.size/1024:.1f} KB)")
        st.image(uploaded.getvalue(), use_container_width=True)
    else:
        st.info("Choose a JPG or PNG forensic sketch using the upload control above.")
    st.markdown("</div>", unsafe_allow_html=True)

with face_col:
    st.markdown('<div class="panel"><div class="panel-title">Generated 2D Face</div>', unsafe_allow_html=True)
    if image_bytes:
        st.image(image_bytes, use_container_width=True)
        if preprocessing_only:
            st.caption("OpenCV-preprocessed input only; this is not a model reconstruction.")
            st.warning(result.get("message", "Trained checkpoints are unavailable."))
            st.download_button("Download Preprocessed Input", image_bytes, "preprocessed_input.png", "image/png", use_container_width=True)
        else:
            st.caption(f"Selected checkpoint-backed model: {selected_model}; heuristic ranking, no calibrated confidence.")
            st.download_button("Download Generated 2D Face", image_bytes, "generated_face.png", "image/png", use_container_width=True)
    elif preprocessing_only:
        st.warning(result.get("message", "Trained checkpoints are unavailable."))
    elif error_message:
        st.error(error_message)
    elif uploaded:
        st.warning("The backend response did not include a recognized generated 2D image field.")
    else:
        st.caption("Generated face will appear here after uploading a sketch.")
    st.markdown("</div>", unsafe_allow_html=True)

with viewer_col:
    st.markdown('<div class="panel"><div class="panel-title">Interactive 3D Face Viewer</div>', unsafe_allow_html=True)
    if uploaded and result and not preprocessing_only:
        # Uses the backend's GLB download route. Viewer controls allow rotation/zoom when a valid GLB exists.
        viewer_html = f"""
        <div style="height:290px;background:#0b1728;border-radius:10px;overflow:hidden">
          <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
          <model-viewer src="{BACKEND_URL}/api/download_3d/glb" camera-controls auto-rotate
            rotation-per-second="12deg" shadow-intensity="1" exposure="0.8"
            style="width:100%;height:100%;background:#0b1728"
            alt="Reconstructed 3D face model">
          </model-viewer>
        </div>"""
        components.html(viewer_html, height=300, scrolling=False)
        dcols = st.columns(3)
        for col, ext, label, mime in [
            (dcols[0], "obj", "Download OBJ", "text/plain"),
            (dcols[1], "ply", "Download PLY", "application/octet-stream"),
            (dcols[2], "glb", "Download GLB", "model/gltf-binary"),
        ]:
            try:
                r = requests.get(f"{BACKEND_URL}/api/download_3d/{ext}", timeout=20)
                if r.ok and r.content:
                    col.download_button(label, r.content, f"forensic_face.{ext}", mime, key=f"dl_{ext}", use_container_width=True)
                else:
                    col.button(label, disabled=True, key=f"disabled_{ext}", use_container_width=True)
            except requests.RequestException:
                col.button(label, disabled=True, key=f"disabled_{ext}", use_container_width=True)
    else:
        if preprocessing_only:
            st.warning("3D output is unavailable because trained-model reconstruction is unavailable.")
        else:
            st.info("The 3D model viewer and download buttons appear after a sketch is processed.")
    st.markdown("</div>", unsafe_allow_html=True)

if not (uploaded and result and not preprocessing_only):
    st.stop()

    st.divider()
    # BOTTOM ROW: model selection engine on left, five candidate comparison on right.
    engine_col, compare_col = st.columns([0.95, 1.65], gap="medium")
    with engine_col:
        st.markdown('<div class="panel"><div class="panel-title">Model Selection Engine</div>', unsafe_allow_html=True)
        st.markdown(f"**SELECTED MODEL:** `{selected_model}`")
        st.caption("Candidate scores are heuristic rankings, not probabilities or calibrated confidence.")
        aliases = {
            "Pix2Pix Conditional GAN": ["pix2pix", "pix2pix gan"],
            "Deep U-Net (Perceptual Loss)": ["deep u-net", "unet", "u-net"],
            "ResNet Bottleneck LSGAN": ["resnet", "lsgan"],
            "Eigenface Dual-PCA + Ridge": ["eigenface", "dual-pca", "ridge"],
            "Multi-Scale Patch K-NN": ["patch k-nn", "knn", "k-nn"],
        }
        score_map = scores if isinstance(scores, dict) else {}
        def get_score(name):
            for key, val in score_map.items():
                if str(key).lower() == name.lower() or any(a in str(key).lower() for a in aliases[name]):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            return None
        score_values = [value for value in (get_score(name) for name in MODELS) if value is not None]
        min_score = min(score_values) if score_values else 0.0
        max_score = max(score_values) if score_values else 0.0
        any_scores = False
        for name in MODELS:
            value = get_score(name)
            bar_width = (
                100.0 if max_score == min_score
                else (value - min_score) / (max_score - min_score) * 100.0
            ) if value is not None else 0.0
            is_selected = name.lower() in str(selected_model).lower() or str(selected_model).lower() in name.lower()
            st.markdown(f"**{'★ ' if is_selected else ''}{name}**" + (f" — {value:.3f} heuristic" if value is not None else ""))
            if value is not None:
                any_scores = True
                st.markdown(f'<div class="score"><div class="fill" style="width:{max(0,min(100,bar_width))}%"></div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="score"><div class="fill" style="width:0%"></div></div>', unsafe_allow_html=True)
        if not any_scores:
            st.caption("Candidate scores were not returned by the backend; bars are empty rather than fabricated.")
        st.markdown("</div>", unsafe_allow_html=True)

    with compare_col:
        st.markdown('<div class="panel"><div class="panel-title">Available Candidate Outputs</div>', unsafe_allow_html=True)
        preview_cols = st.columns(5)
        for idx, name in enumerate(MODELS):
            preview = None
            if isinstance(candidate_previews, dict):
                for key, val in candidate_previews.items():
                    if name.lower() in str(key).lower() or str(key).lower() in name.lower() or any(a in str(key).lower() for a in aliases[name]):
                        preview = val
                        break
            with preview_cols[idx]:
                is_selected = name.lower() in str(selected_model).lower() or str(selected_model).lower() in name.lower()
                if is_selected:
                    st.markdown(f"<div style='border:1px solid #45d9cf;border-radius:8px;padding:5px;background:#10313b'><b style='color:#55dfce;font-size:12px'>{html.escape(name)}</b></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='border:1px solid #2a3c53;border-radius:8px;padding:5px'><b style='font-size:12px'>{html.escape(name)}</b></div>", unsafe_allow_html=True)
                if isinstance(preview, dict):
                    preview = preview.get("image_b64") or preview.get("image_base64") or preview.get("data")
                try:
                    if isinstance(preview, str) and preview:
                        if preview.startswith("data:") and "," in preview:
                            preview = preview.split(",", 1)[1]
                        st.image(base64.b64decode(preview), use_container_width=True)
                    elif preview:
                        st.image(preview, use_container_width=True)
                    else:
                        st.markdown('<div style="height:100px;background:#18283b;border-radius:6px;margin-top:8px"></div>', unsafe_allow_html=True)
                        st.caption("Preview unavailable")
                except Exception:
                    st.markdown('<div style="height:100px;background:#18283b;border-radius:6px;margin-top:8px"></div>', unsafe_allow_html=True)
                    st.caption("Preview unavailable")
        st.markdown("</div>", unsafe_allow_html=True)
