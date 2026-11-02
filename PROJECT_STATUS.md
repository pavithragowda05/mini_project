# Repository Status — GitHub Main Branch Audit

**Repository:** `pavithragowda05/mini_project`  
**Audit date:** 2026-11-02

This status describes the contents of the GitHub repository. The separately supplied project ZIP contains additional local files and is not automatically synchronized with this repository.

## Present in this repository
- FastAPI backend and static HTML/CSS/JavaScript dashboard.
- Inference pipeline and source implementations for five custom model families.
- Configuration at `configs/config.yaml`.
- Three.js-based 3D mesh generation and export source.
- A third-party CUFS Pix2Pix Keras checkpoint under `models/`.

## Not present in the GitHub repository at audit time
- The custom PyTorch checkpoints:
  - `checkpoints/model_1_pix2pix_best.pth`
  - `checkpoints/model_2_unet_best.pth`
  - `checkpoints/model_3_resnet_gan_best.pth`
- The PCA-Ridge and Patch-KNN checkpoint files under `checkpoints/`.
- The paired raw and processed sketch/photo dataset.
- The authentication and case-management modules found in the separately uploaded ZIP.

Because these weights and dataset files are absent from this repository, the five custom model implementations must not be described as all trained and ready to run from a fresh GitHub clone. The third-party CUFS Pix2Pix model is an additional pretrained candidate, not a substitute for the missing custom PyTorch checkpoints.

## Separately uploaded ZIP audit
The uploaded ZIP was tested independently:
- `python -m pytest tests/ -q`: **19 passed**.
- It contains 100 raw sketch/photo pairs, processed data, PCA-Ridge and Patch-KNN checkpoints, and the CUFS Pix2Pix model.
- After restoring the missing `configs/config.yaml` from this repository, FastAPI startup and a real sample-sketch inference/3D-export smoke test succeeded on CPU.
- The ZIP still lacks the three custom PyTorch `.pth` checkpoints. Its inference pipeline loaded the CUFS Pix2Pix, PCA-Ridge, and Patch-KNN candidates only.
- Eye-only input was tested and did not reliably produce a complete face. The current pretrained model is not verified for partial-sketch completion.

## Important limitations
- A partial or eye-only sketch does not uniquely determine unseen facial features. Any completed face is a model hypothesis, not recovered ground truth.
- The generated 3D mesh is a morphometric surface estimate derived from a 2D generated image, not an identity-verified 3D scan.
- These outputs are investigative research aids only and must not be treated as definitive biometric identification.

## Next steps
1. Synchronize the authentication/case-management source and required data/checkpoints from the ZIP where licensing and file-size limits permit.
2. Train the three missing custom neural models on paired training data and verify each checkpoint through actual inference.
3. Add masked/partial-sketch augmentation and validate eye-only completion on held-out partial inputs.
4. If a React + TypeScript frontend is required, replace or migrate the current vanilla dashboard; the current GitHub frontend is not React.
