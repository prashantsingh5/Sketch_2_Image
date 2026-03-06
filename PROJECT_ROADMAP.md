# Sketch2Image Roadmap

## Phase 1: Reliability and Reproducibility

- [x] Convert data pipeline into a configurable CLI.
- [x] Add deterministic data split controls.
- [x] Add project-wide setup and contribution docs.
- [ ] Add automated formatting and linting checks.

## Phase 2: Model Quality

- [ ] Add quantitative metrics: FID, LPIPS, SSIM.
- [ ] Compare baseline Pix2Pix vs tuned variants.
- [ ] Add ablation study for sketch generation kernel parameters.
- [ ] Track experiments with lightweight metadata logging.

## Phase 3: Productization

- [ ] Package inference into a standalone Python CLI.
- [ ] Add batch inference mode and export folder structure.
- [ ] Add Gradio/Streamlit demo app.
- [ ] Add optional ONNX export for faster deployment.

## Phase 4: Showcase Quality

- [ ] Publish benchmark table and visual gallery in README.
- [ ] Add architecture diagram and pipeline illustration.
- [ ] Record a short demo video/GIF for profile visibility.
- [ ] Create tagged releases with changelog.
