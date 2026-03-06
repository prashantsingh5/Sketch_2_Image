"""Streamlit demo for Sketch2Image before/after gallery.

Run:
streamlit run demo/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import streamlit as st
from PIL import Image
import tempfile

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXT


def _normalize_stem(stem: str) -> str:
    lowered = stem.lower()
    for suffix in ["_fake_b", "_real_b", "_real_a", "_fake_a", "_input", "_output"]:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _index_by_stem(root: Path) -> dict[str, Path]:
    files = [p for p in root.rglob("*") if p.is_file() and _is_image(p)]
    index: dict[str, Path] = {}
    for path in files:
        index.setdefault(_normalize_stem(path.stem), path)
    return index


def get_pairs(sketch_dir: Path, generated_dir: Path) -> List[Tuple[Path, Path]]:
    sketch_index = _index_by_stem(sketch_dir)
    generated_index = _index_by_stem(generated_dir)
    keys = sorted(set(sketch_index.keys()) & set(generated_index.keys()))
    return [(sketch_index[k], generated_index[k]) for k in keys]


def _save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


st.set_page_config(page_title="Sketch2Image Live Demo", layout="wide")
st.title("Sketch2Image Live Demo")
st.caption("Interactive before/after gallery for sketch-to-image translation outputs.")

with st.sidebar:
    st.header("Data Sources")
    mode = st.radio("Input mode", options=["Folders", "Upload single pair"], index=0)

pairs: List[Tuple[Path, Path]] = []

if mode == "Folders":
    with st.sidebar:
        sketch_dir_input = st.text_input("Sketch images folder", value="dataset_sketch")
        generated_dir_input = st.text_input("Generated images folder", value="pytorch-CycleGAN-and-pix2pix/output_images")
        max_items = st.slider("Max pairs", min_value=1, max_value=100, value=12)

    sketch_dir = Path(sketch_dir_input).resolve()
    generated_dir = Path(generated_dir_input).resolve()

    if not sketch_dir.exists() or not generated_dir.exists():
        st.warning("Set valid folders in the sidebar to load the gallery.")
        st.stop()

    pairs = get_pairs(sketch_dir, generated_dir)
    if not pairs:
        st.error("No matching pairs found. Ensure filenames align between sketch and generated folders.")
        st.stop()

    pairs = pairs[:max_items]
    st.success(f"Loaded {len(pairs)} matched before/after pairs.")
else:
    with st.sidebar:
        uploaded_before = st.file_uploader("Upload sketch/before image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        uploaded_after = st.file_uploader("Upload generated/after image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_before is None or uploaded_after is None:
        st.info("Upload one before image and one after image to preview instantly.")
        st.stop()

    pairs = [(_save_upload_to_temp(uploaded_before), _save_upload_to_temp(uploaded_after))]
    st.success("Loaded 1 uploaded before/after pair.")

selected_index = st.selectbox(
    "Featured pair",
    options=list(range(len(pairs))),
    format_func=lambda i: pairs[i][0].name,
)

featured_sketch, featured_gen = pairs[selected_index]

st.subheader("Featured Comparison")
col1, col2 = st.columns(2)
with col1:
    st.image(Image.open(featured_sketch), caption=f"Sketch: {featured_sketch.name}", use_container_width=True)
with col2:
    st.image(Image.open(featured_gen), caption=f"Generated: {featured_gen.name}", use_container_width=True)

st.subheader("Gallery")
grid_cols = st.columns(3)
for idx, (sketch_path, gen_path) in enumerate(pairs):
    col = grid_cols[idx % 3]
    with col:
        st.image(Image.open(sketch_path), caption=f"Input: {sketch_path.name}", use_container_width=True)
        st.image(Image.open(gen_path), caption=f"Output: {gen_path.name}", use_container_width=True)
