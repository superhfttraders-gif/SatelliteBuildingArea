import base64
from io import BytesIO
from typing import Optional, Dict, Any, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from src.building_pipeline import predict_building_mask
from src.car_scale_pipeline import estimate_gsd_from_cars
from src.metrics import area_px, area_m2


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Buildings Area Estimator", layout="wide")

st.title("🌐️ Segmentation + GSD + Building Area")
st.caption("Загрузи снимок → получишь маску зданий, масштаб (GSD) и площадь в пикселях и м².")


# -----------------------------
# Helpers
# -----------------------------
def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Return image as RGB np.uint8 array [H,W,3]."""
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img, dtype=np.uint8)


def make_overlay(image_rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    color = np.zeros_like(overlay)
    color[..., 0] = 255  # red channel

    overlay = np.where(
        mask01[..., None].astype(bool),
        overlay * (1 - alpha) + color * alpha,
        overlay,
    )

    return overlay.clip(0, 255).astype(np.uint8)


def header_bar_left(image_shape: Tuple[int, int, int]) -> None:
    html = (
        f'<div style="padding:0.55rem 0.75rem;'
        f'border-bottom:1px solid rgba(49,51,63,0.2);'
        f'margin-bottom:0.6rem;'
        f'font-size:0.95rem;'
        f'color:rgba(49,51,63,0.9);">'
        f'<span style="font-weight:600;">Image shape:</span> '
        f'<code style="font-size:0.9rem;">{tuple(image_shape)}</code>'
        f'<span style="margin-left:0.75rem;color:rgba(49,51,63,0.65);">(H, W, C)</span>'
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def header_bar_right_with_mask_link(mask01: np.ndarray) -> None:
    """
    Важно: HTML должен быть БЕЗ ведущих пробелов/отступов, иначе Streamlit/Markdown
    превращает его в code-block и показывает как текст.
    """
    mask_png = (mask01 * 255).astype(np.uint8)
    pil_mask = Image.fromarray(mask_png)
    buf = BytesIO()
    pil_mask.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    html = (
        '<div style="padding:0.55rem 0.75rem;'
        'border-bottom:1px solid rgba(49,51,63,0.2);'
        'margin-bottom:0.6rem;'
        'font-size:0.95rem;'
        'color:rgba(49,51,63,0.9);'
        'display:flex;'
        'align-items:center;'
        'justify-content:space-between;'
        'gap:0.75rem;">'
        '<div>'
        '<span style="font-weight:600;">Overlay:</span> '
        '<span style="color:rgba(49,51,63,0.65);">buildings highlighted</span>'
        "</div>"
        f'<a href="data:image/png;base64,{b64}" download="buildings_mask.png" '
        'style="text-decoration:none;'
        'font-size:0.95rem;'
        'padding:0.15rem 0.45rem;'
        'border:1px solid rgba(49,51,63,0.25);'
        'border-radius:0.4rem;'
        'color:rgba(49,51,63,0.9);'
        'background:rgba(49,51,63,0.04);'
        'white-space:nowrap;'
        'line-height:1.2;">'
        "⬇ Mask (PNG)</a>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Buildings segmentation")
b_tile = st.sidebar.selectbox("Buildings tile size", [256, 384, 512, 640, 768, 1024], index=0)
b_stride = st.sidebar.selectbox("Buildings stride", [128, 192, 256, 320, 384, 512], index=0)
b_threshold = st.sidebar.slider("Buildings mask threshold", 0.0, 1.0, 0.86, 0.01)

st.sidebar.subheader("Car-based scale (GSD)")
c_tile = st.sidebar.selectbox("Cars tile size", [256, 384, 512, 640, 768, 1024], index=5)
c_stride = st.sidebar.selectbox("Cars stride", [128, 192, 256, 320, 384, 512], index=5)

st.sidebar.subheader("Visualization")
overlay_alpha = st.sidebar.slider("Overlay alpha", 0.0, 1.0, 0.40, 0.05)

st.sidebar.divider()
run_btn = st.sidebar.button("🚀 Run", type="primary")


# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])
if uploaded is None:
    st.info("Загрузи снимок, затем нажми **Run** в боковой панели.")
    st.stop()

image = read_uploaded_image(uploaded)


# -----------------------------
# Session cache to prevent disappearing overlay on reruns/clicks
# -----------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None
if "sig" not in st.session_state:
    st.session_state["sig"] = None

sig = (
    tuple(image.shape),
    int(b_tile),
    int(b_stride),
    float(b_threshold),
    int(c_tile),
    int(c_stride),
    float(overlay_alpha),
)

if st.session_state["sig"] != sig:
    st.session_state["sig"] = sig
    st.session_state["results"] = None


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input")
    header_bar_left(image.shape)
    st.image(image, use_container_width=True)

# If nothing computed yet and user didn't run -> stop
if st.session_state["results"] is None and not run_btn:
    st.warning("Параметры выставлены. Теперь нажми **Run** в боковой панели.")
    st.stop()


# -----------------------------
# Compute (only if needed)
# -----------------------------
if run_btn or st.session_state["results"] is None:
    with st.spinner("🏢️ Running building segmentation..."):
        bin_mask, b_debug = predict_building_mask(
            image_rgb=image,
            tile_size=b_tile,
            stride=b_stride,
            threshold=b_threshold,
        )

    bin_mask = (bin_mask > 0).astype(np.uint8)

    with st.spinner("🚗 Estimating GSD from cars..."):
        gsd_m_per_px, car_stats = estimate_gsd_from_cars(
            image_rgb=image,
            tile_size=c_tile,
            stride=c_stride,
        )

    b_area_px_val = area_px(bin_mask)
    b_area_m2_val = None if gsd_m_per_px is None else area_m2(b_area_px_val, gsd_m_per_px)
    overlay = make_overlay(image, bin_mask, alpha=overlay_alpha)

    st.session_state["results"] = {
        "bin_mask": bin_mask,
        "b_debug": b_debug,
        "gsd_m_per_px": gsd_m_per_px,
        "car_stats": car_stats,
        "b_area_px": b_area_px_val,
        "b_area_m2": b_area_m2_val,
        "overlay": overlay,
    }

res = st.session_state["results"]
bin_mask = res["bin_mask"]
b_debug = res["b_debug"]
gsd_m_per_px = res["gsd_m_per_px"]
car_stats = res["car_stats"]
b_area_px = res["b_area_px"]
b_area_m2 = res["b_area_m2"]
overlay = res["overlay"]


# -----------------------------
# Output (only overlay, no tabs/captions)
# -----------------------------
with right:
    st.subheader("Outputs")
    header_bar_right_with_mask_link(bin_mask)
    st.image(overlay, use_container_width=True)


# -----------------------------
# Metrics
# -----------------------------
st.divider()
st.subheader("📊 Metrics")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Buildings area (px)", f"{b_area_px:,}")

m2.metric("GSD (m/px)", "N/A" if gsd_m_per_px is None else f"{gsd_m_per_px:.6f}")

m3.metric("Buildings area (m²)", "N/A" if b_area_m2 is None else f"{b_area_m2:,.2f}")

cars_count = None
if isinstance(car_stats, dict):
    cars_count = car_stats.get("cars_count", None)
m4.metric("Cars detected", "N/A" if cars_count is None else str(cars_count))


# -----------------------------
# Debug (bottom)
# -----------------------------
st.divider()
with st.expander("🐜 Debug (models + intermediate stats)", expanded=False):
    st.write("Buildings debug:")
    st.json(b_debug if isinstance(b_debug, dict) else {})
    st.write("Cars stats:")
    st.json(car_stats if isinstance(car_stats, dict) else {})


# -----------------------------
# Downloads (keep bottom button as requested)
# -----------------------------
st.subheader("💾 Export")

col_a, col_b = st.columns(2)

with col_a:
    mask_png = (bin_mask * 255).astype(np.uint8)
    pil_mask = Image.fromarray(mask_png)
    buf = BytesIO()
    pil_mask.save(buf, format="PNG")
    st.download_button(
        label="Download buildings mask (PNG)",
        data=buf.getvalue(),
        file_name="buildings_mask.png",
        mime="image/png",
    )

with col_b:
    import json

    report = {
        "image_shape": list(image.shape),
        "buildings": {
            "tile_size": int(b_tile),
            "stride": int(b_stride),
            "threshold": float(b_threshold),
            "area_px": int(b_area_px),
            "area_m2": None if b_area_m2 is None else float(b_area_m2),
        },
        "scale": {
            "tile_size": int(c_tile),
            "stride": int(c_stride),
            "gsd_m_per_px": None if gsd_m_per_px is None else float(gsd_m_per_px),
            "car_stats": car_stats if isinstance(car_stats, dict) else {},
        },
    }

    st.download_button(
        label="Download report (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="report.json",
        mime="application/json",
    )
