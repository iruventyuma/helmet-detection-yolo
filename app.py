import sys
import os
import pathlib
import base64

pathlib.PosixPath = pathlib.WindowsPath

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

import streamlit as st
import torch
import torch.serialization
from PIL import Image
import numpy as np
import cv2
import time

st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
    color: #f0f0f0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.metric-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
}
.status-safe {
    background: rgba(0,200,83,0.1);
    border-left: 4px solid #00c853;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    color: #00c853;
    font-weight: 500;
}
.status-danger {
    background: rgba(255,68,68,0.1);
    border-left: 4px solid #ff4444;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    color: #ff4444;
    font-weight: 500;
}
.badge-helmet {
    background: rgba(0,200,83,0.15);
    color: #00c853;
    border: 1px solid #00c853;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    display: inline-block;
    margin: 0.2rem;
}
.badge-no-helmet {
    background: rgba(255,68,68,0.15);
    color: #ff4444;
    border: 1px solid #ff4444;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.85rem;
    display: inline-block;
    margin: 0.2rem;
}
.section-title {
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 3px;
    color: #f5c518;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero with shield logo using columns ───────────────────────────────────────
# Create shield SVG and encode as base64
shield_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="80" height="90" viewBox="0 0 24 28">
  <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5L12 1z"
        fill="#f5c518" opacity="0.2"/>
  <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5L12 1z"
        fill="none" stroke="#f5c518" stroke-width="1.5"/>
  <path d="M9 12l2 2 4-4" stroke="#f5c518" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>
"""
encoded = base64.b64encode(shield_svg.encode()).decode()
shield_img_tag = f'data:image/svg+xml;base64,{encoded}'

st.markdown("<div style='padding-top:2rem'></div>", unsafe_allow_html=True)

logo_col, title_col = st.columns([1, 8])
with logo_col:
    st.image(shield_img_tag, width=80)
with title_col:
    st.markdown("""
        <div style="padding-top:0.3rem;">
            <p style="font-size:2rem; font-weight:300; letter-spacing:8px;
                      color:#f5c518; margin:0; line-height:1; font-family:Arial,sans-serif;">
                HELMET DETECT
            </p>
            <p style="color:#666; letter-spacing:3px; text-transform:uppercase;
                      font-size:0.8rem; margin-top:0.4rem;">
                YOLOv5 &middot; Real-time Safety Detection &middot; mAP 83.3%
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="border:none;border-top:1px solid #222;margin:1rem 0;">', unsafe_allow_html=True)

# ── Model Stats ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">mAP@0.5</div><div class="metric-value" style="color:#f5c518">83.3%</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value" style="color:#f5c518">78.1%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value" style="color:#f5c518">79.1%</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-label">Helmet mAP</div><div class="metric-value" style="color:#f5c518">87.0%</div></div>', unsafe_allow_html=True)

st.markdown('<hr style="border:none;border-top:1px solid #222;margin:1rem 0;">', unsafe_allow_html=True)


def preprocess_image(img_rgb):
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl        = clahe.apply(l)
    merged    = cv2.merge((cl, a, b))
    enhanced  = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)


def run_detection(img_rgb):
    pathlib.PosixPath = pathlib.WindowsPath
    from yolov5.models.yolo import DetectionModel
    from yolov5.models.common import AutoShape
    torch.serialization.add_safe_globals([DetectionModel])
    ckpt    = torch.load('best.pt', map_location='cpu', weights_only=False)
    model   = ckpt['model'].float().fuse().eval()
    model   = AutoShape(model)
    results = model(img_rgb)
    return results


st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is not None:
    image     = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.markdown('<hr style="border:none;border-top:1px solid #222;margin:1rem 0;">', unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-title">Original</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)
        with st.spinner("Running detection..."):
            try:
                processed  = preprocess_image(img_array)
                t0         = time.time()
                results    = run_detection(processed)
                inf_ms     = round((time.time() - t0) * 1000, 1)
                results.render()
                result_img = results.ims[0]
                st.image(result_img, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

    st.markdown('<hr style="border:none;border-top:1px solid #222;margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detection Summary</div>', unsafe_allow_html=True)

    detections = results.pandas().xyxy[0]
    total      = len(detections)
    helmets    = len(detections[detections['name'] == 'helmet'])
    no_helmets = len(detections[detections['name'] == 'no_helmet'])

    if total == 0:
        st.info("⚠️ No objects detected. Try a clearer or closer image.")
    else:
        if helmets > 0:
            st.markdown(f'<div class="status-safe">✅ {helmets} person(s) wearing a helmet detected.</div>', unsafe_allow_html=True)
        if no_helmets > 0:
            st.markdown(f'<div class="status-danger">⚠️ {no_helmets} person(s) WITHOUT a helmet detected!</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Detected</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">With Helmet</div><div class="metric-value" style="color:#00c853">{helmets}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">No Helmet</div><div class="metric-value" style="color:#ff4444">{no_helmets}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Inference (ms)</div><div class="metric-value">{inf_ms}</div></div>', unsafe_allow_html=True)

    if total > 0:
        st.markdown('<hr style="border:none;border-top:1px solid #222;margin:1rem 0;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Per Detection</div>', unsafe_allow_html=True)
        for i, row in detections.iterrows():
            name  = row['name']
            conf  = round(row['confidence'] * 100, 1)
            label = "🪖 Helmet" if name == 'helmet' else "❌ No Helmet"
            badge = "badge-helmet" if name == 'helmet' else "badge-no-helmet"
            st.markdown(f'<span class="{badge}">{label} — {conf}% confidence</span>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="border:2px dashed #222; border-radius:12px; padding:3rem;
                text-align:center; background:#141414; margin-top:1rem">
        <p style="font-size:2.5rem; margin:0">📸</p>
        <p style="color:#555; margin:0.5rem 0 0;">
            Upload a JPG or PNG image to start detection
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="border:none;border-top:1px solid #222;margin:1rem 0;">', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#333; font-size:0.75rem; letter-spacing:2px;">HELMET DETECTION · YOLOV5 · STREAMLIT</p>', unsafe_allow_html=True)