"""GRIDGUARD AI: Simple, Friendly, and Intelligent Traffic Assistant.

A modern SaaS-style light interface for analyzing road footage and discovering
the best available route with the lowest congestion.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import tempfile
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.route_engine
import src.vehicle_detection
importlib.reload(src.route_engine)
importlib.reload(src.vehicle_detection)

from src.route_engine import (  # noqa: E402
    RoadSegment,
    RouteRecommendationResult,
    build_network_graph_figure,
    build_route_comparison_bar_chart,
    evaluate_routes,
)
from src.vehicle_detection import analyze_video  # noqa: E402

st.set_page_config(
    page_title="GRIDGUARD AI — Smart Traffic Assistant",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Light SaaS Design System Styling
st.markdown("""
<style>
:root {
    color-scheme: light;
    --gg-bg: #F5F8FC;
    --gg-card: #FFFFFF;
    --gg-primary: #2475C8;
    --gg-primary-hover: #1b62ab;
    --gg-primary-light: #EAF3FF;
    --gg-success: #16865B;
    --gg-success-light: #E8F6EF;
    --gg-warning: #D99A27;
    --gg-warning-light: #FEF7EA;
    --gg-danger: #DC5656;
    --gg-danger-light: #FDF1F1;
    --gg-text: #172B45;
    --gg-muted: #64748B;
    --gg-border: #DCE4EF;
    --gg-shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.04);
    --gg-shadow-md: 0 4px 14px rgba(15, 23, 42, 0.06), 0 2px 6px rgba(15, 23, 42, 0.04);
}

/* Global resets for pure light theme */
html, body, .stApp {
    background-color: var(--gg-bg) !important;
    color: var(--gg-text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
    max-width: 1040px;
    padding: 1.5rem 1.25rem 4rem;
}

/* Clean, modern typography */
h1, h2, h3, h4, h5 {
    color: var(--gg-text) !important;
    letter-spacing: -0.025em;
    font-weight: 700;
}
p {
    color: var(--gg-muted);
    font-size: 0.95rem;
    line-height: 1.55;
}

/* Navigation Top Bar */
.gg-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 1.25rem;
    border-bottom: 1px solid var(--gg-border);
    margin-bottom: 2rem;
}
.gg-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.gg-logo-badge {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: var(--gg-primary-light);
    color: var(--gg-primary);
    display: grid;
    place-items: center;
    font-size: 1.2rem;
    font-weight: 800;
}
.gg-brand-title {
    font-size: 1.05rem;
    font-weight: 750;
    color: var(--gg-text);
    line-height: 1.15;
}
.gg-brand-sub {
    font-size: 0.78rem;
    color: var(--gg-muted);
    font-weight: 500;
}
.gg-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    background: var(--gg-card);
    border: 1px solid var(--gg-border);
    color: var(--gg-muted);
}
.gg-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--gg-success);
}
.gg-status-pill.processing .gg-status-dot {
    background: var(--gg-primary);
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.2); }
}

/* Home Page Minimal Hero */
.home-hero {
    text-align: center;
    max-width: 600px;
    margin: 1rem auto 2rem;
}
.home-title {
    font-size: 2.25rem;
    font-weight: 750;
    color: var(--gg-text);
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.home-subtitle {
    font-size: 1.05rem;
    color: var(--gg-muted);
    margin: 0 auto;
}

/* One Large Upload Box */
.upload-wrapper {
    background: var(--gg-card);
    border: 2px dashed var(--gg-border);
    border-radius: 16px;
    padding: 2.25rem 2rem 1.5rem;
    text-align: center;
    box-shadow: var(--gg-shadow-sm);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1.5rem;
}
.upload-wrapper:hover {
    border-color: var(--gg-primary);
    box-shadow: var(--gg-shadow-md);
}
.upload-icon-circle {
    width: 54px;
    height: 54px;
    border-radius: 50%;
    background: var(--gg-primary-light);
    color: var(--gg-primary);
    display: grid;
    place-items: center;
    font-size: 1.5rem;
    margin: 0 auto 1rem;
}
.upload-main-text {
    font-size: 1.15rem;
    font-weight: 650;
    color: var(--gg-text);
    margin-bottom: 0.25rem;
}
.upload-sub-text {
    font-size: 0.88rem;
    color: var(--gg-muted);
    margin-bottom: 1rem;
}

/* Selected Video File Card */
.selected-video-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-radius: 12px;
    background: var(--gg-primary-light);
    border: 1px solid rgba(36, 117, 200, 0.2);
    margin: 1rem 0;
    text-align: left;
}
.selected-video-info {
    display: flex;
    align-items: center;
    gap: 12px;
}
.selected-video-name {
    font-size: 0.92rem;
    font-weight: 650;
    color: var(--gg-text);
}
.selected-video-size {
    font-size: 0.78rem;
    color: var(--gg-muted);
}

/* Button Stylings */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.4rem;
    transition: all 0.16s ease;
}
div.stButton > button[kind="primary"] {
    background-color: var(--gg-primary) !important;
    border: 1px solid var(--gg-primary) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(36, 117, 200, 0.25);
}
div.stButton > button[kind="primary"]:hover {
    background-color: var(--gg-primary-hover) !important;
    box-shadow: 0 4px 12px rgba(36, 117, 200, 0.35);
    transform: translateY(-1px);
}
div.stButton > button[kind="secondary"], div.stButton > button:not([kind="primary"]) {
    background-color: #FFFFFF !important;
    border: 1px solid var(--gg-border) !important;
    color: var(--gg-text) !important;
}
div.stButton > button[kind="secondary"]:hover, div.stButton > button:not([kind="primary"]):hover {
    border-color: var(--gg-primary) !important;
    color: var(--gg-primary) !important;
    background-color: var(--gg-primary-light) !important;
}

/* Live Processing Screen */
.processing-header {
    text-align: center;
    margin: 1.5rem auto 1.5rem;
}
.live-metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 1rem 0 1.5rem;
}
.live-metric-card {
    background: var(--gg-card);
    border: 1px solid var(--gg-border);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    box-shadow: var(--gg-shadow-sm);
}
.live-metric-label {
    font-size: 0.8rem;
    color: var(--gg-muted);
    font-weight: 600;
    margin-bottom: 4px;
}
.live-metric-value {
    font-size: 1.45rem;
    font-weight: 750;
    color: var(--gg-text);
}

/* Results: Best Route Recommendation Card */
.best-route-card {
    background: var(--gg-card);
    border: 1.5px solid #16865B;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    box-shadow: 0 4px 20px rgba(22, 134, 91, 0.08), var(--gg-shadow-sm);
    margin-bottom: 2rem;
}
.best-route-card.all-congested {
    border-color: var(--gg-warning);
    box-shadow: 0 4px 20px rgba(217, 154, 39, 0.08), var(--gg-shadow-sm);
}
.best-route-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    background: var(--gg-success-light);
    color: var(--gg-success);
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.best-route-card.all-congested .best-route-badge {
    background: var(--gg-warning-light);
    color: var(--gg-warning);
}
.best-route-name {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--gg-text);
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}
.best-route-tagline {
    font-size: 1.02rem;
    color: var(--gg-success);
    font-weight: 650;
    margin-bottom: 12px;
}
.best-route-card.all-congested .best-route-tagline {
    color: var(--gg-warning);
}
.best-route-explanation {
    font-size: 0.95rem;
    color: var(--gg-muted);
    line-height: 1.6;
    margin-bottom: 1.25rem;
}
.route-badges-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}
.route-badge-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--gg-bg);
    border: 1px solid var(--gg-border);
    padding: 6px 14px;
    border-radius: 10px;
    font-size: 0.85rem;
    color: var(--gg-text);
    font-weight: 600;
}

/* Clean Cards for Visualizations */
.ui-card {
    background: var(--gg-card);
    border: 1px solid var(--gg-border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: var(--gg-shadow-sm);
    margin-bottom: 1.5rem;
}
.ui-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--gg-text);
    margin-bottom: 0.25rem;
}
.ui-card-subtitle {
    font-size: 0.84rem;
    color: var(--gg-muted);
    margin-bottom: 1rem;
}

/* Results Bottom Actions Bar */
.bottom-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding-top: 1.5rem;
    border-top: 1px solid var(--gg-border);
    margin-top: 2rem;
}

/* Streamlit Expander styling */
div[data-testid="stExpander"] {
    background: var(--gg-card);
    border: 1px solid var(--gg-border);
    border-radius: 12px;
    box-shadow: var(--gg-shadow-sm);
}

@media (max-width: 768px) {
    .live-metrics-row { grid-template-columns: 1fr; }
    .home-title { font-size: 1.75rem; }
    .bottom-actions { flex-direction: column; align-items: stretch; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_detector_model(model_name: str) -> YOLO:
    """Reuses initialized YOLO detector model across sessions."""
    return YOLO(model_name)


def render_header(status: str = "Ready") -> None:
    """Renders a simple, elegant top navigation bar."""
    status_lower = status.lower()
    is_processing = status_lower == "processing"
    pill_class = "processing" if is_processing else "ready"
    display_status = "Analyzing..." if is_processing else "Ready"

    st.markdown(
        f"""
        <div class="gg-nav">
          <div class="gg-brand">
            <div class="gg-logo-badge">🚦</div>
            <div>
              <div class="gg-brand-title">GRIDGUARD AI</div>
              <div class="gg-brand-sub">Smart Traffic Assistant</div>
            </div>
          </div>
          <div class="gg-status-pill {pill_class}">
            <span class="gg-status-dot"></span>
            <span>{display_status}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_session_state() -> None:
    """Resets session state to analyze a new video."""
    st.session_state.metrics_df = None
    st.session_state.route_result = None
    st.session_state.analysis_status = "Ready"
    st.session_state.last_video_name = None
    st.session_state.annotated_path = None
    st.session_state.elapsed_seconds = 0.0
    st.session_state.upload_generation += 1


def apply_light_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    """Applies a clean, minimal light SaaS theme to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=32, b=10),
        height=height,
        font=dict(color="#64748B", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#DCE4EF", font_color="#172B45"),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    return fig


# State initialization
for key, default in {
    "metrics_df": None,
    "route_result": None,
    "analysis_status": "Ready",
    "upload_generation": 0,
    "max_frames": 200,
    "confidence": 0.25,
    "local_video_path": "",
    "annotated_path": None,
    "last_video_name": None,
    "elapsed_seconds": 0.0,
    "corridor_mode": "corridor_baseline",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

render_header(st.session_state.analysis_status)

df: Optional[pd.DataFrame] = st.session_state.metrics_df
route_res: Optional[RouteRecommendationResult] = st.session_state.route_result
scan_requested = False
uploaded_file = None

# =========================================================================
# 1. SIMPLE HOME PAGE (STEP 1: VIDEO UPLOAD)
# =========================================================================
if df is None or df.empty:
    st.markdown("""
    <div class="home-hero">
      <h1 class="home-title">Find a better route.</h1>
      <p class="home-subtitle">Upload a traffic video and let AI understand road conditions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Centered single upload card
    center_l, center_col, center_r = st.columns([1, 8, 1])
    with center_col:
        st.markdown("""
        <div class="upload-wrapper">
          <div class="upload-icon-circle">📹</div>
          <div class="upload-main-text">Upload your traffic video</div>
          <div class="upload-sub-text">Drag and drop or browse your device to begin</div>
        </div>
        """, unsafe_allow_html=True)

        upload_key = f"traffic_clip_{st.session_state.upload_generation}"
        uploaded_file = st.file_uploader(
            "Upload your traffic video",
            type=["mp4", "avi", "mov", "mkv"],
            key=upload_key,
            label_visibility="collapsed",
        )

        # Show selected video info or bundled sample option
        sample_path = ROOT / "results" / "gridguard_annotated.mp4"
        has_sample = sample_path.is_file()

        if uploaded_file is not None:
            st.markdown(f"""
            <div class="selected-video-card">
              <div class="selected-video-info">
                <span style="font-size:1.3rem;">🎬</span>
                <div>
                  <div class="selected-video-name">{escape(uploaded_file.name)}</div>
                  <div class="selected-video-size">{uploaded_file.size / (1024 * 1024):.2f} MB · Video ready for analysis</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Choose another video", key="replace_video_btn"):
                st.session_state.upload_generation += 1
                st.rerun()

        elif has_sample and not st.session_state.local_video_path:
            sample_col_a, sample_col_b = st.columns([2, 1])
            with sample_col_a:
                st.caption("Don't have a traffic clip handy? You can test with bundled sample footage.")
            with sample_col_b:
                if st.button("Use Sample Video", key="sample_video_btn", use_container_width=True):
                    st.session_state.local_video_path = str(sample_path)
                    st.session_state.last_video_name = "sample_traffic_clip.mp4"
                    st.rerun()

        if st.session_state.local_video_path and uploaded_file is None:
            st.info(f"Loaded sample video: **{st.session_state.last_video_name}**")
            if st.button("Remove sample video", key="clear_sample_btn"):
                st.session_state.local_video_path = ""
                st.rerun()

        can_analyze = uploaded_file is not None or bool(st.session_state.local_video_path.strip())

        st.write("")
        scan_requested = st.button(
            "Analyze Traffic",
            type="primary",
            disabled=not can_analyze,
            use_container_width=True,
            key="analyze_traffic_main_btn",
        )
        st.markdown("<p style='text-align:center; font-size:0.8rem; margin-top:8px;'>Uploaded traffic footage is analyzed locally to understand road congestion.</p>", unsafe_allow_html=True)

        # Single clean expander for advanced settings
        with st.expander("Advanced settings", expanded=False):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                st.markdown("**Alternative Route Options**")
                mode_choice = st.radio(
                    "Compare against:",
                    options=[
                        ("corridor_baseline", "Connected road network (Road A, Road B, Road C)"),
                        ("isolated_road", "Observed road only (single road analysis)"),
                    ],
                    format_func=lambda x: x[1],
                    index=0,
                    key="corridor_mode_radio",
                )
                st.session_state.corridor_mode = mode_choice[0]
            with adv_c2:
                st.markdown("**Analysis Detail**")
                st.session_state.max_frames = st.slider(
                    "Maximum frames to analyze",
                    min_value=30,
                    max_value=500,
                    value=180,
                    step=25,
                    help="Analyzing more frames provides longer traffic observation.",
                )
                st.session_state.confidence = st.slider(
                    "Detection sensitivity",
                    min_value=0.15,
                    max_value=0.85,
                    value=0.25,
                    step=0.05,
                    help="Controls how strictly vehicles are detected.",
                )

# =========================================================================
# 2. CLEAN AI PROCESSING SCREEN (STEP 2)
# =========================================================================
scan_completed = False
if scan_requested:
    temp_path: Optional[str] = None
    started = time.perf_counter()
    st.session_state.analysis_status = "Processing"

    st.markdown("""
    <div class="processing-header">
      <h2 style="font-size:1.85rem; margin-bottom:0.35rem;">AI is analyzing your traffic video</h2>
      <p style="font-size:1rem; margin:0 auto;">Detecting vehicles and understanding road congestion.</p>
    </div>
    """, unsafe_allow_html=True)

    # Clean progress bar
    progress_slot = st.progress(0, text="Reading video...")
    activity_slot = st.empty()

    # Three simple live metrics: Vehicles detected, Traffic density, Current congestion level
    metrics_slot = st.empty()

    # ONE main video preview at a time
    main_preview_slot = st.empty()

    try:
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            video_source = temp_path
            display_name = uploaded_file.name
        elif st.session_state.local_video_path.strip():
            video_source = str(Path(st.session_state.local_video_path.strip()).expanduser())
            if not Path(video_source).is_file():
                raise FileNotFoundError(f"Video file not found: {video_source}")
            display_name = Path(video_source).name
        else:
            raise ValueError("No video provided for analysis.")

        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        output_video_path = results_dir / "gridguard_annotated.mp4"
        weights_file = ROOT / "yolov8n.pt"
        model_name = str(weights_file) if weights_file.is_file() else "yolov8n.pt"
        detector = load_detector_model(model_name)

        def on_progress_update(pct: float, stage_text: str) -> None:
            percent_int = min(100, max(0, int(pct * 100)))
            # Map technical stages to human-friendly activity labels
            friendly_stage = "Reading video..."
            if pct > 0.10 and pct <= 0.40:
                friendly_stage = "Detecting vehicles..."
            elif pct > 0.40 and pct <= 0.70:
                friendly_stage = "Measuring traffic density..."
            elif pct > 0.70 and pct <= 0.88:
                friendly_stage = "Evaluating congestion..."
            elif pct > 0.88:
                friendly_stage = "Preparing route insights..."

            progress_slot.progress(percent_int, text=f"{friendly_stage} ({percent_int}%)")

        def on_frame_update(annotated_img, row_data) -> None:
            # ONE clean main video preview
            main_preview_slot.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            # Three human-friendly live metrics
            veh_count = row_data["vehicle_count"]
            density_label = "Light" if veh_count < 6 else ("Moderate" if veh_count < 14 else "Heavy")
            risk_pct = int(row_data["risk_score"] * 100)
            risk_label = "Low" if risk_pct < 35 else ("Moderate" if risk_pct < 68 else "High")

            with metrics_slot.container():
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"<div class='live-metric-card'><div class='live-metric-label'>Vehicles detected</div><div class='live-metric-value'>{veh_count}</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='live-metric-card'><div class='live-metric-label'>Traffic density</div><div class='live-metric-value'>{density_label}</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='live-metric-card'><div class='live-metric-label'>Current congestion level</div><div class='live-metric-value'>{risk_label} ({risk_pct}%)</div></div>", unsafe_allow_html=True)

        video_args = {
            "video_path": video_source,
            "output_path": str(output_video_path),
            "max_frames": st.session_state.max_frames,
            "image_size": 320,
            "conf": st.session_state.confidence,
            "model": detector,
            "progress_callback": on_progress_update,
            "frame_callback": on_frame_update,
            "preview_every": 2,
        }

        analyzed_df, saved_video = analyze_video(**video_args)

        # Generate route recommendations
        on_progress_update(0.95, "Preparing route insights...")
        eval_mode = "isolated_road" if st.session_state.corridor_mode == "isolated_road" else "corridor_baseline"
        route_recommendation = evaluate_routes(
            primary_df=analyzed_df,
            primary_name="Road A (Main Road)",
            mode=eval_mode,
        )

        elapsed_total = time.perf_counter() - started
        progress_slot.progress(100, text="Analysis ready!")

        # Save results
        (results_dir / "gridguard_metrics.csv").write_text(analyzed_df.to_csv(index=False), encoding="utf-8-sig")

        st.session_state.metrics_df = analyzed_df
        st.session_state.route_result = route_recommendation
        st.session_state.annotated_path = saved_video
        st.session_state.last_video_name = display_name
        st.session_state.elapsed_seconds = elapsed_total
        st.session_state.analysis_status = "Completed"
        scan_completed = True

    except Exception as exc:
        st.session_state.analysis_status = "Error"
        st.error(f"Could not complete analysis: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

if scan_completed:
    st.rerun()

# =========================================================================
# 3. CLEAN & FRIENDLY RESULTS PAGE (STEP 3)
# =========================================================================
df = st.session_state.metrics_df
route_res = st.session_state.route_result

if df is not None and not df.empty and route_res is not None:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
      <h2 style="font-size:2rem; margin-bottom:0.25rem;">Your traffic analysis is ready.</h2>
      <p style="margin:0;">Here is the road condition analysis and the recommended route to avoid traffic.</p>
    </div>
    """, unsafe_allow_html=True)

    rec_seg = next((s for s in route_res.segments if s.segment_id == route_res.recommended_segment_id), route_res.segments[0])
    is_congested_all = route_res.all_congested
    card_extra_class = "all-congested" if is_congested_all else ""
    badge_title = "LEAST DELAY ROUTE" if is_congested_all else "BEST AVAILABLE ROUTE"
    tagline = "Lowest observed congestion among available routes." if not is_congested_all else "All routes experiencing heavy traffic; this route offers the lowest delay."

    # Best Route Recommendation Prominent Card
    st.markdown(f"""
    <div class="best-route-card {card_extra_class}">
      <div class="best-route-badge">⭐ {badge_title}</div>
      <div class="best-route-name">{escape(rec_seg.name)}</div>
      <div class="best-route-tagline">{tagline}</div>
      <div class="best-route-explanation">{escape(route_res.recommendation_reason)}</div>
      <div class="route-badges-strip">
        <div class="route-badge-item">🚦 Congestion: <b>{int(rec_seg.congestion_score * 100)}%</b> ({escape(rec_seg.density_indicator)})</div>
        <div class="route-badge-item">🚗 Average Traffic: <b>{rec_seg.avg_vehicles_per_frame:.1f} vehicles/frame</b></div>
        <div class="route-badge-item">⏱️ Estimated Travel Time: <b>{rec_seg.estimated_travel_time_min} mins</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Congestion Comparison Chart & Road Network
    grid_c1, grid_c2 = st.columns([1, 1])

    with grid_c1:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">Congestion Comparison</div>
          <div class="ui-card-subtitle">Real observed congestion values across available roads.</div>
        </div>
        """, unsafe_allow_html=True)
        bar_fig = build_route_comparison_bar_chart(route_res.segments, route_res.recommended_segment_id, height=310)
        st.plotly_chart(bar_fig, use_container_width=True)

    with grid_c2:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">Road Network Overview</div>
          <div class="ui-card-subtitle">Green indicates lower congestion; recommended path is highlighted.</div>
        </div>
        """, unsafe_allow_html=True)
        net_fig = build_network_graph_figure(route_res.graph, route_res.segments, route_res.recommended_segment_id, height=310)
        st.plotly_chart(net_fig, use_container_width=True)

    # 2. Helpful Charts: Vehicle Distribution & Traffic Over Time
    chart_c1, chart_c2 = st.columns([1, 1])

    with chart_c1:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">Vehicles Over Time</div>
          <div class="ui-card-subtitle">Vehicle count changes across the video footage.</div>
        </div>
        """, unsafe_allow_html=True)
        act_fig = px.line(
            df,
            x="time_seconds",
            y=["vehicle_count", "roi_vehicle_count"],
            labels={"time_seconds": "Video Time (seconds)", "value": "Vehicles", "variable": "Area"},
            color_discrete_sequence=["#2475C8", "#16865B"],
        )
        act_fig.for_each_trace(lambda t: t.update(name="Total Vehicles" if "vehicle_count" in t.name else "In Congested Area"))
        act_fig.update_traces(line_width=2.5)
        st.plotly_chart(apply_light_theme(act_fig, 260), use_container_width=True)

    with chart_c2:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">Vehicle Types Detected</div>
          <div class="ui-card-subtitle">Breakdown of cars, motorcycles, buses, and trucks.</div>
        </div>
        """, unsafe_allow_html=True)
        class_cols = {"Cars": "car_count", "Motorcycles": "motorcycle_count", "Buses": "bus_count", "Trucks": "truck_count"}
        class_summary = pd.DataFrame({
            "Vehicle Type": list(class_cols.keys()),
            "Detections": [int(df[col].fillna(0).sum()) for col in class_cols.values()],
        })
        class_fig = px.bar(
            class_summary,
            x="Vehicle Type",
            y="Detections",
            color="Vehicle Type",
            text="Detections",
            color_discrete_sequence=["#2475C8", "#5A9FE6", "#16865B", "#D99A27"],
        )
        class_fig.update_layout(showlegend=False)
        st.plotly_chart(apply_light_theme(class_fig, 260), use_container_width=True)

    # 3. Optional Technical Details
    with st.expander("Technical details & full measurements", expanded=False):
        st.markdown("**Road Corridor Telemetry Table**")
        st.dataframe(route_res.comparison_df, hide_index=True, use_container_width=True)
        st.markdown("**Annotated Video Preview**")
        annotated_file = st.session_state.get("annotated_path")
        if annotated_file and os.path.isfile(annotated_file):
            st.video(annotated_file)
        st.markdown("**Frame-by-Frame Measurements**")
        st.dataframe(df, hide_index=True, use_container_width=True)

    # 4. Simple Results Actions Bar
    st.write("")
    act_col1, act_col2, act_col3 = st.columns([1.2, 1, 1])

    with act_col1:
        if st.button("🔄 Analyze another video", type="primary", use_container_width=True):
            reset_session_state()
            st.rerun()

    with act_col2:
        annotated_file = st.session_state.get("annotated_path")
        if annotated_file and os.path.isfile(annotated_file):
            with open(annotated_file, "rb") as vf:
                vid_data = vf.read()
            st.download_button(
                "📥 Download analyzed video",
                vid_data,
                "gridguard_analyzed_video.mp4",
                "video/mp4",
                use_container_width=True,
            )
        else:
            st.button("Analyzed video ready", disabled=True, use_container_width=True)

    with act_col3:
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📄 Download traffic report (CSV)",
            csv_data,
            "gridguard_traffic_report.csv",
            "text/csv",
            use_container_width=True,
        )
