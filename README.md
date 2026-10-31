# GRIDGUARD AI — AI-Powered Traffic Intelligence and Smart Route Recommendation System

GRIDGUARD AI analyzes road traffic conditions using computer vision and recommends the best available road with the lowest congestion so drivers can avoid traffic delays.

---

## Key Highlights

- **Smart Route Recommendation Engine (`src/route_engine.py`):** Automatically compares multiple road corridors (Road A, Road B, Road C) using normalized congestion scoring and estimated travel times to identify and recommend the optimal route.
- **Truthful 8-Stage Processing Pipeline:** Visibly guides the user through real-time processing:
  1. Reading video frames
  2. Extracting frame information
  3. Detecting vehicles using YOLO
  4. Classifying detected vehicles (cars, motorcycles, buses, trucks)
  5. Measuring queue-zone activity
  6. Aggregating traffic observations
  7. Comparing available road options
  8. Generating the route recommendation
- **Interactive Schematic Road Network Graph:** Plotly-powered NetworkX graph depicting connected junctions and road corridors color-coded by congestion (green, amber, red) with the recommended route highlighted.
- **Corridor Comparison Chart:** Side-by-side comparative bar chart visualizing congestion scores, vehicle counts, and queue densities with hover telemetry.
- **Multi-Modal Road Network Support:**
  - *Connected Regional Corridor Network:* Compares observed primary road against calibrated regional alternative corridors.
  - *Spatial Sub-Zone Partitioning:* Partitions a single intersection camera into distinct diverging segments.
  - *Multi-Feed Ingestion:* Ingests independent video clips for each alternative road.
  - *Isolated Road Analysis:* Honestly reports single-road observation when no alternative network data is configured.
- **Scientific Honesty & Data Reliability:** Clearly distinguishes real observed video data from configured regional baselines and simulations. Does not fabricate speeds, GPS tracks, or city-wide coverage from a single camera.
- **Multi-Format Exports:** Download frame-by-frame analysis CSV, annotated video MP4, and comprehensive JSON route recommendation reports.

---

## Quick Start on Windows / VS Code

1. Activate your Python environment or install dependencies:
```powershell
python -m pip install -r requirements.txt
```

2. Run the Streamlit application:
```powershell
python -m streamlit run dashboard/app.py
```

3. Open the local URL displayed by Streamlit (typically `http://localhost:8501`).

---

## User Workflow

1. **Video Ingestion:** Upload an MP4, AVI, MOV, or MKV traffic video, or click "Use Bundled Sample Video".
2. **Configure Corridor Network:** In advanced settings, select between Connected Corridor Network, Spatial Sub-Zones, or Isolated Road analysis.
3. **Analyze Traffic:** Click **ANALYZE TRAFFIC & RECOMMEND BEST ROUTE**. Watch the live dual-preview (original vs YOLO-annotated) and live telemetry.
4. **Discover Best Route:** Inspect the prominent **BEST ROUTE FOUND** recommendation card with detailed empirical justifications and transit time estimates.
5. **Explore Intelligence:** Switch tabs to inspect the interactive Network Graph, Corridor Comparison Chart, Class Distribution, and Video Output.
6. **Export Data:** Download CSV frame metrics, JSON route recommendation summary, or MP4 annotated video.

---

## Project Architecture

```
GRIDGUARD-AI-FINAL/
├── dashboard/
│   └── app.py                     # Premium Streamlit UI & Route Guidance Studio
├── src/
│   ├── route_engine.py            # Route Recommendation Engine & Plotly Network Visualizer
│   ├── vehicle_detection.py       # YOLOv8 Detection, Queue ROI & Telemetry Pipeline
│   ├── traffic_metrics.py         # Frame metrics, spatial counts, queue occupancy
│   ├── risk_engine.py             # Heuristic risk calculation & classification
│   ├── network_analysis.py        # NetworkX corridor graph & bottleneck ranking
│   ├── intervention_engine.py     # Action selection & penalty evaluations
│   └── simulation_controller.py   # Optional TraCI / SUMO simulation connector
├── tests/
│   └── test_pipeline.py           # Automated end-to-end verification suite
├── results/                       # Generated annotated videos & exported metrics
├── yolov8n.pt                     # Pre-trained YOLOv8 nano model weights
├── requirements.txt
└── README.md
```

---

## Data Reliability & Limitations

- Uploaded video analysis processes sample clips; it is not connected to a live municipal CCTV network or vehicle GPS transponders.
- Detections represent observed instances per frame and are normalized across frame counts to prevent sample duration distortion.
- Congestion indices and risk ratings are heuristic measures intended for situational comparison, not certified safety or emergency navigation decisions.
- SUMO simulation utilities remain separate tools for synthetic traffic grid experiments.
