"""YOLO vehicle detection with per-frame metrics and annotated video output."""
from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.risk_engine import calculate_risk
from src.traffic_metrics import calculate_metrics

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


def analyze_video(
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: int = 300,
    image_size: int = 320,
    conf: float = 0.25,
    model_name: str = "yolov8n.pt",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    frame_callback: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None,
    preview_every: int = 1,
    model: Optional[YOLO] = None,
    segment_rois: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Analyze up to ``max_frames`` sequentially and return frame-level detections.

    ``model`` may be supplied by a long-lived UI cache. The CLI continues to load
    the model from ``model_name`` when no model object is passed.

    Supports truthful multistage progress notification, detection confidence tracking,
    and optional spatial sub-segment partitioning for multi-lane/multi-road videos.
    """
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer.")
    if image_size not in (320, 480, 640):
        raise ValueError("image_size must be one of 320, 480, or 640.")
    if not 0.0 < float(conf) < 1.0:
        raise ValueError("conf must be greater than 0 and less than 1.")
    if preview_every < 1:
        raise ValueError("preview_every must be at least 1.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Stage 1: Reading video frames
    if progress_callback:
        progress_callback(0.02, "Stage 1/8: Reading video frames and extracting metadata...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open this video. Try MP4, AVI, MOV, or MKV.")

    # Stage 2: Extracting frame information
    if progress_callback:
        progress_callback(0.05, "Stage 2/8: Extracting frame information and dimensions...")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    target = min(max_frames, total) if total > 0 else max_frames

    # Stage 3: Initializing YOLO Model
    if progress_callback:
        progress_callback(0.08, "Stage 3/8: Preparing YOLO neural network for vehicle detection...")

    model = model or YOLO(model_name)

    writer = None
    if output_path and width > 0 and height > 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None

    rows: List[Dict[str, Any]] = []
    previous_roi_count: Optional[int] = None
    index = 0

    try:
        while index < target:
            ok, frame = cap.read()
            if not ok:
                break

            raw_frame_copy = frame.copy()

            # YOLO inference
            prediction = model.predict(frame, imgsz=image_size, conf=conf, verbose=False)[0]

            centres: List[Tuple[int, int]] = []
            boxes: List[Tuple[int, int, int, int, str, float]] = []
            class_counts = {name: 0 for name in sorted(VEHICLE_CLASSES)}
            confidences: List[float] = []

            # Stage 4: Classifying detected vehicles
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    class_id = int(box.cls[0].item())
                    label = prediction.names.get(class_id, str(class_id))
                    if label not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
                    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    centres.append((cx, cy))
                    score = float(box.conf[0].item())
                    confidences.append(score)
                    boxes.append((x1, y1, x2, y2, label, score))
                    class_counts[label] += 1

            # Stage 5: Measuring queue-zone activity
            metrics = calculate_metrics(centres, width, height, previous_roi_count=previous_roi_count)
            previous_roi_count = metrics.roi_vehicle_count

            # Measure optional spatial sub-segments if configured
            sub_counts: Dict[str, int] = {}
            if segment_rois:
                for seg_name, seg_roi in segment_rois.items():
                    sx1, sy1, sx2, sy2 = seg_roi
                    sc = sum(
                        int(sx1 * width) <= cx <= int(sx2 * width) and int(sy1 * height) <= cy <= int(sy2 * height)
                        for cx, cy in centres
                    )
                    sub_counts[f"{seg_name}_count"] = sc

            risk = calculate_risk(
                min(metrics.roi_vehicle_count / 10.0, 1.0),
                metrics.occupancy,
                metrics.queue_growth,
                metrics.downstream_congestion,
            )

            # Build Annotated Frame
            annotated = frame.copy()

            # Main Queue Zone ROI (Amber outline)
            roi = (int(0.15 * width), int(0.45 * height), int(0.85 * width), int(0.98 * height))
            cv2.rectangle(annotated, (roi[0], roi[1]), (roi[2], roi[3]), (255, 190, 40), 2)
            cv2.putText(
                annotated, "QUEUE ANALYSIS ZONE", (roi[0] + 6, roi[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 190, 40), 1, cv2.LINE_AA
            )

            # Draw spatial sub-segments if provided
            if segment_rois:
                colors = [(57, 216, 242), (102, 230, 193), (243, 184, 95)]
                for i, (seg_name, seg_roi) in enumerate(segment_rois.items()):
                    c = colors[i % len(colors)]
                    sx1, sy1, sx2, sy2 = int(seg_roi[0] * width), int(seg_roi[1] * height), int(seg_roi[2] * width), int(seg_roi[3] * height)
                    cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), c, 2)
                    cv2.putText(annotated, f"SEGMENT: {seg_name.upper()}", (sx1 + 6, sy1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

            # Draw Vehicle Bounding Boxes
            for x1, y1, x2, y2, label, score in boxes:
                # Green box outline
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 230, 145), 2)
                caption = f"{label.upper()} {score:.0%}"
                tag_w = min(140, max(80, (x2 - x1)))
                cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (min(width - 1, x1 + tag_w), y1), (16, 24, 39), -1)
                cv2.putText(
                    annotated, caption, (x1 + 4, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
                )

            # Top telemetry banner
            avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
            banner = f"GRIDGUARD AI | FRAME {index+1} | DETECTIONS: {metrics.vehicle_count} | QUEUE: {metrics.roi_vehicle_count} | {risk.level} RISK ({risk.score:.2f})"
            cv2.rectangle(annotated, (0, 0), (width, 36), (16, 24, 39), -1)
            cv2.putText(
                annotated, banner[:120], (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
            )

            # Format active classes list
            classes_present = [f"{cls.capitalize()} ({count})" for cls, count in class_counts.items() if count > 0]
            classes_str = ", ".join(classes_present) if classes_present else "None"

            row = {
                "frame": index + 1,
                "time_seconds": round(index / fps, 3),
                "vehicle_count": metrics.vehicle_count,
                "roi_vehicle_count": metrics.roi_vehicle_count,
                "occupancy_estimate": metrics.occupancy,
                "queue_growth": metrics.queue_growth,
                "risk_score": risk.score,
                "risk_level": risk.level,
                "risk_explanation": risk.explanation,
                "car_count": class_counts["car"],
                "motorcycle_count": class_counts["motorcycle"],
                "bus_count": class_counts["bus"],
                "truck_count": class_counts["truck"],
                "avg_confidence": avg_conf,
                "classes_detected": classes_str,
                "raw_frame": raw_frame_copy,
            }
            row.update(sub_counts)
            rows.append(row)

            if writer:
                writer.write(annotated)

            if frame_callback and (index % max(1, preview_every) == 0):
                frame_callback(annotated, row)

            index += 1

            if progress_callback:
                frac = min(index / max(1, target), 1.0)
                # Map progress across Stages 3-5 (0.10 to 0.85)
                scaled_pct = 0.10 + 0.75 * frac
                curr_stage = f"Stage 3-5/8: Detecting vehicles & measuring queue zones (frame {index}/{target})"
                progress_callback(scaled_pct, curr_stage)

    finally:
        cap.release()
        if writer:
            writer.release()

    if not rows:
        raise RuntimeError("No frames were processed. Check that the video contains readable frames.")

    # Stage 6: Aggregating traffic observations
    if progress_callback:
        progress_callback(0.88, "Stage 6/8: Aggregating traffic observations across analyzed frames...")

    result_df = pd.DataFrame(rows)

    # Clean raw_frame from returned DataFrame so memory stays small, but keep metadata
    df_for_return = result_df.drop(columns=["raw_frame"], errors="ignore")

    saved_video = output_path if writer is not None and output_path and os.path.isfile(output_path) else None
    return df_for_return, saved_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/annotated.mp4")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    df, output = analyze_video(args.video, args.output, args.max_frames, args.image_size, args.conf)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/traffic_metrics.csv", index=False)
    print(f"Processed {len(df)} frames. Metrics saved to results/traffic_metrics.csv")
    if output:
        print(f"Annotated video: {output}")


if __name__ == "__main__":
    main()
