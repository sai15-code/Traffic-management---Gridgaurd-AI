"""GRIDGUARD AI: Smart Route Recommendation & Road Network Comparison Engine.

Evaluates traffic conditions across available road corridors, computes normalized
congestion scores and estimated travel times, and recommends the optimal route
with the lowest congestion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class RoadSegment:
    """Represents a discrete road segment or alternative corridor."""
    segment_id: str
    name: str
    start_node: str
    end_node: str
    data_source: str
    is_observed_video: bool
    length_km: float = 3.0
    speed_limit_kmh: float = 60.0
    total_detections: int = 0
    frames_observed: int = 1
    avg_vehicles_per_frame: float = 0.0
    peak_vehicles_in_frame: int = 0
    queue_detections: int = 0
    avg_queue_per_frame: float = 0.0
    occupancy_estimate: float = 0.0
    congestion_score: float = 0.0
    risk_level: str = "LOW"
    density_indicator: str = "Light Flow"
    estimated_speed_kmh: Optional[float] = None
    estimated_travel_time_min: Optional[float] = None
    route_cost: float = 0.0
    recommendation_status: str = "VIABLE ALTERNATIVE"
    selection_notes: str = ""
    coordinates: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class RouteRecommendationResult:
    """Complete route recommendation result containing all compared options."""
    recommended_segment_id: str
    recommended_segment_name: str
    recommendation_headline: str
    recommendation_reason: str
    all_congested: bool
    has_alternatives: bool
    segments: List[RoadSegment]
    comparison_df: pd.DataFrame
    graph: nx.DiGraph
    cost_weights: Dict[str, float]


def compute_segment_metrics(
    avg_vehicles: float,
    peak_vehicles: int,
    avg_queue: float,
    avg_occupancy: float,
    length_km: float = 3.0,
    speed_limit_kmh: float = 60.0,
) -> Tuple[float, str, str, float, float, float]:
    """Calculate normalized congestion score, risk level, density label, estimated speed,

    estimated travel time, and composite route cost.
    """
    # 1. Normalize vehicle density (calibrated against arterial capacity: ~14 vehicles/frame max density)
    norm_veh = min(1.0, max(0.0, avg_vehicles / 14.0))

    # 2. Normalize queue density (calibrated against bottleneck ROI capacity: ~7 vehicles in ROI)
    norm_queue = min(1.0, max(0.0, avg_queue / 7.0))

    # 3. Queue occupancy estimate
    norm_occ = min(1.0, max(0.0, avg_occupancy))

    # Heuristic Congestion Score: 40% vehicle density + 40% queue density + 20% occupancy
    congestion_score = round(0.40 * norm_veh + 0.40 * norm_queue + 0.20 * norm_occ, 3)
    congestion_score = min(1.0, max(0.0, congestion_score))

    # Risk level classification
    if congestion_score < 0.35:
        risk_level = "LOW"
        density_indicator = "Light Flow"
    elif congestion_score < 0.68:
        risk_level = "MEDIUM"
        density_indicator = "Moderate Flow"
    elif congestion_score < 0.85:
        risk_level = "HIGH"
        density_indicator = "Heavy Congestion"
    else:
        risk_level = "HIGH"
        density_indicator = "Gridlock / Severe"

    # Travel time estimation based on speed degradation under congestion
    # Free-flow speed degrades non-linearly with congestion
    effective_speed = max(8.0, speed_limit_kmh * max(0.12, (1.0 - 0.72 * (congestion_score ** 1.35))))
    effective_speed = round(effective_speed, 1)

    travel_time_min = round((length_km / effective_speed) * 60.0, 1)

    # Route cost combining congestion, queue severity, and normalized travel delay
    free_flow_time = (length_km / max(20.0, speed_limit_kmh)) * 60.0
    delay_ratio = travel_time_min / max(1.0, free_flow_time)
    norm_delay = min(1.0, max(0.0, (delay_ratio - 1.0) / 2.5))

    route_cost = round(0.50 * congestion_score + 0.30 * norm_queue + 0.20 * norm_delay, 3)

    return congestion_score, risk_level, density_indicator, effective_speed, travel_time_min, route_cost


def evaluate_routes(
    primary_df: Optional[pd.DataFrame] = None,
    primary_name: str = "Road A (Primary Arterial)",
    secondary_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    mode: str = "corridor_baseline",
    custom_baseline_data: Optional[List[Dict[str, Any]]] = None,
) -> RouteRecommendationResult:
    """Core route comparison engine. Evaluates Road A, Road B, Road C, and additional

    corridors using real video observations where available, and explicit configured
    network baselines for alternative corridors.
    """
    segments: List[RoadSegment] = []

    # 1. Process Primary Road (Observed Video Feed)
    if primary_df is not None and not primary_df.empty:
        n_frames = len(primary_df)
        total_veh = int(primary_df["vehicle_count"].sum())
        avg_veh = float(primary_df["vehicle_count"].mean())
        peak_veh = int(primary_df["vehicle_count"].max())
        total_queue = int(primary_df["roi_vehicle_count"].sum())
        avg_queue = float(primary_df["roi_vehicle_count"].mean())
        avg_occ = float(primary_df["occupancy_estimate"].mean())

        cong, risk, density, speed, t_min, cost = compute_segment_metrics(
            avg_vehicles=avg_veh,
            peak_vehicles=peak_veh,
            avg_queue=avg_queue,
            avg_occupancy=avg_occ,
            length_km=3.2,
            speed_limit_kmh=60.0,
        )

        road_a = RoadSegment(
            segment_id="ROAD_A",
            name=primary_name,
            start_node="Origin Junction (W)",
            end_node="Destination Hub (E)",
            data_source="Observed Video Feed",
            is_observed_video=True,
            length_km=3.2,
            speed_limit_kmh=60.0,
            total_detections=total_veh,
            frames_observed=n_frames,
            avg_vehicles_per_frame=round(avg_veh, 2),
            peak_vehicles_in_frame=peak_veh,
            queue_detections=total_queue,
            avg_queue_per_frame=round(avg_queue, 2),
            occupancy_estimate=round(avg_occ, 2),
            congestion_score=cong,
            risk_level=risk,
            density_indicator=density,
            estimated_speed_kmh=speed,
            estimated_travel_time_min=t_min,
            route_cost=cost,
            selection_notes="Directly analyzed from uploaded video footage.",
            coordinates={"start": (0.0, 0.5), "end": (1.0, 0.5)},
        )
        segments.append(road_a)

    # 2. Process Secondary Uploaded Videos (if multi-video mode)
    if secondary_dfs:
        road_labels = ["ROAD_B", "ROAD_C", "ROAD_D"]
        for idx, (road_name, sec_df) in enumerate(secondary_dfs.items()):
            if sec_df is None or sec_df.empty:
                continue
            r_id = road_labels[min(idx, len(road_labels) - 1)]
            n_f = len(sec_df)
            t_v = int(sec_df["vehicle_count"].sum())
            a_v = float(sec_df["vehicle_count"].mean())
            p_v = int(sec_df["vehicle_count"].max())
            t_q = int(sec_df["roi_vehicle_count"].sum())
            a_q = float(sec_df["roi_vehicle_count"].mean())
            a_o = float(sec_df["occupancy_estimate"].mean())

            # Calibrated length/speed based on index
            len_km = 4.2 if idx == 0 else 3.8
            spd_lim = 80.0 if idx == 0 else 50.0

            cong, risk, density, speed, t_min, cost = compute_segment_metrics(
                avg_vehicles=a_v,
                peak_vehicles=p_v,
                avg_queue=a_q,
                avg_occupancy=a_o,
                length_km=len_km,
                speed_limit_kmh=spd_lim,
            )

            sec_seg = RoadSegment(
                segment_id=r_id,
                name=road_name,
                start_node="Origin Junction (W)",
                end_node="Destination Hub (E)",
                data_source="Secondary Video Feed",
                is_observed_video=True,
                length_km=len_km,
                speed_limit_kmh=spd_lim,
                total_detections=t_v,
                frames_observed=n_f,
                avg_vehicles_per_frame=round(a_v, 2),
                peak_vehicles_in_frame=p_v,
                queue_detections=t_q,
                avg_queue_per_frame=round(a_q, 2),
                occupancy_estimate=round(a_o, 2),
                congestion_score=cong,
                risk_level=risk,
                density_indicator=density,
                estimated_speed_kmh=speed,
                estimated_travel_time_min=t_min,
                route_cost=cost,
                selection_notes="Directly analyzed from secondary video feed.",
                coordinates={"start": (0.0, 0.5), "end": (1.0, 0.5)},
            )
            segments.append(sec_seg)

    # 3. Connected Corridor Network Alternatives (Configured Baseline)
    # If mode is 'corridor_baseline' and we only have Road A, add Road B and Road C
    # using explicitly configured regional corridor baselines
    if mode == "corridor_baseline" and (not secondary_dfs or len(secondary_dfs) == 0):
        # Road B: Northern Ring / Outer Bypass (Longer distance, higher speed limit, free flow)
        b_avg_veh = 3.6
        b_peak_veh = 6
        b_avg_q = 0.8
        b_avg_occ = 0.12
        b_len = 4.6
        b_spd = 80.0

        b_cong, b_risk, b_dens, b_speed, b_tmin, b_cost = compute_segment_metrics(
            avg_vehicles=b_avg_veh,
            peak_vehicles=b_peak_veh,
            avg_queue=b_avg_q,
            avg_occupancy=b_avg_occ,
            length_km=b_len,
            speed_limit_kmh=b_spd,
        )

        road_b = RoadSegment(
            segment_id="ROAD_B",
            name="Road B (Northern Bypass)",
            start_node="Origin Junction (W)",
            end_node="Destination Hub (E)",
            data_source="Configured Network Baseline",
            is_observed_video=False,
            length_km=b_len,
            speed_limit_kmh=b_spd,
            total_detections=int(b_avg_veh * (segments[0].frames_observed if segments else 100)),
            frames_observed=segments[0].frames_observed if segments else 100,
            avg_vehicles_per_frame=b_avg_veh,
            peak_vehicles_in_frame=b_peak_veh,
            queue_detections=int(b_avg_q * (segments[0].frames_observed if segments else 100)),
            avg_queue_per_frame=b_avg_q,
            occupancy_estimate=b_avg_occ,
            congestion_score=b_cong,
            risk_level=b_risk,
            density_indicator=b_dens,
            estimated_speed_kmh=b_speed,
            estimated_travel_time_min=b_tmin,
            route_cost=b_cost,
            selection_notes="Calibrated regional corridor: bypass route avoiding central signals.",
            coordinates={"start": (0.0, 0.5), "waypoint": (0.5, 0.9), "end": (1.0, 0.5)},
        )
        segments.append(road_b)

        # Road C: Eastern Commercial Link (Moderate congestion, mixed retail traffic)
        c_avg_veh = 9.8
        c_peak_veh = 16
        c_avg_q = 4.2
        c_avg_occ = 0.45
        c_len = 3.6
        c_spd = 50.0

        c_cong, c_risk, c_dens, c_speed, c_tmin, c_cost = compute_segment_metrics(
            avg_vehicles=c_avg_veh,
            peak_vehicles=c_peak_veh,
            avg_queue=c_avg_q,
            avg_occupancy=c_avg_occ,
            length_km=c_len,
            speed_limit_kmh=c_spd,
        )

        road_c = RoadSegment(
            segment_id="ROAD_C",
            name="Road C (Eastern Commercial Link)",
            start_node="Origin Junction (W)",
            end_node="Destination Hub (E)",
            data_source="Configured Network Baseline",
            is_observed_video=False,
            length_km=c_len,
            speed_limit_kmh=c_spd,
            total_detections=int(c_avg_veh * (segments[0].frames_observed if segments else 100)),
            frames_observed=segments[0].frames_observed if segments else 100,
            avg_vehicles_per_frame=c_avg_veh,
            peak_vehicles_in_frame=c_peak_veh,
            queue_detections=int(c_avg_q * (segments[0].frames_observed if segments else 100)),
            avg_queue_per_frame=c_avg_q,
            occupancy_estimate=c_avg_occ,
            congestion_score=c_cong,
            risk_level=c_risk,
            density_indicator=c_dens,
            estimated_speed_kmh=c_speed,
            estimated_travel_time_min=c_tmin,
            route_cost=c_cost,
            selection_notes="Calibrated regional corridor: urban commercial arterial.",
            coordinates={"start": (0.0, 0.5), "waypoint": (0.5, 0.1), "end": (1.0, 0.5)},
        )
        segments.append(road_c)

    # 4. Single Road Only Mode (No alternative data configured)
    has_alternatives = len(segments) > 1

    if not segments:
        # Fallback empty result
        return RouteRecommendationResult(
            recommended_segment_id="",
            recommended_segment_name="No Data",
            recommendation_headline="No traffic data available",
            recommendation_reason="Please upload a traffic video to begin analysis.",
            all_congested=False,
            has_alternatives=False,
            segments=[],
            comparison_df=pd.DataFrame(),
            graph=nx.DiGraph(),
            cost_weights={"congestion": 0.50, "queue": 0.30, "delay": 0.20},
        )

    # 5. Route Selection Logic
    # Sort segments by route cost ascending
    sorted_segments = sorted(segments, key=lambda s: s.route_cost)
    best_segment = sorted_segments[0]

    # Check if all routes are congested (score >= 0.65)
    all_congested = all(s.congestion_score >= 0.65 for s in segments) if has_alternatives else (best_segment.congestion_score >= 0.65)

    # Assign recommendation status
    for s in segments:
        if s.segment_id == best_segment.segment_id:
            s.recommendation_status = "BEST ROUTE"
        elif s.congestion_score < 0.40:
            s.recommendation_status = "VIABLE ALTERNATIVE"
        elif s.congestion_score < 0.70:
            s.recommendation_status = "MODERATE DELAY"
        else:
            s.recommendation_status = "HEAVILY CONGESTED - AVOID"

    # Formulate Headline and Reasoning
    if not has_alternatives:
        headline = f"Analysis Complete — {best_segment.name}"
        reason = (
            f"Observed congestion score is {best_segment.congestion_score:.2f} ({best_segment.density_indicator}) "
            f"with an average of {best_segment.avg_vehicles_per_frame:.1f} vehicles/frame. "
            f"To compare connected alternative routes (Road B, Road C), enable Connected Corridor Comparison."
        )
    elif all_congested:
        headline = f"HEAVY TRAFFIC ACROSS ALL CORRIDORS — {best_segment.name} is Least Congested"
        reason = (
            f"All compared routes are experiencing elevated traffic (congestion scores >= 0.65). "
            f"{best_segment.name} offers the lowest relative impact with a congestion score of {best_segment.congestion_score:.2f} "
            f"and an estimated transit time of {best_segment.estimated_travel_time_min} mins."
        )
    else:
        # Normal best route selection
        headline = f"BEST ROUTE FOUND — {best_segment.name}"
        # Compute delta with other segments
        other_congs = [s.congestion_score for s in segments if s.segment_id != best_segment.segment_id]
        avg_other_cong = sum(other_congs) / len(other_congs) if other_congs else best_segment.congestion_score
        reduction_pct = max(0, int(round((1.0 - best_segment.congestion_score / max(0.01, avg_other_cong)) * 100)))

        reason = (
            f"{best_segment.name} is recommended with the lowest observed congestion score of {best_segment.congestion_score:.2f} "
            f"({best_segment.density_indicator}), representing a {reduction_pct}% traffic reduction compared to alternatives. "
            f"Observed queue density is {best_segment.avg_queue_per_frame:.1f} vehicles/frame, yielding the fastest estimated "
            f"travel time of {best_segment.estimated_travel_time_min} mins."
        )

    # 6. Build Comparison DataFrame
    df_rows = []
    for s in segments:
        df_rows.append({
            "Road ID": s.segment_id,
            "Road Name": s.name,
            "Status": s.recommendation_status,
            "Congestion Score": s.congestion_score,
            "Risk Category": s.risk_level,
            "Density Level": s.density_indicator,
            "Avg Vehicles / Frame": s.avg_vehicles_per_frame,
            "Queue Count / Frame": s.avg_queue_per_frame,
            "Estimated Travel Time": f"{s.estimated_travel_time_min} min" if s.estimated_travel_time_min else "N/A",
            "Speed Limit": f"{int(s.speed_limit_kmh)} km/h",
            "Distance": f"{s.length_km} km",
            "Data Source": s.data_source,
        })
    comp_df = pd.DataFrame(df_rows)

    # 7. Build NetworkX Graph
    g = nx.DiGraph()
    origin = "Origin Junction (W)"
    dest = "Destination Hub (E)"
    g.add_node(origin, pos=(0.0, 0.5), label="Origin")
    g.add_node(dest, pos=(1.0, 0.5), label="Destination")

    for s in segments:
        # Create intermediate waypoint if bypass or curved
        if s.segment_id == "ROAD_A":
            mid = "Jct A (Central Arterial)"
            g.add_node(mid, pos=(0.5, 0.5), label="Mid Arterial")
            g.add_edge(origin, mid, road_id=s.segment_id, name=s.name, congestion=s.congestion_score, recommended=(s.segment_id == best_segment.segment_id))
            g.add_edge(mid, dest, road_id=s.segment_id, name=s.name, congestion=s.congestion_score, recommended=(s.segment_id == best_segment.segment_id))
        elif s.segment_id == "ROAD_B":
            mid = "Jct B (North Bypass)"
            g.add_node(mid, pos=(0.5, 0.9), label="North Bypass")
            g.add_edge(origin, mid, road_id=s.segment_id, name=s.name, congestion=s.congestion_score, recommended=(s.segment_id == best_segment.segment_id))
            g.add_edge(mid, dest, road_id=s.segment_id, name=s.name, congestion=s.congestion_score, recommended=(s.segment_id == best_segment.segment_id))
        elif s.segment_id == "ROAD_C":
            mid = "Jct C (East Link)"
            g.add_node(mid, pos=(0.5, 0.1), label="East Link")
            g.add_edge(origin, mid, road_id=s.segment_id, name=s.name, congestion=s.congestion_score, recommended=(s.segment_id == best_segment.segment_id))
            g.add_edge(mid, dest, road_id=s.segment_id, name=s.name, congestion=s.congestion_score, recommended=(s.segment_id == best_segment.segment_id))

    return RouteRecommendationResult(
        recommended_segment_id=best_segment.segment_id,
        recommended_segment_name=best_segment.name,
        recommendation_headline=headline,
        recommendation_reason=reason,
        all_congested=all_congested,
        has_alternatives=has_alternatives,
        segments=segments,
        comparison_df=comp_df,
        graph=g,
        cost_weights={"congestion": 0.50, "queue": 0.30, "delay": 0.20},
    )


def build_network_graph_figure(
    graph: nx.DiGraph,
    segments: List[RoadSegment],
    recommended_id: str,
    height: int = 380,
) -> go.Figure:
    """Builds a clean, light-themed schematic road-network graph visualization using Plotly."""
    fig = go.Figure()

    # Schematic coordinates for nodes
    pos = {
        "Origin Junction (W)": (0.06, 0.5),
        "Destination Hub (E)": (0.94, 0.5),
        "Jct A (Central Arterial)": (0.50, 0.5),
        "Jct B (North Bypass)": (0.50, 0.88),
        "Jct C (East Link)": (0.50, 0.12),
    }

    # Color mapping matching light design system
    def get_color(score: float, is_rec: bool) -> str:
        if is_rec:
            return "#16865B"  # Fresh success green for recommended route
        if score < 0.35:
            return "#16865B"  # Low congestion green
        if score < 0.68:
            return "#D99A27"  # Moderate congestion amber
        return "#DC5656"      # High congestion red

    # Draw edges for each segment
    for s in segments:
        is_rec = (s.segment_id == recommended_id)
        color = get_color(s.congestion_score, is_rec)
        line_width = 5.5 if is_rec else 3.2

        if s.segment_id == "ROAD_A":
            p0 = pos["Origin Junction (W)"]
            p1 = pos["Jct A (Central Arterial)"]
            p2 = pos["Destination Hub (E)"]
            xs = [p0[0], p1[0], p2[0]]
            ys = [p0[1], p1[1], p2[1]]
        elif s.segment_id == "ROAD_B":
            p0 = pos["Origin Junction (W)"]
            p1 = pos["Jct B (North Bypass)"]
            p2 = pos["Destination Hub (E)"]
            xs = [p0[0], p1[0], p2[0]]
            ys = [p0[1], p1[1], p2[1]]
        elif s.segment_id == "ROAD_C":
            p0 = pos["Origin Junction (W)"]
            p1 = pos["Jct C (East Link)"]
            p2 = pos["Destination Hub (E)"]
            xs = [p0[0], p1[0], p2[0]]
            ys = [p0[1], p1[1], p2[1]]
        else:
            xs = [0.06, 0.94]
            ys = [0.5, 0.5]

        # Friendly hover card
        hover_text = (
            f"<b>{s.name}</b><br>"
            f"Status: {s.recommendation_status}<br>"
            f"Congestion Level: {int(s.congestion_score * 100)}% ({s.density_indicator})<br>"
            f"Average Vehicles: {s.avg_vehicles_per_frame:.1f} per frame<br>"
            f"Estimated Travel Time: {s.estimated_travel_time_min} mins<br>"
            f"Data Source: {s.data_source}"
        )

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(color=color, width=line_width),
            marker=dict(size=7, color=color),
            hoverinfo="text",
            text=hover_text,
            name=s.name,
            showlegend=True,
        ))

        # Add road label badge near the midpoint
        badge_text = f"<b>{s.segment_id}</b> ({int(s.congestion_score*100)}%)"
        if is_rec:
            badge_text = f"⭐ <b>{s.segment_id} [BEST ROUTE]</b>"

        fig.add_annotation(
            x=xs[1], y=ys[1],
            text=badge_text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor=color,
            ax=0,
            ay=-28 if ys[1] >= 0.5 else 28,
            bgcolor="#FFFFFF",
            bordercolor=color,
            borderwidth=1.5,
            borderpad=4,
            font=dict(color="#172B45", size=11, family="Segoe UI, sans-serif"),
        )

    # Draw Origin and Destination node markers
    node_x = [pos["Origin Junction (W)"][0], pos["Destination Hub (E)"][0]]
    node_y = [pos["Origin Junction (W)"][1], pos["Destination Hub (E)"][1]]
    node_text = ["<b>ORIGIN</b><br>Trip start point", "<b>DESTINATION</b><br>Trip end point"]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=22, color="#2475C8", symbol="circle", line=dict(color="#FFFFFF", width=3)),
        text=["START", "END"],
        textposition=["top center", "top center"],
        textfont=dict(color="#172B45", size=11, family="Segoe UI, sans-serif"),
        hoverinfo="text",
        hovertext=node_text,
        name="Network Endpoints",
        showlegend=False,
    ))

    # Clean light SaaS styling
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=height,
        margin=dict(l=10, r=10, t=32, b=25),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.04, 1.04]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.08, 1.08]),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#DCE4EF", font_color="#172B45"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.14,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DCE4EF",
            borderwidth=1,
            font=dict(size=11, color="#64748B"),
        ),
        title=dict(
            text="Road Network & Route Guidance",
            font=dict(size=13, color="#172B45", family="Segoe UI, sans-serif"),
            x=0.01,
        ),
    )
    return fig


def build_route_comparison_bar_chart(
    segments: List[RoadSegment],
    recommended_id: str,
    height: int = 340,
) -> go.Figure:
    """Creates a clean, light-themed comparative bar chart across road options."""
    road_names = [s.name for s in segments]
    congestion_pcts = [round(s.congestion_score * 100, 1) for s in segments]
    avg_vehs = [s.avg_vehicles_per_frame for s in segments]
    avg_queues = [s.avg_queue_per_frame for s in segments]
    travel_times = [s.estimated_travel_time_min or 0.0 for s in segments]

    # Bar colors in light theme
    bar_colors = [
        "#16865B" if s.segment_id == recommended_id else (
            "#DC5656" if s.congestion_score >= 0.68 else "#D99A27" if s.congestion_score >= 0.35 else "#2475C8"
        )
        for s in segments
    ]

    fig = go.Figure()

    # Congestion Level Bars
    fig.add_trace(go.Bar(
        name="Congestion Level (%)",
        x=road_names,
        y=congestion_pcts,
        marker=dict(
            color=bar_colors,
            line=dict(color="#FFFFFF", width=[2.5 if s.segment_id == recommended_id else 0.5 for s in segments]),
        ),
        text=[f"{val}%" + (" ⭐ [RECOMMENDED]" if s.segment_id == recommended_id else "") for val, s in zip(congestion_pcts, segments)],
        textposition="auto",
        customdata=list(zip(avg_vehs, avg_queues, travel_times, [s.data_source for s in segments])),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Congestion Level: %{y}%<br>"
            "Vehicles / Frame: %{customdata[0]:.1f}<br>"
            "Vehicles in Congested Area: %{customdata[1]:.1f}<br>"
            "Estimated Travel Time: %{customdata[2]} mins<br>"
            "Source: %{customdata[3]}<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=10, r=10, t=32, b=25),
        yaxis=dict(
            title="Congestion Level (%)",
            gridcolor="#E2E8F0",
            range=[0, 105],
            title_font=dict(color="#64748B", size=11),
            tickfont=dict(color="#64748B"),
        ),
        xaxis=dict(
            gridcolor="#E2E8F0",
            tickfont=dict(color="#172B45", size=11),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#DCE4EF", font_color="#172B45"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#64748B"),
        ),
        title=dict(
            text="Road Congestion Comparison (Recommended Road Highlighted)",
            font=dict(size=13, color="#172B45", family="Segoe UI, sans-serif"),
            x=0.01,
        ),
    )
    return fig
