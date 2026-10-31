"""Network graph modeling and bottleneck analysis for road corridors."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd


def build_demo_road_graph() -> nx.DiGraph:
    """Builds an illustrative 5-node directed traffic graph."""
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C", "D", "E"])
    graph.add_edges_from([("A", "B"), ("D", "B"), ("E", "B"), ("B", "C"), ("C", "A")])
    return graph


def rank_bottlenecks(
    congestion_by_node: Dict[str, float],
    graph: Optional[nx.DiGraph] = None,
) -> List[Dict[str, Any]]:
    """Ranks nodes by bottleneck score combining local congestion and downstream pressure."""
    graph = graph or build_demo_road_graph()
    rows = []
    for node in graph.nodes:
        own = float(congestion_by_node.get(node, 0.0))
        nxt = list(graph.successors(node))
        downstream = sum(float(congestion_by_node.get(n, 0.0)) for n in nxt) / len(nxt) if nxt else 0.0
        rows.append({
            "node": node,
            "own_congestion": round(own, 3),
            "downstream_congestion": round(downstream, 3),
            "bottleneck_score": round(0.65 * own + 0.35 * downstream, 3),
        })
    return sorted(rows, key=lambda row: row["bottleneck_score"], reverse=True)
