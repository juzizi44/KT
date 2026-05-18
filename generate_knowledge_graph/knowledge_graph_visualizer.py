"""Visualize the generated knowledge graph as a static plot.

Ported from kt_llm/src/generate_knowledge_graph/knowledge_graph_visualizer.py
with adaptations for the Explainable-Few-shot-Knowledge-Tracing project.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager


# =========================
# 中文字体支持
# =========================
FONT_PATH = Path(__file__).resolve().parent / "NotoSansCJKsc-Regular.otf"
CH_FONT = None

if FONT_PATH.exists():
    CH_FONT = font_manager.FontProperties(fname=str(FONT_PATH))
else:
    # Try to use system default font or fallback
    try:
        CH_FONT = font_manager.FontProperties(family="sans-serif")
    except Exception:
        CH_FONT = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH_PATH = BASE_DIR.parent / "datasets" / "sparse" / "FrcSub" / "knowledge_graph.json"
DEFAULT_OUTPUT_PATH = BASE_DIR.parent / "results" / "knowledge_graph_visualization.png"
MERGED_DEFAULT_MIN_WEIGHT = 0.02
LABEL_NODE_LIMIT = 400
SPIRAL_BASE_GAP = 0.32
DEFAULT_MAX_EDGES_PER_NODE = 8

ConceptDict = Dict[str, Dict[str, object]]


# =========================
# 数据结构
# =========================
@dataclass(frozen=True)
class NodeViz:
    concept_id: str
    name: str
    accuracy: float
    total_attempts: int
    relative_attempts: float


@dataclass(frozen=True)
class EdgeViz:
    source: str
    target: str
    weight: float
    evidence: str | None = None


# =========================
# 数据加载与筛选
# =========================
def load_graph(graph_path: Path) -> Dict:
    if not graph_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found at {graph_path}")

    with graph_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _focus_neighbors(
    concepts: ConceptDict, focus_id: str, min_edge_weight: float
) -> List[str]:
    """Collect neighbors connected to focus_id via merge_relations (in or out)."""

    scores: Dict[str, float] = {}
    focus_node = concepts.get(focus_id, {})

    # Outgoing neighbors
    for dst, payload in (focus_node.get("merge_relations") or {}).items():
        try:
            weight = float(payload.get("final_score", 0.0))
        except (TypeError, ValueError):
            continue
        if weight < min_edge_weight:
            continue
        scores[str(dst)] = max(scores.get(str(dst), 0.0), weight)

    # Incoming neighbors
    for cid, concept in concepts.items():
        if cid == focus_id:
            continue
        payload = (concept.get("merge_relations") or {}).get(focus_id)
        if not payload:
            continue
        try:
            weight = float(payload.get("final_score", 0.0))
        except (TypeError, ValueError):
            continue
        if weight < min_edge_weight:
            continue
        scores[str(cid)] = max(scores.get(str(cid), 0.0), weight)

    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [cid for cid, _ in ordered]


def select_nodes(
    concepts: ConceptDict,
    max_nodes: int | None,
    focus_concept: str | None,
    min_edge_weight: float,
    focus_edge_cap: int | None,
) -> List[str]:
    if not concepts:
        return []

    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")

    if max_nodes is None:
        if focus_concept and str(focus_concept) in concepts:
            focus_id = str(focus_concept)
            ordered = [focus_id] + [
                cid for cid in concepts.keys() if cid != focus_id
            ]
            return ordered
        return list(concepts.keys())

    if focus_concept:
        focus_id = str(focus_concept)
        if focus_id not in concepts:
            raise ValueError(f"Concept id {focus_concept} not found")
        ordered = [focus_id]
        neighbors = _focus_neighbors(concepts, focus_id, min_edge_weight)
        neighbor_limit = max_nodes - 1
        if focus_edge_cap is not None:
            neighbor_limit = min(neighbor_limit, focus_edge_cap)
        for neighbor in neighbors[: max(0, neighbor_limit)]:
            if neighbor == focus_id:
                continue
            ordered.append(neighbor)
        return ordered

    by_attempts = sorted(
        concepts.items(), key=lambda x: -x[1].get("total_attempts", 0)
    )
    return [cid for cid, _ in by_attempts[:max_nodes]]


def build_viz_payload(
    concepts: ConceptDict,
    selected_nodes: Iterable[str],
    min_edge_weight: float,
    max_edges_per_node: int | None,
) -> Tuple[List[NodeViz], List[EdgeViz]]:
    selected = [nid for nid in selected_nodes if nid in concepts]

    max_attempts = max(
        (concepts[n]["total_attempts"] for n in selected), default=1
    )

    nodes: List[NodeViz] = []
    for nid in selected:
        c = concepts[nid]
        attempts = c.get("total_attempts", 0)
        nodes.append(
            NodeViz(
                concept_id=nid,
                name=c.get("name", "Unknown Concept"),
                accuracy=float(c.get("accuracy", 0.0)),
                total_attempts=attempts,
                relative_attempts=attempts / max_attempts if max_attempts else 0.0,
            )
        )

    edges: List[EdgeViz] = []
    for nid in selected:
        per_node: List[EdgeViz] = []
        merge_relations = concepts[nid].get("merge_relations") or {}
        for dst, payload in merge_relations.items():
            dst_id = str(dst)
            try:
                weight = float(payload.get("final_score", 0.0))
            except (TypeError, ValueError):
                continue
            if weight < min_edge_weight or dst_id not in selected:
                continue
            evidence = str(payload.get("source") or "merge_relations")
            per_node.append(EdgeViz(nid, dst_id, weight, evidence))
        if max_edges_per_node is not None and len(per_node) > max_edges_per_node:
            per_node.sort(key=lambda e: e.weight, reverse=True)
            per_node = per_node[:max_edges_per_node]
        edges.extend(per_node)

    return nodes, edges


def prune_isolated_nodes(
    nodes: List[NodeViz], edges: List[EdgeViz], focus_concept: str | None
) -> Tuple[List[NodeViz], List[EdgeViz]]:
    if not nodes:
        return nodes, edges

    connected_ids = {e.source for e in edges} | {e.target for e in edges}
    focus_id = str(focus_concept) if focus_concept is not None else None
    focus_present = bool(
        focus_id and any(n.concept_id == focus_id for n in nodes)
    )
    if focus_present:
        connected_ids.add(focus_id)

    if not connected_ids:
        return (
            [next((n for n in nodes if n.concept_id == focus_id), nodes[0])]
            if nodes
            else [],
            [],
        )

    filtered_nodes = [n for n in nodes if n.concept_id in connected_ids]
    return filtered_nodes, edges


# =========================
# 布局
# =========================
def _place_circle(
    pos: Dict[str, Tuple[float, float]],
    ids: List[str],
    radius: float,
    start: float = 0.0,
):
    n = len(ids)
    for i, cid in enumerate(ids):
        a = start + 2 * math.pi * i / n
        pos[cid] = (radius * math.cos(a), radius * math.sin(a))


def compute_layout(nodes: List[NodeViz], focus: str | None):
    ids = [n.concept_id for n in nodes]
    pos: Dict[str, Tuple[float, float]] = {}

    if not ids:
        return pos

    total = len(ids)

    if total <= 40:
        if focus and focus in ids:
            pos[focus] = (0.0, 0.0)
            others = [i for i in ids if i != focus]
            _place_circle(pos, others[:10], 1.8)
            if len(others) > 10:
                _place_circle(pos, others[10:], 3.0, math.pi / 8)
        else:
            _place_circle(pos, ids[:12], 2.0)
            if len(ids) > 12:
                _place_circle(pos, ids[12:], 3.2, math.pi / 12)
        return pos

    if focus and focus in ids:
        pos[focus] = (0.0, 0.0)
        remaining = [cid for cid in ids if cid != focus]
    else:
        remaining = ids

    golden_angle = math.pi * (3 - math.sqrt(5))
    gap_scale = 1.0 + min(1.5, max(0, total - 50) / 600)
    spacing = SPIRAL_BASE_GAP * gap_scale

    for idx, cid in enumerate(remaining):
        radius = spacing * math.sqrt(idx + 1)
        angle = idx * golden_angle
        pos[cid] = (radius * math.cos(angle), radius * math.sin(angle))

    return pos


# =========================
# 绘图
# =========================
def draw_graph(
    nodes: List[NodeViz],
    edges: List[EdgeViz],
    output_path: Path,
    focus_concept: str | None,
    force_labels: bool,
) -> Path:
    pos = compute_layout(nodes, focus_concept)
    node_count = len(nodes)
    fig_width = min(32.0, 10.0 + node_count / 90.0)
    fig_height = max(8.0, fig_width * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

    # Edges
    if edges:
        weights = [e.weight for e in edges]
        min_w = min(weights)
        max_w = max(weights)
        spread = max(max_w - min_w, 1e-9)
        edge_cmap = plt.get_cmap("YlOrRd")
        min_width, max_width = 0.4, 5.4
        min_mutation, max_mutation = 9.0, 26.0
        for e in edges:
            strength = (e.weight - min_w) / spread
            strength = max(0.0, min(1.0, strength))
            emphasis = strength ** 0.65
            linewidth = min_width + (max_width - min_width) * emphasis
            mutation = min_mutation + (max_mutation - min_mutation) * emphasis
            color_strength = max(0.0, min(1.0, strength ** 0.85))
            color = edge_cmap(0.05 + 0.9 * color_strength)
            alpha = 0.12 + 0.88 * emphasis
            arrow = FancyArrowPatch(
                pos[e.source],
                pos[e.target],
                arrowstyle="-|>",
                linewidth=linewidth,
                mutation_scale=mutation,
                color=color,
                alpha=alpha,
                shrinkA=12,
                shrinkB=12,
                connectionstyle="arc3,rad=0.08",
            )
            ax.add_patch(arrow)

    # Nodes
    norm = colors.Normalize(
        vmin=min(n.accuracy for n in nodes),
        vmax=max(n.accuracy for n in nodes) + 1e-3,
    )
    cmap = plt.get_cmap("YlGnBu")

    show_labels = force_labels or node_count <= LABEL_NODE_LIMIT
    if node_count <= 120:
        label_fontsize = 8
    elif node_count <= 400:
        label_fontsize = 6
    elif node_count <= 1200:
        label_fontsize = 4
    else:
        label_fontsize = 3
    density_scale = (
        1.0
        if node_count <= 200
        else max(0.25, min(1.0, 200 / max(node_count, 1)))
    )
    label_offset = 0.12 * density_scale
    size_base = 420 * density_scale
    size_span = 1600 * density_scale

    xs, ys, cs, ss = [], [], [], []
    for n in nodes:
        x, y = pos[n.concept_id]
        xs.append(x)
        ys.append(y)
        cs.append(cmap(norm(n.accuracy)))
        ss.append(size_base + size_span * n.relative_attempts)

        if show_labels:
            label_text = f"{n.name} ({n.concept_id})"
            if CH_FONT:
                ax.text(
                    x,
                    y + label_offset,
                    label_text,
                    fontsize=label_fontsize,
                    ha="center",
                    va="bottom",
                    fontproperties=CH_FONT,
                    zorder=4,
                )
            else:
                ax.text(
                    x,
                    y + label_offset,
                    label_text,
                    fontsize=label_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=4,
                )

    ax.scatter(xs, ys, s=ss, c=cs, edgecolors="#333", linewidths=1.2, zorder=3)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    if CH_FONT:
        cbar.set_label("Concept Accuracy", fontproperties=CH_FONT)
    else:
        cbar.set_label("Concept Accuracy")

    title = f"Knowledge Graph ({len(nodes)} concepts, {len(edges)} edges)"
    if CH_FONT:
        ax.set_title(title, fontsize=14, fontproperties=CH_FONT)
    else:
        ax.set_title(title, fontsize=14)

    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH_PATH)
    p.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Limit the number of nodes to visualize (default: include all nodes).",
    )
    p.add_argument(
        "--min-edge-weight",
        type=float,
        default=MERGED_DEFAULT_MIN_WEIGHT,
        help=f"Minimum merged edge final_score to draw (default: {MERGED_DEFAULT_MIN_WEIGHT}).",
    )
    p.add_argument("--focus-concept-id", type=str, default=None)
    p.add_argument(
        "--max-edges-per-node",
        type=int,
        default=DEFAULT_MAX_EDGES_PER_NODE,
        help=(
            "Maximum number of outgoing edges to draw per node (<=0 keeps all). "
            f"Default: {DEFAULT_MAX_EDGES_PER_NODE}."
        )
    )
    p.add_argument(
        "--force-labels",
        action="store_true",
        help="Always render concept labels (may be unreadable when nodes > 400).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data = load_graph(args.graph_path)
    concepts = data.get("concepts", {})
    min_edge_weight = max(args.min_edge_weight, 0.0)
    max_edges_per_node = args.max_edges_per_node
    if max_edges_per_node is not None and max_edges_per_node <= 0:
        max_edges_per_node = None

    selected = select_nodes(
        concepts,
        max_nodes=args.max_nodes,
        focus_concept=args.focus_concept_id,
        min_edge_weight=min_edge_weight,
        focus_edge_cap=max_edges_per_node,
    )
    nodes, edges = build_viz_payload(
        concepts,
        selected,
        min_edge_weight=min_edge_weight,
        max_edges_per_node=max_edges_per_node,
    )
    nodes, edges = prune_isolated_nodes(nodes, edges, args.focus_concept_id)

    out = draw_graph(
        nodes,
        edges,
        args.output_path,
        args.focus_concept_id,
        args.force_labels,
    )
    print(f"Visualization saved to {out}")


if __name__ == "__main__":
    main()
