"""Knowledge graph generation module.

This module provides tools to build prerequisite-style knowledge graphs
from exercise records, adapted for the Explainable-Few-shot-Knowledge-Tracing
data format (JSONL files).

Ported from kt_llm/src/generate_knowledge_graph with adaptations.
"""

from .knowledge_graph_builder import build_knowledge_graph

__all__ = ["build_knowledge_graph"]
