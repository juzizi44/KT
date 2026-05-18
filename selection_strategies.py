"""Few-shot selection strategy implementations.

Ported from kt_llm/src/selection_strategies.py with adaptations for
Explainable-Few-shot-Knowledge-Tracing data format.

Key adaptations:
- concept_ids -> skill_ids (string type)
- knowledge_concepts -> skill_desc
- timestamp is optional (using index as implicit order)
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import jieba

BASE_DIR = Path(__file__).resolve().parent

DATASET_GRAPH_PATHS: Dict[str, Path] = {
    "frcsub": BASE_DIR / "datasets" / "sparse" / "FrcSub" / "knowledge_graph.json",
    "moocradar": BASE_DIR / "datasets" / "moderate" / "MOOCRadar" / "knowledge_graph.json",
    "xes3g5m": BASE_DIR / "datasets" / "moderate" / "XES3G5M" / "knowledge_graph.json",
}

DATASET_ALIASES: Dict[str, str] = {
    "frcsub": "frcsub",
    "math2015": "frcsub",
    "moocradar": "moocradar",
    "moocradar_middle": "moocradar",
    "xes3g5m": "xes3g5m",
}


def _infer_dataset_key_from_name(dataset_name: str) -> Optional[str]:
    """Infer dataset identifier from dataset name."""
    lowered = dataset_name.lower()
    for alias, canonical in DATASET_ALIASES.items():
        if alias in lowered:
            return canonical
    return None


class FewShotSelector(ABC):
    """Base class for few-shot selection strategies."""

    @abstractmethod
    def select(
        self,
        records: List[Dict[str, Any]],
        n_shots: int,
        test_record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Select few-shot samples for the given test record.

        Args:
            records: List of available training records, each containing:
                - exercise_id: str
                - skill_ids: List[str]
                - skill_desc: List[str] (optional, for concept-based selection)
                - is_correct: bool/int
                - index: int (implicit order, used as timestamp substitute)
            n_shots: Number of samples to select
            test_record: The test record with same structure

        Returns:
            Selected records sorted by index (implicit time order)
        """
        raise NotImplementedError


class RandomSelector(FewShotSelector):
    """Selects a random subset of the available records."""

    def select(self, records, n_shots, test_record):
        if len(records) <= n_shots:
            selected = records
        else:
            selected = random.sample(records, n_shots)

        return sorted(selected, key=lambda r: r.get("index", 0))


class FirstSelector(FewShotSelector):
    """Selects the earliest records (by index)."""

    def select(self, records, n_shots, test_record):
        sorted_records = sorted(records, key=lambda r: r.get("index", 0))
        selected = sorted_records[: min(n_shots, len(sorted_records))]
        return selected


class RecentSelector(FewShotSelector):
    """Selects the most recent records (by index)."""

    def select(self, records, n_shots, test_record):
        sorted_records = sorted(records, key=lambda r: r.get("index", 0))
        selected = sorted_records[-min(n_shots, len(sorted_records)) :]
        return selected


class ConceptBasedSelector(FewShotSelector):
    """Selects records based on concept overlap using Jieba for loose matching.

    Adapted from kt_llm:
    - Uses skill_ids instead of concept_ids
    - Uses skill_desc instead of knowledge_concepts
    """

    def _get_concept_words(self, concept_names: List[str]) -> Set[str]:
        """Tokenize concept names to support loose matching."""
        all_words: Set[str] = set()
        for concept_name in concept_names:
            words = jieba.cut(concept_name, cut_all=False)
            all_words.update(word.lower() for word in words if word.strip())
        return all_words

    def select(self, records, n_shots, test_record):
        test_concepts = set(test_record.get("skill_ids", []))
        test_desc = test_record.get("skill_desc", [])
        test_words = self._get_concept_words(test_desc) if test_desc else set()

        relevant_records = []
        loose_relevant_records = []
        other_records = []

        for record in records:
            record_concepts = set(record.get("skill_ids", []))

            if test_concepts & record_concepts:
                relevant_records.append(record)
            else:
                record_desc = record.get("skill_desc", [])
                if record_desc:
                    record_words = self._get_concept_words(record_desc)
                    if test_words & record_words:
                        loose_relevant_records.append(record)
                    else:
                        other_records.append(record)
                else:
                    other_records.append(record)

        if len(relevant_records) >= n_shots:
            selected = random.sample(relevant_records, n_shots)
        else:
            selected = relevant_records.copy()
            need_more = n_shots - len(selected)

            if need_more > 0 and loose_relevant_records:
                additional = random.sample(
                    loose_relevant_records,
                    min(need_more, len(loose_relevant_records)),
                )
                selected.extend(additional)
                need_more = n_shots - len(selected)

            if need_more > 0 and other_records:
                additional = random.sample(
                    other_records, min(need_more, len(other_records))
                )
                selected.extend(additional)

        return sorted(selected, key=lambda r: r.get("index", 0))


class KnowledgeGraphSelector(FewShotSelector):
    """Selects records informed by the prerequisite-style knowledge graph.

    Adapted from kt_llm:
    - Uses skill_ids (string) converted to str for graph lookup
    - No timestamp, uses index for relative ordering
    - Recency bonus based on index position
    """

    def __init__(
        self,
        graph_path: Optional[Path] = None,
        max_related_concepts: int = 5,
        dataset_name: Optional[str] = None,
    ):
        self.graph_path = Path(graph_path) if graph_path else None
        self._graph_path_locked = graph_path is not None
        if not self._graph_path_locked and dataset_name:
            dataset_key = _infer_dataset_key_from_name(dataset_name)
            if dataset_key and dataset_key in DATASET_GRAPH_PATHS:
                self.graph_path = DATASET_GRAPH_PATHS[dataset_key]
        self.max_related_concepts = max_related_concepts
        self._graph_cache: Optional[Dict] = None

    def configure_for_dataset(self, dataset_name: Optional[str]) -> None:
        """Update the graph path based on the provided dataset name."""
        if self._graph_path_locked or not dataset_name:
            return
        dataset_key = _infer_dataset_key_from_name(dataset_name)
        if dataset_key:
            new_path = DATASET_GRAPH_PATHS.get(dataset_key)
            if new_path and new_path != self.graph_path:
                self.graph_path = new_path
                self._graph_cache = None

    def _load_graph(self) -> Dict:
        if self._graph_cache is None:
            if self.graph_path and self.graph_path.exists():
                try:
                    with self.graph_path.open("r", encoding="utf-8") as graph_fp:
                        self._graph_cache = json.load(graph_fp)
                except (FileNotFoundError, json.JSONDecodeError):
                    self._graph_cache = {"concepts": {}}
            else:
                self._graph_cache = {"concepts": {}}
        return self._graph_cache

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _collect_from_merge_relations(
        self, node: Dict[str, Any], weights: Dict[str, float]
    ) -> None:
        merge_relations = node.get("merge_relations") or {}
        if not merge_relations:
            return

        sorted_neighbors = sorted(
            merge_relations.items(),
            key=lambda item: -self._safe_float(item[1].get("final_score", 0.0)),
        )[: self.max_related_concepts]

        for neighbor_id, payload in sorted_neighbors:
            final_score = self._safe_float(payload.get("final_score", 0.0))
            if final_score <= 0:
                continue

            weights[str(neighbor_id)] = max(
                weights.get(str(neighbor_id), 0.0), final_score
            )

    def _collect_related_concepts(self, test_concepts: List[str]) -> Dict[str, float]:
        graph = self._load_graph()
        concept_data = graph.get("concepts", {})
        weights: Dict[str, float] = {}

        for concept_id in test_concepts:
            concept_id_str = str(concept_id)
            weights[concept_id_str] = max(weights.get(concept_id_str, 0.0), 1.0)

            node = concept_data.get(concept_id_str)
            if not node:
                continue
            self._collect_from_merge_relations(node, weights)

        return weights

    def _get_recency_bounds(
        self, records: List[Dict[str, Any]]
    ) -> Optional[Tuple[int, int]]:
        indices = [record.get("index", 0) for record in records]
        if not indices:
            return None
        return min(indices), max(indices)

    def _score_record(
        self,
        record: Dict[str, Any],
        weights: Dict[str, float],
        recency_bounds: Optional[Tuple[int, int]],
    ) -> float:
        score = 0.0

        for concept_id in record.get("skill_ids", []):
            score += weights.get(str(concept_id), 0.0)

        # Recency bonus based on index (since no timestamp)
        if recency_bounds:
            index = record.get("index", 0)
            earliest, latest = recency_bounds
            if latest > earliest:
                recency = (index - earliest) / (latest - earliest)
            else:
                recency = 0.0
            score += recency * 0.2

        # Correctness bonus
        is_correct = record.get("is_correct")
        if is_correct is True or is_correct == 1 or is_correct == "1":
            score += 0.1

        return score

    def select(self, records, n_shots, test_record):
        if not records:
            return []

        concept_weights = self._collect_related_concepts(
            test_record.get("skill_ids", [])
        )
        if not concept_weights:
            selector = RecentSelector()
            return selector.select(records, n_shots, test_record)

        recency_bounds = self._get_recency_bounds(records)
        scored_records = [
            (self._score_record(record, concept_weights, recency_bounds), record)
            for record in records
        ]
        scored_records.sort(
            key=lambda item: (-item[0], item[1].get("index", 0))
        )
        selected = [record for _, record in scored_records[: min(n_shots, len(records))]]
        return sorted(selected, key=lambda r: r.get("index", 0))


# Factory function for easy instantiation
def create_selector(
    strategy: str,
    dataset_name: Optional[str] = None,
    graph_path: Optional[Union[str, Path]] = None,
) -> FewShotSelector:
    """Factory function to create a selector based on strategy name.

    Args:
        strategy: One of 'random', 'first', 'recent', 'concept_based', 'knowledge_graph'
        dataset_name: Dataset name for KnowledgeGraphSelector path resolution
        graph_path: Explicit path to knowledge graph (overrides dataset_name)

    Returns:
        Appropriate FewShotSelector instance
    """
    strategy_lower = strategy.lower()

    if strategy_lower == "random":
        return RandomSelector()
    elif strategy_lower == "first":
        return FirstSelector()
    elif strategy_lower == "recent" or strategy_lower == "last":
        return RecentSelector()
    elif strategy_lower == "concept_based":
        return ConceptBasedSelector()
    elif strategy_lower == "knowledge_graph":
        return KnowledgeGraphSelector(
            graph_path=Path(graph_path) if graph_path else None,
            dataset_name=dataset_name,
        )
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
