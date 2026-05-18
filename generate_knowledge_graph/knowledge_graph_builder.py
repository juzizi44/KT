"""Utility functions to build and persist a knowledge graph for the dataset.

Adapted from kt_llm/src/generate_knowledge_graph/knowledge_graph_builder_0410.py
for the Explainable-Few-shot-Knowledge-Tracing data format (JSONL).

Key adaptations:
- Input format: recordings.jsonl + exercise_info.jsonl (instead of CSV)
- Field mapping: skill_ids -> concept_ids, skill_desc -> knowledge_concepts
- Timestamp: Using list index as implicit temporal order
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR.parent / "datasets" / "sparse" / "FrcSub"
DEFAULT_OUTPUT_PATH = BASE_DIR.parent / "datasets" / "sparse" / "FrcSub" / "knowledge_graph.json"

# Heuristics for correctness-driven prerequisite inference
MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS = 3
MIN_JOINT_SUCCESSES_FOR_CORRECTNESS = 2
PREREQUISITE_CONFIDENCE_THRESHOLD = 0.6
PREREQUISITE_FAIL_RATIO_MAX = 0.3
MIN_PREREQ_ATTEMPT_OVERLAP = 1
MIN_DEP_FAIL_OVERLAP = 2
BACKWARD_FAIL_RATIO_MIN = 0.55
COMBINED_SCORE_THRESHOLD = 0.6


@dataclass
class ExerciseRow:
    """Parsed representation of a single exercise submission."""

    student_id: str
    exercise_id: str
    is_correct: int
    concept_ids: List[str]
    knowledge_concepts: List[str]
    timestamp: int  # Using index as implicit timestamp


def _load_exercise_info(data_path: Path, data_mode: str = "sparse") -> Dict[str, Dict]:
    """Load exercise info from JSONL file.

    Args:
        data_path: Path to dataset directory containing exercise_info.jsonl
        data_mode: One of 'onehot', 'sparse', 'moderate'

    Returns:
        Dict mapping exercise_id to exercise info (skill_ids, skill_desc)
    """
    if data_mode == "onehot":
        info_file = data_path / "onehot_exercise_info.jsonl"
    elif data_mode == "sparse":
        info_file = data_path / "sparse_exercise_info.jsonl"
    elif data_mode == "moderate":
        info_file = data_path / "moderate_exercise_info.jsonl"
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

    if not info_file.exists():
        raise FileNotFoundError(f"Exercise info file not found: {info_file}")

    exercise_info = {}
    with info_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            ex_id = str(record.get("exercise_id", ""))
            if ex_id:
                exercise_info[ex_id] = {
                    "skill_ids": record.get("skill_ids", []),
                    "skill_desc": record.get("skill_desc", []),
                }
    return exercise_info


def _load_rows(
    data_path: Path,
    data_mode: str = "sparse",
    train_split: float = 1.0,
) -> Iterable[ExerciseRow]:
    """Yield parsed ExerciseRow objects from JSONL files.

    Args:
        data_path: Path to dataset directory
        data_mode: One of 'onehot', 'sparse', 'moderate'
        train_split: Fraction of each student's history to use (for consistency with main.py)

    Yields:
        ExerciseRow objects with implicit timestamp based on index
    """
    recordings_path = data_path / "recordings.jsonl"
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings file not found: {recordings_path}")

    exercise_info = _load_exercise_info(data_path, data_mode)

    with recordings_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            student_id = str(record.get("student_id", ""))
            exercises_logs = record.get("exercises_logs", [])
            is_corrects = record.get("is_corrects", [])

            # Determine split point
            if train_split < 1.0:
                split_idx = int(len(exercises_logs) * train_split)
                exercises_logs = exercises_logs[:split_idx]
                is_corrects = is_corrects[:split_idx]

            for idx, (ex_id, is_correct) in enumerate(zip(exercises_logs, is_corrects)):
                ex_id_str = str(ex_id)
                ex_info = exercise_info.get(ex_id_str, {})
                skill_ids = ex_info.get("skill_ids", [])
                skill_desc = ex_info.get("skill_desc", [])

                # Convert skill_ids to strings (to match concept_ids format)
                concept_ids = [str(sid) for sid in skill_ids]

                yield ExerciseRow(
                    student_id=student_id,
                    exercise_id=ex_id_str,
                    is_correct=int(is_correct),
                    concept_ids=concept_ids,
                    knowledge_concepts=skill_desc if skill_desc else ["Unknown Concept"],
                    timestamp=idx,  # Using index as implicit timestamp
                )


def _evaluate_prerequisite_relation(
    prereq: str,
    dependent: str,
    both_success: int,
    concept_success_students: Dict[str, Set[str]],
    concept_attempt_students: Dict[str, Set[str]],
    concept_failed_students: Dict[str, Set[str]],
) -> Optional[Dict[str, float]]:
    """
    Determine whether `prereq` behaves like a prerequisite for `dependent`.

    Conditions (heuristic):
    1. Most students who succeed on `dependent` also succeed on `prereq`.
    2. A meaningful portion of students who succeed on `prereq` attempt `dependent`
       but fail it, indicating `dependent` is harder.
    3. Among students who fail `dependent`, many also fail `prereq`
       (backward support: 缺先修→后续失败).
    4. The final edge score is a geometric mean of forward confidence,
       forward stability, and backward failure support.
    """

    prereq_success = concept_success_students.get(prereq, set())
    dependent_success = concept_success_students.get(dependent, set())
    if (
        len(prereq_success) < MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS
        or len(dependent_success) < MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS
        or both_success < MIN_JOINT_SUCCESSES_FOR_CORRECTNESS
    ):
        return None

    prereq_given_dependent = both_success / len(dependent_success)
    if prereq_given_dependent < PREREQUISITE_CONFIDENCE_THRESHOLD:
        return None

    dependent_attempters = concept_attempt_students.get(dependent, set())
    if not dependent_attempters:
        return None

    prereq_success_attempted_dependent = prereq_success & dependent_attempters
    if len(prereq_success_attempted_dependent) < MIN_PREREQ_ATTEMPT_OVERLAP:
        return None

    dependent_failed_students = concept_failed_students.get(dependent, set())
    fail_support = prereq_success_attempted_dependent & dependent_failed_students
    if not fail_support:
        return None

    fail_ratio = len(fail_support) / len(prereq_success_attempted_dependent)
    if fail_ratio > PREREQUISITE_FAIL_RATIO_MAX:
        return None

    prereq_attempters = concept_attempt_students.get(prereq, set())
    prereq_failed_students = concept_failed_students.get(prereq, set())
    backward_attempt_overlap = dependent_failed_students & prereq_attempters
    if len(backward_attempt_overlap) < MIN_DEP_FAIL_OVERLAP:
        return None

    backward_fail_support = backward_attempt_overlap & prereq_failed_students
    back_fail_ratio = len(backward_fail_support) / len(backward_attempt_overlap)
    if back_fail_ratio < BACKWARD_FAIL_RATIO_MIN:
        return None

    stable = 1 - fail_ratio
    combined_score = (prereq_given_dependent * stable * back_fail_ratio) ** (1 / 3)
    if combined_score < COMBINED_SCORE_THRESHOLD:
        return None

    return {
        "correctness_score": round(combined_score, 4),
        "combined_score": round(combined_score, 4),
        "confidence": round(prereq_given_dependent, 4),
        "fail_ratio": round(fail_ratio, 4),
        "back_fail": round(back_fail_ratio, 4),
        "joint_success": both_success,
        "fail_support": len(fail_support),
        "back_fail_support": len(backward_fail_support),
        "forward_support": len(prereq_success_attempted_dependent),
        "backward_support": len(backward_attempt_overlap),
    }


def _compute_sequence_scores(forward_edges: Dict[str, int]) -> Dict[str, float]:
    """Normalize outgoing transition counts into probabilities."""

    total_weight = 0.0
    cleaned_edges: Dict[str, float] = {}
    for dst, weight in forward_edges.items():
        if weight <= 0:
            continue
        cleaned_edges[dst] = float(weight)
        total_weight += float(weight)

    if total_weight <= 0:
        return {}

    return {dst: weight / total_weight for dst, weight in cleaned_edges.items()}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _merge_relations(
    correctness_rel: Dict[str, Dict[str, Any]],
    forward_rel: Dict[str, Dict[str, Any]],
    sequence_scale: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """Merge correctness-based and sequence-based relations into a unified scoring system."""

    merged: Dict[str, Dict[str, float]] = {}
    stats = {"correctness_edges_used": 0, "sequence_edges_used": 0}
    targets = sorted(set(correctness_rel.keys()) | set(forward_rel.keys()))

    for dst in targets:
        correctness_score = _safe_float(
            (correctness_rel.get(dst) or {}).get("correctness_score")
        )
        sequence_score = _safe_float(
            (forward_rel.get(dst) or {}).get("sequence_score")
        )

        if correctness_score <= 0 and sequence_score <= 0:
            continue

        final_score = (correctness_score * (1 - sequence_scale)) + (sequence_score * sequence_scale)
        stats["correctness_edges_used"] += 1
        stats["sequence_edges_used"] += 1
        source = "both"

        merged[dst] = {
            "correctness_score": round(correctness_score, 4),
            "sequence_score": round(sequence_score, 4),
            "source": source,
            "final_score": round(final_score, 4),
        }

    return merged, stats


def build_knowledge_graph(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    data_mode: str = "sparse",
    train_split: float = 1.0,
    min_edge_weight: int = 1,
    sequence_scale: float = 0.2,
) -> Dict:
    """
    Build a prerequisite-style knowledge graph enriched with merged scoring signals.

    Args:
        data_path: Path to dataset directory containing recordings.jsonl and exercise_info.jsonl
        output_path: Path where the generated knowledge graph JSON should be stored
        data_mode: One of 'onehot', 'sparse', 'moderate'
        train_split: Fraction of each student's history to use
        min_edge_weight: Minimum transition count for time-sequence edges
        sequence_scale: Weight of sequence signal in merged score

    Returns:
        The generated knowledge graph dictionary
    """

    student_records: Dict[str, List[ExerciseRow]] = defaultdict(list)
    concept_attempts = defaultdict(int)
    concept_correct = defaultdict(int)
    concept_name = {}
    student_attempted_concepts: Dict[str, Set[str]] = defaultdict(set)
    student_correct_concepts: Dict[str, Set[str]] = defaultdict(set)
    concept_attempt_students: Dict[str, Set[str]] = defaultdict(set)
    concept_success_students: Dict[str, Set[str]] = defaultdict(set)

    total_rows = 0
    for row in _load_rows(data_path, data_mode, train_split):
        total_rows += 1
        student_records[row.student_id].append(row)
        if row.concept_ids:
            student_attempted_concepts[row.student_id].update(row.concept_ids)
            if row.is_correct:
                student_correct_concepts[row.student_id].update(row.concept_ids)
        for idx, concept_id in enumerate(row.concept_ids):
            concept_attempts[concept_id] += 1
            if row.is_correct:
                concept_correct[concept_id] += 1
                concept_success_students[concept_id].add(row.student_id)
            concept_attempt_students[concept_id].add(row.student_id)

            if concept_id not in concept_name:
                if idx < len(row.knowledge_concepts):
                    concept_name[concept_id] = row.knowledge_concepts[idx]
                elif row.knowledge_concepts:
                    concept_name[concept_id] = row.knowledge_concepts[0]
                else:
                    concept_name[concept_id] = "Unknown Concept"

    concept_failed_students: Dict[str, Set[str]] = defaultdict(set)
    for student_id, attempted in student_attempted_concepts.items():
        successes = student_correct_concepts.get(student_id, set())
        for concept_id in attempted - successes:
            concept_failed_students[concept_id].add(student_id)

    joint_success_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for concepts in student_correct_concepts.values():
        unique_concepts = sorted(concepts)
        for idx, concept_a in enumerate(unique_concepts):
            for concept_b in unique_concepts[idx + 1 :]:
                joint_success_counts[(concept_a, concept_b)] += 1

    correctness_enables: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (concept_a, concept_b), both_success in joint_success_counts.items():
        relation_ab = _evaluate_prerequisite_relation(
            prereq=concept_a,
            dependent=concept_b,
            both_success=both_success,
            concept_success_students=concept_success_students,
            concept_attempt_students=concept_attempt_students,
            concept_failed_students=concept_failed_students,
        )
        if relation_ab:
            correctness_enables[concept_a][concept_b] = relation_ab

        relation_ba = _evaluate_prerequisite_relation(
            prereq=concept_b,
            dependent=concept_a,
            both_success=both_success,
            concept_success_students=concept_success_students,
            concept_attempt_students=concept_attempt_students,
            concept_failed_students=concept_failed_students,
        )
        if relation_ba:
            correctness_enables[concept_b][concept_a] = relation_ba

    forward_links: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for history in student_records.values():
        history.sort(key=lambda record: record.timestamp)
        previous_concepts: List[str] = []
        for record in history:
            if previous_concepts:
                for prev in previous_concepts:
                    for current in record.concept_ids:
                        if prev == current:
                            continue
                        forward_links[prev][current] += 1
            previous_concepts = record.concept_ids

    filtered_forward_links: Dict[str, Dict[str, int]] = {}
    for src, targets in forward_links.items():
        strong_targets = {
            dst: weight for dst, weight in targets.items() if weight >= min_edge_weight
        }
        if strong_targets:
            filtered_forward_links[src] = strong_targets

    correctness_edge_count = sum(len(targets) for targets in correctness_enables.values())

    graph = {
        "meta": {
            "source": str(data_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_mode": data_mode,
            "train_split": train_split,
            "total_rows": total_rows,
            "students": len(student_records),
            "min_edge_weight": min_edge_weight,
            "sequence_scale": sequence_scale,
            "correctness_config": {
                "min_success_students": MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS,
                "min_joint_successes": MIN_JOINT_SUCCESSES_FOR_CORRECTNESS,
                "confidence_threshold": PREREQUISITE_CONFIDENCE_THRESHOLD,
                "fail_ratio_max": PREREQUISITE_FAIL_RATIO_MAX,
                "min_attempt_overlap": MIN_PREREQ_ATTEMPT_OVERLAP,
                "min_backward_overlap": MIN_DEP_FAIL_OVERLAP,
                "backward_fail_min": BACKWARD_FAIL_RATIO_MIN,
                "combined_score_threshold": COMBINED_SCORE_THRESHOLD,
            },
            "correctness_edges": correctness_edge_count,
        },
        "concepts": {},
    }

    forward_edge_count = 0
    merge_edge_count = 0
    merge_correctness_edges = 0
    merge_sequence_edges = 0

    for concept_id, attempts in concept_attempts.items():
        if attempts == 0:
            continue

        accuracy = concept_correct[concept_id] / attempts
        forward_edges = filtered_forward_links.get(concept_id, {})
        sequence_scores = _compute_sequence_scores(forward_edges)

        forward_time_relations = {}
        for dst, edge_count in sorted(
            forward_edges.items(),
            key=lambda item: -sequence_scores.get(item[0], 0.0),
        ):
            dst_key = str(dst)
            forward_time_relations[dst_key] = {
                "edge_count": int(edge_count),
                "sequence_score": round(sequence_scores.get(dst, 0.0), 4),
            }

        correctness_relations = {}
        enables_rel = correctness_enables.get(concept_id, {})
        for dst, payload in sorted(
            enables_rel.items(),
            key=lambda item: -item[1].get("combined_score", item[1].get("correctness_score", 0.0)),
        ):
            dst_key = str(dst)
            correctness_relations[dst_key] = {
                "correctness_score": round(payload.get("correctness_score", 0.0), 4),
                "combined_score": round(payload.get("combined_score", 0.0), 4),
                "confidence": round(payload.get("confidence", 0.0), 4),
                "joint_success": int(payload.get("joint_success", 0)),
                "fail_support": int(payload.get("fail_support", 0)),
                "fail_ratio": round(payload.get("fail_ratio", 0.0), 4),
                "back_fail": round(payload.get("back_fail", 0.0), 4),
                "back_fail_support": int(payload.get("back_fail_support", 0)),
                "forward_support": int(payload.get("forward_support", 0)),
                "backward_support": int(payload.get("backward_support", 0)),
            }

        merged_relations, merge_stats = _merge_relations(
            correctness_relations, forward_time_relations, sequence_scale
        )

        forward_edge_count += len(forward_time_relations)
        merge_edge_count += len(merged_relations)
        merge_correctness_edges += merge_stats["correctness_edges_used"]
        merge_sequence_edges += merge_stats["sequence_edges_used"]

        graph["concepts"][str(concept_id)] = {
            "concept_id": concept_id,
            "name": concept_name.get(concept_id, "Unknown Concept"),
            "accuracy": round(accuracy, 4),
            "correct_attempts": concept_correct[concept_id],
            "total_attempts": attempts,
            "forward_time_relations": forward_time_relations,
            "correctness_relations": correctness_relations,
            "merge_relations": merged_relations,
        }

    graph["meta"]["edge_statistics"] = {
        "forward_time_edges": forward_edge_count,
        "correctness_edges": correctness_edge_count,
        "merge_edges": merge_edge_count,
        "merge_correctness_edges": merge_correctness_edges,
        "merge_sequence_edges": merge_sequence_edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as graph_fp:
        json.dump(graph, graph_fp, ensure_ascii=False, indent=2)

    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prerequisite-style knowledge graph from exercise records (JSONL format)."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the dataset directory containing recordings.jsonl and exercise_info.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the generated knowledge graph JSON should be stored.",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="sparse",
        choices=["onehot", "sparse", "moderate"],
        help="Data mode to determine exercise_info file name.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Fraction of each student's history to use (default: 1.0 = all).",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=int,
        default=2,
        help="Drop edges whose transition count is below this threshold.",
    )
    parser.add_argument(
        "--sequence-scale",
        type=float,
        default=0.2,
        help="Scaling factor applied when only the time-sequence signal is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph = build_knowledge_graph(
        data_path=args.data_path,
        output_path=args.output_path,
        data_mode=args.data_mode,
        train_split=args.train_split,
        min_edge_weight=max(1, args.min_edge_weight),
        sequence_scale=max(0.0, args.sequence_scale),
    )
    print(
        f"Knowledge graph built for {len(graph['concepts'])} concepts "
        f"with data from {graph['meta']['students']} students."
    )
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
