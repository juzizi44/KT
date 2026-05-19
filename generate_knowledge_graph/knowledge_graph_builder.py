"""Build a prerequisite-style knowledge graph following the three-step pipeline:

   Step 1 — Aggregate a, b, c, d counts via Cartesian product of exercise pairs
   Step 2 — Filter by Phi coefficient (φ ≥ 0.35)
   Step 3 — Direction discrimination: forward/backward confidence & combined score ≥ 0.6
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR.parent / "datasets" / "sparse" / "FrcSub"
DEFAULT_OUTPUT_PATH = BASE_DIR.parent / "datasets" / "sparse" / "FrcSub" / "knowledge_graph.json"

# Step 2
PHI_THRESHOLD = 0.35

# Step 3
MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS = 3
MIN_JOINT_SUCCESSES_FOR_CORRECTNESS = 2
CONFIDENCE_THRESHOLD = 0.6
COMBINED_SCORE_THRESHOLD = 0.6


@dataclass
class ExerciseRow:
    student_id: str
    exercise_id: str
    is_correct: int
    concept_ids: List[str]
    knowledge_concepts: List[str]


def _load_exercise_info(data_path: Path, data_mode: str = "sparse") -> Dict[str, Dict]:
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
    exclude_first_n_students: int = 0,
) -> Iterable[ExerciseRow]:
    recordings_path = data_path / "recordings.jsonl"
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings file not found: {recordings_path}")

    exercise_info = _load_exercise_info(data_path, data_mode)

    # Count total students for progress bar
    with recordings_path.open("r", encoding="utf-8") as _count_f:
        total_students = sum(1 for _ in _count_f)

    with recordings_path.open("r", encoding="utf-8") as f:
        iterator = tqdm(
            enumerate(f),
            total=total_students,
            desc="Loading data",
            unit="student",
        )
        for student_idx, line in iterator:
            if student_idx < exclude_first_n_students:
                continue

            record = json.loads(line.strip())
            student_id = str(record.get("student_id", ""))
            exercises_logs = record.get("exercises_logs", [])
            is_corrects = record.get("is_corrects", [])

            if train_split < 1.0:
                split_idx = int(len(exercises_logs) * train_split)
                exercises_logs = exercises_logs[:split_idx]
                is_corrects = is_corrects[:split_idx]

            for ex_id, is_correct in zip(exercises_logs, is_corrects):
                ex_id_str = str(ex_id)
                ex_info = exercise_info.get(ex_id_str, {})
                skill_ids = ex_info.get("skill_ids", [])
                skill_desc = ex_info.get("skill_desc", [])

                yield ExerciseRow(
                    student_id=student_id,
                    exercise_id=ex_id_str,
                    is_correct=int(is_correct),
                    concept_ids=[str(sid) for sid in skill_ids],
                    knowledge_concepts=skill_desc if skill_desc else ["Unknown Concept"],
                )


def _compute_abcd(
    student_exercises: Dict[str, List[ExerciseRow]],
) -> Dict[Tuple[str, str], Tuple[int, int, int, int]]:
    """Compute a, b, c, d counts for every ordered concept pair.

    For each student, group exercises by concept, then take the Cartesian
    product of exercise subsets for each concept pair (i, j):

      a = i wrong & j wrong
      b = i wrong & j correct
      c = i correct & j wrong
      d = i correct & j correct

    Returns:
        Dict mapping (concept_i, concept_j) -> (a, b, c, d)
    """
    counter: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for rows in tqdm(
        student_exercises.values(),
        desc="Step 1: Computing a/b/c/d",
        unit="student",
    ):
        # Group exercises by concept for this student
        by_concept: Dict[str, List[ExerciseRow]] = defaultdict(list)
        for row in rows:
            for cid in row.concept_ids:
                by_concept[cid].append(row)

        concepts = list(by_concept.keys())
        for idx_i in range(len(concepts)):
            for idx_j in range(len(concepts)):
                if idx_i == idx_j:
                    continue
                ci, cj = concepts[idx_i], concepts[idx_j]
                ex_i_list = by_concept[ci]
                ex_j_list = by_concept[cj]

                for ex_i in ex_i_list:
                    for ex_j in ex_j_list:
                        si, sj = ex_i.is_correct, ex_j.is_correct
                        if si == 0 and sj == 0:
                            counter[(ci, cj)][0] += 1  # a
                        elif si == 0 and sj == 1:
                            counter[(ci, cj)][1] += 1  # b
                        elif si == 1 and sj == 0:
                            counter[(ci, cj)][2] += 1  # c
                        else:
                            counter[(ci, cj)][3] += 1  # d

    return {k: tuple(v) for k, v in counter.items()}


def _phi(a: int, b: int, c: int, d: int) -> float:
    """Phi coefficient for a 2x2 contingency table."""
    denom = sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return 0.0 if denom == 0 else (a * d - b * c) / denom


def _evaluate_prerequisite_relation(
    a: int,
    b: int,
    c: int,
    d: int,
    prereq_success_count: int,
    phi_coef: float,
) -> Optional[Dict[str, Any]]:
    """Apply Step-3 filters to decide whether `prereq -> dependent` is valid.

    Returns a dict with correctness_score (combined score) and intermediate
    values, or None when any filter rejects the pair.
    """
    # 3.1 — Sample frequency check
    if prereq_success_count < MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS:
        return None
    if d < MIN_JOINT_SUCCESSES_FOR_CORRECTNESS:
        return None

    # 3.2 — Forward confidence  P(I|J) = d / (b + d)
    denom_fwd = b + d
    if denom_fwd == 0:
        return None
    forward_conf = d / denom_fwd
    if forward_conf < CONFIDENCE_THRESHOLD:
        return None

    # 3.2 — Backward confidence  P(¬J|¬I) = a / (a + b)
    denom_bwd = a + b
    if denom_bwd == 0:
        return None
    backward_conf = a / denom_bwd
    if backward_conf < CONFIDENCE_THRESHOLD:
        return None

    # 3.3 — Combined score (geometric mean)
    combined = sqrt(forward_conf * backward_conf)
    if combined < COMBINED_SCORE_THRESHOLD:
        return None

    return {
        "correctness_score": round(combined, 4),
        "forward_confidence": round(forward_conf, 4),
        "backward_confidence": round(backward_conf, 4),
        "phi_coefficient": round(phi_coef, 4),
        "a": a,
        "b_j": b,
        "c_i": c,
        "d_ij": d,
    }


def build_knowledge_graph(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    data_mode: str = "sparse",
    train_split: float = 1.0,
    exclude_first_n_students: int = 0,
) -> Dict:
    """Build a prerequisite knowledge graph following the three-step pipeline.

    Args:
        data_path: Directory containing recordings.jsonl and exercise_info.jsonl.
        output_path: Where the generated knowledge graph JSON is saved.
        data_mode: One of 'onehot', 'sparse', 'moderate'.
        train_split: Fraction of each student's history to use.
        exclude_first_n_students: Students to skip (avoid data leakage).

    Returns:
        The generated knowledge graph dictionary.
    """
    # ── Load data ──────────────────────────────────────────────────────────
    student_exercises: Dict[str, List[ExerciseRow]] = defaultdict(list)
    concept_names: Dict[str, str] = {}
    concept_success_students: Dict[str, Set[str]] = defaultdict(set)

    total_rows = 0
    for row in _load_rows(data_path, data_mode, train_split, exclude_first_n_students):
        total_rows += 1
        student_exercises[row.student_id].append(row)
        if row.is_correct:
            for cid in row.concept_ids:
                concept_success_students[cid].add(row.student_id)
        for idx, cid in enumerate(row.concept_ids):
            if cid not in concept_names:
                names = row.knowledge_concepts
                concept_names[cid] = (
                    names[idx] if idx < len(names) else names[0] if names else "Unknown Concept"
                )

    # ── Step 1:  a, b, c, d aggregation ────────────────────────────────────
    abcd = _compute_abcd(student_exercises)

    # ── Step 2 & 3:  Phi filter → direction evaluation ────────────────────
    edges: Dict[str, Dict[str, Dict]] = {}

    for (ci, cj), (a, b, c, d) in tqdm(abcd.items(), desc="Step 2&3: Filtering edges", unit="pair"):
        # Step 2
        phi_coef = _phi(a, b, c, d)
        if phi_coef < PHI_THRESHOLD:
            continue

        # Step 3
        result = _evaluate_prerequisite_relation(
            a=a,
            b=b,
            c=c,
            d=d,
            prereq_success_count=len(concept_success_students.get(ci, set())),
            phi_coef=phi_coef,
        )
        if result is not None:
            edges.setdefault(ci, {})[cj] = result

    # ── Build output ───────────────────────────────────────────────────────
    graph: Dict[str, Any] = {
        "meta": {
            "source": str(data_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_mode": data_mode,
            "train_split": train_split,
            "exclude_first_n_students": exclude_first_n_students,
            "total_rows": total_rows,
            "students": len(student_exercises),
            "phi_threshold": PHI_THRESHOLD,
            "correctness_config": {
                "min_success_students": MIN_SUCCESS_STUDENTS_FOR_CORRECTNESS,
                "min_joint_successes": MIN_JOINT_SUCCESSES_FOR_CORRECTNESS,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "combined_score_threshold": COMBINED_SCORE_THRESHOLD,
            },
        },
        "concepts": {},
    }

    total_edges = 0
    for cid in sorted(concept_names):
        out_edges = edges.get(cid, {})
        total_edges += len(out_edges)
        graph["concepts"][cid] = {
            "concept_id": cid,
            "name": concept_names[cid],
            "is_prerequisite_for": {
                dst: {
                    "name": concept_names[dst],
                    "correctness_score": v["correctness_score"],
                    "forward_confidence": v["forward_confidence"],
                    "backward_confidence": v["backward_confidence"],
                    "phi_coefficient": v["phi_coefficient"],
                    "a": v["a"],
                    "b_j": v["b_j"],
                    "c_i": v["c_i"],
                    "d_ij": v["d_ij"],
                }
                for dst, v in sorted(out_edges.items(), key=lambda x: -x[1]["correctness_score"])
            },
        }

    graph["meta"]["total_edges"] = total_edges

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prerequisite-style knowledge graph from exercise records (JSONL format)."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="sparse",
        choices=["onehot", "sparse", "moderate"],
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--exclude-first-n-students",
        type=int,
        default=0,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph = build_knowledge_graph(
        data_path=args.data_path,
        output_path=args.output_path,
        data_mode=args.data_mode,
        train_split=args.train_split,
        exclude_first_n_students=args.exclude_first_n_students,
    )
    print(
        f"Knowledge graph built for {len(graph['concepts'])} concepts "
        f"with data from {graph['meta']['students']} students. "
        f"Total edges: {graph['meta']['total_edges']}."
    )
    if args.exclude_first_n_students > 0:
        print(f"Excluded first {args.exclude_first_n_students} students to avoid data leakage.")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
