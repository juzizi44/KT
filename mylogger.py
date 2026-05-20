import os
import time
import json
import threading
from typing import Dict, Any, Optional, Set


class Logger:
    def __init__(self, args):
        self.log_path = args.log_path
        self.args = args
        self._lock = threading.Lock()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Extract graph config name from path if provided
        graph_config = getattr(args, 'graph_config_name', None)
        if graph_config is None and hasattr(args, 'knowledge_graph_path') and args.knowledge_graph_path:
            # Auto-detect from path like: knowledge_graph_correctness1.0_sequence0.0.json
            filename = os.path.basename(args.knowledge_graph_path)
            if filename.startswith('knowledge_graph_') and filename.endswith('.json'):
                graph_config = filename[len('knowledge_graph_'):-len('.json')]
            else:
                graph_config = filename.replace('.json', '')

        self.graph_config = graph_config
        self.version = getattr(args, 'version', 'v1')
        self.version_suffix = '_v3' if self.version == 'v3' else ''

        # Create log directory (now includes fewshot_strategy)
        log_dir = os.path.join(args.log_path, args.model_name, args.data_mode, args.fewshot_strategy, args.dataset_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create result directory (now includes fewshot_strategy)
        result_dir = os.path.join("results", args.model_name, args.data_mode, args.fewshot_strategy, args.dataset_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Build filename suffix (include graph config for knowledge_graph strategy)
        config_suffix = ""
        if args.fewshot_strategy == 'knowledge_graph' and self.graph_config:
            config_suffix = f"_{self.graph_config}"

        # Log file path (use consistent name for resume)
        self.log_file = os.path.join(log_dir,
            f"{args.model_type}_{args.model_name}_fsn{args.fewshot_num}_fss{args.fewshot_strategy}{config_suffix}_es{args.eval_strategy}{self.version_suffix}.txt")

        # JSON result file path
        self.result_file = os.path.join(result_dir,
            f"{args.model_type}_{args.model_name}_fsn{args.fewshot_num}_fss{args.fewshot_strategy}{config_suffix}_es{args.eval_strategy}{self.version_suffix}.json")

        # Checkpoint file path
        self.checkpoint_file = os.path.join(result_dir,
            f"{args.model_type}_{args.model_name}_fsn{args.fewshot_num}_fss{args.fewshot_strategy}{config_suffix}_es{args.eval_strategy}{self.version_suffix}_checkpoint.json")

        # Initialize or load existing results
        self.results: Dict[str, Any] = {}
        self.completed_students: Set[str] = set()
        self._load_checkpoint()

        # Write header if new run
        if not self._is_resume:
            self.write(f"Log created at {timestamp}")
            self.write(f"Args: {args}")
            if self.graph_config:
                self.write(f"Knowledge Graph Config: {self.graph_config}")
        else:
            self.write(f"[RESUME] Resuming from checkpoint at {timestamp}")
            self.write(f"[RESUME] Already completed {len(self.completed_students)} students")

    def _load_checkpoint(self):
        """Load existing results and checkpoint if they exist."""
        self._is_resume = False

        # Load checkpoint file
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                    self.completed_students = set(checkpoint.get("completed_students", []))
                    self.results = checkpoint.get("results", {})
                    self._is_resume = len(self.completed_students) > 0
            except (json.JSONDecodeError, IOError):
                self.completed_students = set()
                self.results = {}

        # Also load result file if exists
        if os.path.exists(self.result_file):
            try:
                with open(self.result_file, "r", encoding="utf-8") as f:
                    loaded_results = json.load(f)
                    if loaded_results:
                        self.results = loaded_results
                        # Extract completed students from results
                        if "student_results" in loaded_results:
                            self.completed_students = set(loaded_results["student_results"].keys())
                        self._is_resume = len(self.completed_students) > 0
            except (json.JSONDecodeError, IOError):
                pass

    @property
    def is_resume(self) -> bool:
        """Check if this is a resumed run."""
        return self._is_resume

    def get_completed_students(self) -> Set[str]:
        """Get set of already completed student IDs (thread-safe)."""
        with self._lock:
            return self.completed_students.copy()

    def is_student_completed(self, student_id: str) -> bool:
        """Check if a student has already been evaluated (thread-safe)."""
        with self._lock:
            return str(student_id) in self.completed_students

    def write(self, log_message: str, print_log=True):
        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                log_line = f"{log_message}\n"
                f.write(log_line)
                if print_log:
                    print("To log:", log_line)
                f.flush()

    def flush(self):
        """Flush the log file to disk"""
        open(self.log_file, "a").close()

    def write_dict(self, log_dict: dict, print_log=True):
        for key, value in log_dict.items():
            self.write(f"{key}: {value}", print_log=print_log)

    def save_student_result(self, student_id: str, result: Dict[str, Any]):
        """Save a single student's result immediately (thread-safe)."""
        student_id = str(student_id)

        with self._lock:
            # Update in-memory data
            self.completed_students.add(student_id)
            if "student_results" not in self.results:
                self.results["student_results"] = {}
            self.results["student_results"][student_id] = result

            # Save result file (real-time)
            self._save_result_file()

            # Save checkpoint file (real-time)
            self._save_checkpoint()

        # Write to log (already acquires lock internally)
        self.write(f"[SAVED] Student {student_id} result saved")

    def _save_result_file(self):
        """Save results to JSON file."""
        with open(self.result_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def _save_checkpoint(self):
        """Save checkpoint to JSON file."""
        checkpoint = {
            "completed_students": list(self.completed_students),
            "results": self.results,
            "last_updated": time.strftime("%Y%m%d_%H%M%S", time.localtime())
        }
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def save_final_results(self, final_metrics: Dict[str, Any]):
        """Save final aggregated results (thread-safe)."""
        with self._lock:
            self.results["final_metrics"] = final_metrics
            self.results["config"] = {
                "model_name": self.args.model_name,
                "data_mode": self.args.data_mode,
                "dataset_name": self.args.dataset_name,
                "fewshot_num": self.args.fewshot_num,
                "fewshot_strategy": self.args.fewshot_strategy,
                "eval_strategy": self.args.eval_strategy,
                "version": getattr(self.args, "version", None),
                "test_num": self.args.test_num,
                "knowledge_graph_config": self.graph_config,
            }
            self._save_result_file()

        # Remove checkpoint file after successful completion
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            self.write("[CLEANUP] Checkpoint file removed after successful completion")

    def get_results(self) -> Dict[str, Any]:
        """Get all results (thread-safe)."""
        with self._lock:
            return self.results.copy()
