from .piplines import Pipeline
from mylogger import Logger
from LLM_factory.GLM import GLM47, GLM4, GLM3
from LLM_factory.GLM_v2 import GLM47V2, GLM4V2, GLM3V2
from LLM_factory.GLM_v3 import GLM47V3, GLM4V3, GLM3V3
from LLM_factory.GPT import GPTChat, GPT4, GPT35
from LLM_factory.GPT_v2 import GPTChatV2, GPT4V2, GPT35V2
from LLM_factory.GPT_v3 import GPTChatV3, GPT4V3, GPT35V3
from LLM_factory.model import LLMModelBase
from evaluator_factory import LLMEvaluator, LLMEvaluatorV2, LLMEvaluatorV3
from utils import aggregate_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

'''
For student in train_data:
    Get all the information of a student and exercises
    For each given exercise to predict in test_data:
        Select proper few-shots
        LLM creates analysis of student knowledge based on few shots
        LLM predicts student performance
        LLM gives explainations of prediction
        Collect evaluation results
    return student's evaluation results
'''

class LLMPipeline(Pipeline):
    def __init__(self,
                 model_name: str,
                 train_data,
                 test_data,
                 extra_datas,
                 logger: Logger,
                 data_mode: str,
                 fewshots_num: int,
                 fewshots_strategy: str,
                 eval_strategy: str,
                 test_num: int,
                 random_seed: int,
                 skip_post_explain: bool = False,
                 dataset_name: str = None,
                 knowledge_graph_path: str = None,
                 max_workers: int = 4,
                 version: str = 'v1',
                 ):
        self.model_name = model_name
        self.train_data = train_data
        self.test_data = test_data
        self.extra_datas = extra_datas # list of pd
        self.logger = logger
        self.data_mode = data_mode
        self.version = version
        self.llm = self.init_llm(model_name, version)
        self.fewshots_num = fewshots_num
        self.fewshots_strategy = fewshots_strategy
        self.eval_strategy = eval_strategy
        self.evaluator = self.init_evaluator(eval_strategy, logger, skip_post_explain)
        self.test_num = test_num
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.knowledge_graph_path = knowledge_graph_path
        self.max_workers = max_workers
        self._results_lock = threading.Lock()


    def init_llm(self, model_name, version='v1'):
        """
        Initialize LLM model.

        Args:
            model_name: Model name (e.g., 'glm-4.7', 'gpt-4')
            version: 'v1' for multi-turn, 'v2' for single-turn

        Returns:
            LLM model instance
        """
        if version == 'v2':
            return self._init_llm_v2(model_name)
        if version == 'v3':
            return self._init_llm_v3(model_name)
        return self._init_llm_v1(model_name)

    def _init_llm_v1(self, model_name):
        """Initialize V1 LLM (multi-turn)."""
        if model_name.startswith("glm"):
            if model_name == "glm-4.7":
                llm = GLM47()
            elif model_name == "glm-4":
                llm = GLM4()
            elif model_name == "glm-3-turbo":
                llm = GLM3()
            else:
                from LLM_factory.GLM import GLMChat
                llm = GLMChat(model_name)
        elif model_name.startswith('gpt'):
            if model_name == 'gpt-4' or model_name == 'gpt-4-1106-preview' or model_name == 'gpt-4-32k':
                llm = GPT4()
            elif model_name == 'gpt-3.5-turbo':
                llm = GPT35()
            elif model_name.startswith('gpt-3.5-turbo'):
                llm = GPTChat(model_name)
            else:
                raise ValueError(f"Invalid gpt model name: {model_name}")
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        return llm

    def _init_llm_v2(self, model_name):
        """Initialize V2 LLM (single-turn)."""
        if model_name.startswith("glm"):
            if model_name == "glm-4.7":
                llm = GLM47V2()
            elif model_name == "glm-4":
                llm = GLM4V2()
            elif model_name == "glm-3-turbo":
                llm = GLM3V2()
            else:
                from LLM_factory.GLM_v2 import GLMChatV2
                llm = GLMChatV2(model_name)
        elif model_name.startswith('gpt'):
            if model_name == 'gpt-4' or model_name == 'gpt-4-1106-preview' or model_name == 'gpt-4-32k':
                llm = GPT4V2()
            elif model_name == 'gpt-3.5-turbo':
                llm = GPT35V2()
            elif model_name.startswith('gpt-3.5-turbo'):
                llm = GPTChatV2(model_name)
            else:
                raise ValueError(f"Invalid gpt model name: {model_name}")
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        return llm

    def _init_llm_v3(self, model_name):
        """Initialize V3 LLM (producer-critic-judge)."""
        if model_name.startswith('glm'):
            if model_name == 'glm-4.7':
                llm = GLM47V3()
            elif model_name == 'glm-4':
                llm = GLM4V3()
            elif model_name == 'glm-3-turbo':
                llm = GLM3V3()
            else:
                from LLM_factory.GLM_v3 import GLMChatV3
                llm = GLMChatV3(model_name)
        elif model_name.startswith('gpt'):
            if model_name in ('gpt-4', 'gpt-4-1106-preview', 'gpt-4-32k'):
                llm = GPT4V3()
            elif model_name == 'gpt-3.5-turbo':
                llm = GPT35V3()
            elif model_name.startswith('gpt-3.5-turbo'):
                llm = GPTChatV3(model_name)
            else:
                raise ValueError(f"Invalid gpt model name: {model_name}")
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        return llm

    def init_evaluator(self, eval_strategy, logger, skip_post_explain):
        """Initialize evaluator based on version."""
        if self.version == 'v2':
            return LLMEvaluatorV2(
                eval_strategy=eval_strategy,
                logger=logger,
                llm=self.llm,
            )
        if self.version == 'v3':
            return LLMEvaluatorV3(
                eval_strategy=eval_strategy,
                logger=logger,
                llm=self.llm,
                skip_post_explain=skip_post_explain,
            )
        return LLMEvaluator(
            eval_strategy=eval_strategy,
            logger=logger,
            llm=self.llm,
            skip_post_explain=skip_post_explain,
        )


    def evaluate(self, train_data, test_data, data_mode, extra_datas, fewshots_num, fewshots_strategy, eval_strategy):
        # train_data has n_student lines, each line has one student's logs, extra_datas is a list of pd, each pd has side information of exercises
        # test_data has n_test_student lines, each line has one student's logs

        # initialize eval results
        # key: student_id, value: dict of eval results for one student
        eval_results = {}
        # randomly select number of lines in train_data to evaluate, but only keep students with non-empty test logs
        valid_test_mask = test_data['exercises_logs'].apply(lambda logs: hasattr(logs, '__len__') and len(logs) > 0)
        valid_test_students = set(test_data[valid_test_mask]['student_id'])
        eligible_train = train_data[train_data['student_id'].isin(valid_test_students)]

        if eligible_train.empty:
            self.logger.write("No students with non-empty test logs are available for evaluation.")
            return eval_results

        if self.test_num != -1:
            if len(eligible_train) < self.test_num:
                self.logger.write(f"Requested {self.test_num} students but only {len(eligible_train)} have test logs. Evaluating all available students.")
                selected_train = eligible_train
            else:
                selected_train = eligible_train.sample(n=self.test_num, random_state=self.random_seed)
        else:
            selected_train = eligible_train
            self.logger.write(f"Evaluate all {len(selected_train)} students")

        # Check for resume - get completed students
        completed_students = self.logger.get_completed_students()
        if completed_students:
            self.logger.write(f"[RESUME] Skipping {len(completed_students)} already completed students")

        # Load previous results if resuming
        prev_results = self.logger.get_results()
        if "student_results" in prev_results:
            eval_results = {k: v for k, v in prev_results["student_results"].items()}

        total_y_pre = []
        total_y_true = []

        # Collect metrics from previously completed students
        for student_id, result in eval_results.items():
            if "eval_results" in result:
                for ex_id, ex_result in result["eval_results"].items():
                    if "prediction" in ex_result and "is_correct" in ex_result:
                        total_y_pre.append(int(ex_result["prediction"]))
                        total_y_true.append(int(ex_result["is_correct"]))

        students_to_process = []
        for i, student_info in selected_train.iterrows():
            student_id = str(student_info['student_id'])
            if student_id in completed_students:
                print(f"[SKIP] student {student_id} (already completed)", flush=True)
                continue
            students_to_process.append((i, student_info))

        self.logger.write(f"[INFO] Total students to process: {len(students_to_process)} / {len(selected_train)}")
        self.logger.write(f"[INFO] Using {self.max_workers} workers for concurrent student processing")

        # start evaluation with thread pool
        if self.max_workers <= 1:
            # sequential fallback
            for idx, (i, student_info) in enumerate(students_to_process):
                student_id, stu_result, y_pre, y_true = self._evaluate_one_student(
                    idx, len(students_to_process), i, student_info, test_data,
                    extra_datas, data_mode, fewshots_num, fewshots_strategy, eval_strategy
                )
                if stu_result is not None:
                    eval_results[student_id] = stu_result
                    total_y_pre.extend(y_pre)
                    total_y_true.extend(y_true)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self._evaluate_one_student,
                        idx, len(students_to_process), i, student_info, test_data,
                        extra_datas, data_mode, fewshots_num, fewshots_strategy, eval_strategy,
                    ): idx
                    for idx, (i, student_info) in enumerate(students_to_process)
                }
                for future in as_completed(future_to_idx):
                    try:
                        student_id, stu_result, y_pre, y_true = future.result()
                    except Exception as exc:
                        self.logger.write(f"[ERROR] Student evaluation failed: {exc}")
                        continue
                    if stu_result is not None:
                        with self._results_lock:
                            eval_results[student_id] = stu_result
                            total_y_pre.extend(y_pre)
                            total_y_true.extend(y_true)

        # add final eval results to eval_results, return eval_results
        if len(total_y_true) == 0:
            self.logger.write("No predictions generated; skipping aggregate metrics.")
            return eval_results

        final_acc = accuracy_score(total_y_true, total_y_pre)
        final_precision = precision_score(total_y_true, total_y_pre)
        final_recall = recall_score(total_y_true, total_y_pre)
        final_f1 = f1_score(total_y_true, total_y_pre)
        self.logger.write(f"Total test student: {len(eval_results)}, total test count: {len(total_y_true)}")
        self.logger.write(f"Final accuracy: {final_acc}, precision: {final_precision}, recall: {final_recall}, f1: {final_f1}")

        # Save final results
        final_metrics = {
            "total_students": len(eval_results),
            "total_predictions": len(total_y_true),
            "accuracy": final_acc,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1
        }
        self.logger.save_final_results(final_metrics)

        return eval_results
    

    def _evaluate_one_student(self, idx, total, i, student_info, test_data,
                               extra_datas, data_mode, fewshots_num, fewshots_strategy, eval_strategy):
        """Process a single student's evaluation. Returns (student_id, stu_result, y_pre, y_true)."""
        student_id = str(student_info['student_id'])
        print(f"[EVAL {idx+1}/{total}] student {student_id}", flush=True)
        self.logger.write(f"----------------Evaluating student {student_id}")
        test_rows = test_data[test_data['student_id'] == student_id]
        if test_rows.empty:
            self.logger.write(f"No test logs for student {student_id}, skipping.")
            return student_id, None, [], []
        test_exercises = test_rows['exercises_logs'].values[0]
        test_corrects = test_rows['is_corrects'].values[0]
        if len(test_exercises) == 0:
            self.logger.write(f"Student {student_id} has empty test split, skipping.")
            return student_id, None, [], []

        stu_eval_results = {}
        flag = True
        for j, test_exercise_id in enumerate(test_exercises):
            print(f"  [EVAL {idx+1}/{total}] student {student_id} exercise {j+1}/{len(test_exercises)} id={test_exercise_id}", flush=True)
            self.logger.write(f"*****Evaluating student {student_id} on exercise {test_exercise_id}")
            is_correct = test_corrects[j]
            test_exercise_info = {'exercise_id': test_exercise_id, 'is_correct': is_correct}
            test_exercise_info.update(aggregate_data(test_exercise_id, extra_datas['exercise_info'], 'exercise'))

            prompts = self.llm.get_prompts(data_mode=data_mode)
            if fewshots_strategy == 'first' or fewshots_strategy == 'last':
                if flag:
                    fewshots = self.llm.create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts, self.dataset_name, self.knowledge_graph_path)
                    flag = False
            else:
                fewshots = self.llm.create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts, self.dataset_name, self.knowledge_graph_path)

            self.logger.write(f"Fewshots:\n{fewshots}")
            exe_eval_results = self.evaluator.evaluate(student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode)
            exe_eval_results.update({'is_correct': is_correct})
            stu_eval_results[test_exercise_id] = exe_eval_results

        y_pre = []
        y_true = []
        for k, v in stu_eval_results.items():
            y_pre.append(int(v['prediction']))
            y_true.append(int(v['is_correct']))
        stu_acc = accuracy_score(y_true, y_pre)
        stu_precision = precision_score(y_true, y_pre)
        stu_recall = recall_score(y_true, y_pre)
        stu_f1 = f1_score(y_true, y_pre)
        stu_test_count = len(y_true)
        self.logger.write(f"y_true: {y_true}, y_pre: {y_pre}")
        self.logger.write(f"Student {student_id}, len: {stu_test_count}, acc: {stu_acc}, precision: {stu_precision}, recall: {stu_recall}, f1: {stu_f1}")
        stu_result = {'student_id': student_id, 'accuracy': stu_acc, 'precision': stu_precision, 'recall': stu_recall, 'f1': stu_f1, 'test_count': stu_test_count, 'eval_results': stu_eval_results}

        # Save result immediately (real-time, thread-safe via logger lock)
        self.logger.save_student_result(student_id, stu_result)

        return student_id, stu_result, y_pre, y_true

    def run(self):
        # LLMs only need to evaluate
        eval_results = self.evaluate(self.train_data,
                                     self.test_data,
                                     self.data_mode,
                                     self.extra_datas,
                                     self.fewshots_num,
                                     self.fewshots_strategy,
                                     self.eval_strategy
                                     )
        self.display_results(eval_results, self.logger)
    
    def display_results(self, eval_results, logger):
        # display eval results and save to loggers
        # write end time to logger
        logger.write(f"End time: {datetime.datetime.now()}")
        logger.write(f"Eval results:\n{eval_results}")
    
