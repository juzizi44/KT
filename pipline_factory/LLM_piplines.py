from .piplines import Pipeline
from mylogger import Logger
from LLM_factory.GLM import GLM47, GLM4, GLM3
from LLM_factory.GPT import GPTChat, GPT4, GPT35
from LLM_factory.model import LLMModelBase
from evaluator_factory import LLMEvaluator
from utils import aggregate_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime

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
                 ):
        self.model_name = model_name
        self.train_data = train_data
        self.test_data = test_data
        self.extra_datas = extra_datas # list of pd
        self.logger = logger
        self.data_mode = data_mode
        self.llm =self.init_llm(model_name)
        self.fewshots_num = fewshots_num
        self.fewshots_strategy = fewshots_strategy
        self.eval_strategy = eval_strategy
        self.evaluator = LLMEvaluator(
            eval_strategy=eval_strategy,
            logger=logger,
            llm=self.llm,
            skip_post_explain=skip_post_explain,
        )
        self.test_num = test_num
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.knowledge_graph_path = knowledge_graph_path


    def init_llm(self, model_name):
        # initialize llm（GLM 仅云端 API：glm-4、glm-3-turbo）
        if model_name.startswith("glm"):
            if model_name == "glm-4.7":
                llm = GLM47()
            elif model_name == "glm-4":
                llm = GLM4()
            elif model_name == "glm-3-turbo":
                llm = GLM3()
            else:
                # 支持其他 glm 变体
                llm = GLMChat(model_name)
        elif model_name.startswith('gpt'):
            if model_name == 'gpt-4' or model_name == 'gpt-4-1106-preview' or model_name == 'gpt-4-32k':
                llm = GPT4()
            elif model_name == 'gpt-3.5-turbo':
                llm = GPT35()
            elif model_name.startswith('gpt-3.5-turbo'):
                # allow other gpt-3.5 variants such as gpt-3.5-turbo-0125
                llm = GPTChat(model_name)
            else:
                raise ValueError(f"Invalid gpt model name: {model_name}")
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        return llm


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

        # start evaluation, each iteration returns a dict of eval results for one student
        for idx, (i, student_info) in enumerate(students_to_process):
            student_id = str(student_info['student_id'])
            print(f"[EVAL {idx+1}/{len(students_to_process)}] student {student_id}", flush=True)
            self.logger.write(f"----------------Evaluating student {student_id}")
            test_rows = test_data[test_data['student_id'] == student_id]
            if test_rows.empty:
                self.logger.write(f"No test logs for student {student_id}, skipping.")
                continue
            test_exercises = test_rows['exercises_logs'].values[0]
            test_corrects = test_rows['is_corrects'].values[0]
            if len(test_exercises) == 0:
                self.logger.write(f"Student {student_id} has empty test split, skipping.")
                continue
            # key: exercise_id, value: dict of eval results for one exercise
            stu_eval_results = {}
            # extra_exercicse_info = extra_datas['exercise_info']
            flag = True
            # test_exercises is a nump array of exercise_ids
            for j, test_exercise_id in enumerate(test_exercises):
                print(f"  [EXERCISE {j+1}/{len(test_exercises)}] id={test_exercise_id}", flush=True)
                self.logger.write(f"*****Evaluating student {student_id} on exercise {test_exercise_id}")
                is_correct = test_corrects[j]
                # get exercise_info
                test_exercise_info = {'exercise_id': test_exercise_id, 'is_correct': is_correct}
                test_exercise_info.update(aggregate_data(test_exercise_id, extra_datas['exercise_info'], 'exercise'))

                prompts = self.llm.get_prompts(data_mode=data_mode)
                if fewshots_strategy == 'first' or fewshots_strategy == 'last':
                    if flag:
                        print(f"  [FEWSHOT] building {fewshots_num} fewshots (strategy={fewshots_strategy})...", flush=True)
                        fewshots = self.llm.create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts, self.dataset_name, self.knowledge_graph_path)
                        print(f"  [FEWSHOT] done, {len(fewshots)} fewshots built", flush=True)
                        flag = False
                else:
                    print(f"  [FEWSHOT] building {fewshots_num} fewshots (strategy={fewshots_strategy})...", flush=True)
                    fewshots = self.llm.create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts, self.dataset_name, self.knowledge_graph_path)
                    print(f"  [FEWSHOT] done, {len(fewshots)} fewshots built", flush=True)

                self.logger.write(f"Fewshots:\n{fewshots}")
                print(f"  [PREDICT] calling evaluator (strategy={eval_strategy})...", flush=True)
                exe_eval_results = self.evaluator.evaluate(student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode)
                print(f"  [PREDICT] done, prediction={exe_eval_results.get('prediction','?')}", flush=True)
                exe_eval_results.update({'is_correct': is_correct})
                stu_eval_results[test_exercise_id] = exe_eval_results
                # break # for debug only, only evaluate one exercise
                # exit()
            # collect each student's eval results
            # ...
            y_pre = []
            y_true = []
            for k, v in stu_eval_results.items(): # k is exercise_id, v is dict of eval results for one exercise
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
            eval_results[student_id] = stu_result

            # Save result immediately (real-time)
            self.logger.save_student_result(student_id, stu_result)

            total_y_pre.extend(y_pre)
            total_y_true.extend(y_true)
            # break # for debug only, only evaluate one student

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
    
