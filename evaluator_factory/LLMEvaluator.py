from .Evaluator import EvaluatorBase
from mylogger import Logger
from LLM_factory.model import LLMModelBase
from utils import check_response_format
import random


def _coerce_response_text(response):
    """Normalize different LLM client return types into a plain string."""
    if isinstance(response, list):
        filtered = [item for item in response if isinstance(item, str)]
        return "\n".join(filtered).strip()
    if isinstance(response, str):
        return response.strip()
    if response is None:
        return ""
    return str(response).strip()


class LLMEvaluator(EvaluatorBase):
    def __init__(
        self,
        eval_strategy: str,
        logger: Logger,
        llm: LLMModelBase,
        skip_post_explain: bool = False,
    ):
        self.eval_strategy = eval_strategy
        self.logger = logger
        self.llm = llm
        self.skip_post_explain = skip_post_explain

    def evaluate(self, student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode):
        eval_result = {'student_id': student_info['student_id'], 
                'pre_exe_id': test_exercise_info['exercise_id']
                }
        if eval_strategy =='simple':
            # generate fewshots and prompts to predict
            user_data = self.llm.create_user_data(student_info, test_exercise_info, extra_datas, data_mode=data_mode)
            prediction_response = self.llm.generate_prediction(fewshots, user_data, prompts)
            try:
                eval_result['prediction'] = check_response_format(prediction_response)
            except ValueError as e:
                # if the repsonse format is not correct, generate again, if again, then randomly choose 0 or 1
                response_tmp = self.llm.generate_prediction(fewshots, user_data, prompts)
                try:
                    eval_result['prediction'] = check_response_format(response_tmp)
                except ValueError as e:
                    self.logger.write(f"Error in prediction response format: at student {student_info['student_id']}, exercise {test_exercise_info['exercise_id']}")
                    self.logger.write(f"response: {prediction_response}")
                    eval_result['prediction'] = '0' if random.random() < 0.5 else '1'
            # generate explaination
            # delete previous "<Exercise to Predict>"
            # add new prediction to fewshots
            user_data = user_data.replace("<Exercise to Predict>", "")
            user_data += ("is_correct: " + eval_result['prediction'] + "\n")
            fewshots.append(user_data)
            if self.skip_post_explain:
                eval_result['explaination'] = ""
            else:
                eval_result['explaination'] = self.llm.generate_explaination(
                    fewshots, eval_result['prediction'], prompts
                )

        elif eval_strategy == 'analysis':
            user_data = self.llm.create_user_data(student_info, test_exercise_info, extra_datas, data_mode=data_mode)
            # include the current exercise context when requesting an analysis
            analysis_context = list(fewshots) if fewshots else []
            analysis_context.append(user_data)

            analysis_text = ""
            try:
                analysis_response = self.llm.generate_analysis(analysis_context, prompts)
                analysis_text = _coerce_response_text(analysis_response)
            except Exception as exc:  # pragma: no cover - fallback for proxy errors
                self.logger.write(
                    f"Error generating analysis for student {student_info['student_id']}, "
                    f"exercise {test_exercise_info['exercise_id']}: {exc}"
                )
            if not analysis_text:
                self.logger.write(
                    f"Empty analysis response for student {student_info['student_id']}, "
                    f"exercise {test_exercise_info['exercise_id']}. Continuing without analysis context."
                )

            eval_result['analysis'] = analysis_text if analysis_text else "Analysis not generated."
            analysis_block = ""
            if analysis_text:
                analysis_block = "<Analysis Summary>\n" + analysis_text + "\n</Analysis Summary>\n"

            if analysis_block and "<Output Predicted is_correct>" in user_data:
                user_data_with_analysis = user_data.replace(
                    "<Output Predicted is_correct>\n",
                    analysis_block + "<Output Predicted is_correct>\n",
                    1,
                )
            elif analysis_block:
                user_data_with_analysis = user_data + analysis_block
            else:
                user_data_with_analysis = user_data

            prediction_response = self.llm.generate_prediction(fewshots, user_data_with_analysis, prompts)
            try:
                eval_result['prediction'] = check_response_format(prediction_response)
            except ValueError as e:
                response_tmp = self.llm.generate_prediction(fewshots, user_data_with_analysis, prompts)
                try:
                    eval_result['prediction'] = check_response_format(response_tmp)
                except ValueError as e:
                    self.logger.write(f"Error in prediction response format: at student {student_info['student_id']}, exercise {test_exercise_info['exercise_id']}")
                    self.logger.write(f"response: {prediction_response}")
                    eval_result['prediction'] = '0' if random.random() < 0.5 else '1'

            # generate explanation using the analysis-augmented fewshot log
            user_data_with_analysis = user_data_with_analysis.replace("<Exercise to Predict>", "")
            user_data_with_analysis += ("is_correct: " + eval_result['prediction'] + "\n")
            fewshots.append(user_data_with_analysis)
            if self.skip_post_explain:
                eval_result['explaination'] = ""
            else:
                eval_result['explaination'] = self.llm.generate_explaination(
                    fewshots, eval_result['prediction'], prompts
                )
        elif eval_strategy =='self_correct':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid eval_strategy: {eval_strategy}")
        
        # self.logger.write(f"Evaluation result: {eval_result}")
        return eval_result
