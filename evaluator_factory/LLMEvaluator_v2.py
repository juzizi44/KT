"""
V2 LLM Evaluator - Single-turn evaluation
单轮调用完成所有分析和预测
"""

from .Evaluator import EvaluatorBase
from mylogger import Logger
from LLM_factory.model import LLMModelBase
from LLM_factory.prompt_factory_v2 import validate_parsed_response
import random


class LLMEvaluatorV2(EvaluatorBase):
    """
    V2 Evaluator - Single-turn structured output.

    一次 API 调用完成：
    1. 历史题目递进分析
    2. 知识状态更新
    3. 预测
    4. 解释
    """

    def __init__(
        self,
        eval_strategy: str,
        logger: Logger,
        llm: LLMModelBase,
        max_retries: int = 2,
    ):
        self.eval_strategy = eval_strategy
        self.logger = logger
        self.llm = llm
        self.max_retries = max_retries

    def evaluate(
        self,
        student_info,
        test_exercise_info,
        extra_datas,
        eval_strategy,
        fewshots,
        prompts,
        data_mode
    ):
        """
        V2 evaluation: single-turn prediction.

        Args:
            student_info: Student information
            test_exercise_info: Test exercise information
            extra_datas: Extra data
            eval_strategy: Evaluation strategy (for V2, always 'single_turn')
            fewshots: List of few-shot strings (raw data, no explanations)
            prompts: V2 prompts dictionary
            data_mode: Data mode

        Returns:
            Evaluation result dictionary
        """
        eval_result = {
            'student_id': student_info['student_id'],
            'pre_exe_id': test_exercise_info['exercise_id']
        }

        # Create user data
        user_data = self.llm.create_user_data(
            student_info, test_exercise_info, extra_datas, data_mode=data_mode
        )

        # Single-turn prediction
        attempt = 0
        parsed = None

        while attempt <= self.max_retries:
            try:
                parsed = self.llm.generate_single_turn_prediction(
                    fewshots, user_data, prompts
                )

                # Validate prediction format
                if parsed['prediction'] in ['0', '1']:
                    break
                else:
                    self.logger.write(
                        f"[V2 RETRY] Invalid prediction format: {parsed['prediction']}, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
            except Exception as e:
                self.logger.write(
                    f"[V2 ERROR] Single-turn prediction failed: {e}, "
                    f"attempt {attempt + 1}/{self.max_retries}"
                )

            attempt += 1

        # Fallback if all attempts failed
        if parsed is None or parsed['prediction'] not in ['0', '1']:
            self.logger.write(
                f"[V2 FALLBACK] Random prediction for student {student_info['student_id']}, "
                f"exercise {test_exercise_info['exercise_id']}"
            )
            eval_result['prediction'] = '0' if random.random() < 0.5 else '1'
            eval_result['history_analyses'] = []
            eval_result['final_knowledge_state'] = ''
            eval_result['target_analysis'] = ''
            eval_result['explanation'] = 'Prediction failed, using random fallback.'
            eval_result['raw_response'] = ''
        else:
            eval_result['prediction'] = parsed['prediction']
            eval_result['history_analyses'] = parsed.get('history_analyses', [])
            eval_result['final_knowledge_state'] = parsed.get('final_knowledge_state', '')
            eval_result['target_analysis'] = parsed.get('target_analysis', '')
            eval_result['explanation'] = parsed.get('explanation', '')
            eval_result['raw_response'] = parsed.get('raw_response', '')

        return eval_result
