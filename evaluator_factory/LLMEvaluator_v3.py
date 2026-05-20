from typing import Dict
import random

from .Evaluator import EvaluatorBase
from .LLMEvaluator import LLMEvaluator
from mylogger import Logger
from LLM_factory.model import LLMModelBase
from LLM_factory.prompt_factory import generic_get_prompts as generic_get_prompts_v1
from LLM_factory.prompt_factory_v3 import (
    create_critic_messages,
    create_judge_messages,
    parse_critic_response,
    parse_judge_response,
    validate_judge_parsed_response,
)
from LLM_factory.GLM import GLMChat, GLM4, GLM3
from LLM_factory.GPT import GPTChat, GPT4, GPT35
from LLM_factory.DeepSeek import DeepSeekChat, DeepSeekFlash


def _model_family(model_name: str) -> str:
    if model_name.startswith('glm') or model_name.startswith('z-ai/'):
        return 'glm'
    return 'gpt'


def _build_v1_llm(model_name: str) -> LLMModelBase:
    if model_name.startswith('glm') or model_name.startswith('z-ai/'):
        if model_name in {'glm-4.7', 'z-ai/glm-4.7'}:
            return GLMChat('z-ai/glm-4.7')
        if model_name == 'glm-4':
            return GLM4()
        if model_name == 'glm-3-turbo':
            return GLM3()
        return GLMChat(model_name)
    if model_name.startswith('gpt'):
        if model_name in {'gpt-4', 'gpt-4-1106-preview', 'gpt-4-32k'}:
            return GPT4()
        if model_name == 'gpt-3.5-turbo':
            return GPT35()
        if model_name.startswith('gpt-3.5-turbo'):
            return GPTChat(model_name)
        raise ValueError(f'Invalid gpt model name: {model_name}')
    if model_name.startswith('deepseek'):
        if model_name == 'deepseek-v4-flash':
            return DeepSeekFlash()
        return DeepSeekChat(model_name)
    raise ValueError(f'Invalid model name: {model_name}')


def _coerce_text(response):
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, list):
        return '\n'.join(str(item) for item in response if item is not None).strip()
    if response is None:
        return ''
    return str(response).strip()


class LLMEvaluatorV3(EvaluatorBase):
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
        self.producer_llm = _build_v1_llm(llm.name)
        self.producer_evaluator = LLMEvaluator(
            eval_strategy=eval_strategy,
            logger=logger,
            llm=self.producer_llm,
            skip_post_explain=skip_post_explain,
        )

    def evaluate(self, student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode):
        eval_result = {
            'student_id': student_info['student_id'],
            'pre_exe_id': test_exercise_info['exercise_id'],
        }

        user_data = self.llm.create_user_data(student_info, test_exercise_info, extra_datas, data_mode=data_mode)

        # ========== Stage 1: Producer ==========
        print(f"    [V3 Stage 1] Producer: generating initial prediction...", flush=True)
        producer_prompts = generic_get_prompts_v1(_model_family(self.producer_llm.name), data_mode)
        producer_result = self.producer_evaluator.evaluate(
            student_info,
            test_exercise_info,
            extra_datas,
            eval_strategy,
            list(fewshots) if fewshots else [],
            producer_prompts,
            data_mode,
        )
        eval_result['producer_result'] = producer_result
        eval_result['producer_prediction'] = producer_result.get('prediction', '')
        print(f"    [V3 Stage 1] Producer done: prediction={producer_result.get('prediction', 'N/A')}", flush=True)

        # ========== Stage 2: Critic ==========
        print(f"    [V3 Stage 2] Critic: analyzing producer output...", flush=True)
        critic_messages = create_critic_messages(fewshots, user_data, producer_result, prompts)
        critic_raw = self.llm.generate_chat(critic_messages, max_tokens=2048, temperature=0.2, num_comps=1)
        critic_text = _coerce_text(critic_raw)
        critic_result = parse_critic_response(critic_text)
        eval_result['critic_result'] = critic_result
        eval_result['critic_raw_response'] = critic_text
        print(f"    [V3 Stage 2] Critic done: verdict={critic_result.get('verdict', 'N/A')}, "
              f"alt_prediction={critic_result.get('alternative_prediction', 'N/A')}", flush=True)

        # ========== Stage 3: Judge ==========
        print(f"    [V3 Stage 3] Judge: making final decision...", flush=True)
        judge_messages = create_judge_messages(fewshots, user_data, producer_result, critic_result, prompts)
        judge_raw = self.llm.generate_chat(judge_messages, max_tokens=2048, temperature=0.2, num_comps=1)
        judge_text = _coerce_text(judge_raw)
        judge_result = parse_judge_response(judge_text)
        eval_result['judge_result'] = judge_result
        eval_result['judge_raw_response'] = judge_text
        print(f"    [V3 Stage 3] Judge done: final_prediction={judge_result.get('final_prediction', 'N/A')}, "
              f"confidence={judge_result.get('confidence', 'N/A')}", flush=True)

        # ========== Stage 4: Result Integration & Fallback ==========
        print(f"    [V3 Stage 4] Result Integration: validating and finalizing...", flush=True)
        final_prediction = judge_result.get('final_prediction')
        fallback_used = False

        if not validate_judge_parsed_response(judge_result):
            self.logger.write(
                f"[V3 WARN] Invalid judge prediction for student {student_info['student_id']}, exercise {test_exercise_info['exercise_id']}. Falling back to producer prediction."
            )
            print(f"    [V3 Stage 4] WARNING: Invalid judge response, falling back to producer prediction", flush=True)
            final_prediction = producer_result.get('prediction', '')
            fallback_used = True

        if final_prediction not in {'0', '1'}:
            print(f"    [V3 Stage 4] WARNING: Invalid prediction '{final_prediction}', using random fallback", flush=True)
            final_prediction = '0' if random.random() < 0.5 else '1'
            fallback_used = True

        print(f"    [V3 Stage 4] Final result: prediction={final_prediction}, fallback_used={fallback_used}", flush=True)

        eval_result['prediction'] = final_prediction
        eval_result['explaination'] = (
            judge_result.get('decision_reason')
            or critic_result.get('evidence')
            or producer_result.get('explaination', '')
        )
        eval_result['analysis'] = producer_result.get('analysis', '')
        eval_result['producer_explaination'] = producer_result.get('explaination', '')
        eval_result['critic_verdict'] = critic_result.get('verdict', '')
        eval_result['critic_explaination'] = critic_result.get('flaws', '')
        eval_result['judge_explaination'] = judge_result.get('decision_reason', '')

        return eval_result
