"""
V2 GLM Implementation - Single-turn structured output
单轮调用完成所有分析和预测
"""

from typing import List, Union, Dict
import dataclasses
import time

from LLM_factory.model import Message, LLMModelBase
from LLM_factory.prompt_factory_v2 import (
    generic_get_prompts_v2,
    parse_single_turn_response,
    validate_parsed_response
)
from LLM_factory.fewshot_generator_v2 import (
    generic_create_user_data_v2,
    generic_create_fewshots_v2
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import httpx as _httpx

# 智谱配置（通过 OpenRouter）
ZHIPU_API_KEY = "***REMOVED***"
ZHIPU_BASE_URL = "https://openrouter.ai/api/v1"
GLM_MODEL_NAME = "z-ai/glm-4.7"


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def glm_chat_v2(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.2,
    num_comps=1,
) -> Union[List[str], str]:
    """
    GLM chat API call for V2.
    """
    print(f"  [GLM V2 API] calling {model}, msgs={len(messages)}, max_tokens={max_tokens}", flush=True)
    with _httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{ZHIPU_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {ZHIPU_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [dataclasses.asdict(message) for message in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    print(f"  [GLM V2 API] response received, status={resp.status_code}", flush=True)

    time.sleep(1)
    choices = data["choices"]
    if num_comps == 1:
        return choices[0]["message"]["content"]
    return [c["message"]["content"] for c in choices]


class GLMChatV2(LLMModelBase):
    """
    V2 GLM Chat - Single-turn structured output.
    一次调用完成：历史分析 + 知识状态更新 + 预测 + 解释
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.is_chat = True

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 8192,
        temperature: float = 0.2,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        return glm_chat_v2(self.name, messages, max_tokens, temperature, num_comps)

    def create_single_turn_messages(
        self,
        fewshots: List[str],
        user_data: str,
        prompts: dict
    ) -> List[Message]:
        """
        Create messages for single-turn prediction.
        """
        fewshots_str = '\n'.join(fewshots) if fewshots else 'No historical exercises.'

        messages = [
            Message(
                role='system',
                content=prompts['sys_instr']
            ),
            Message(
                role='user',
                content=prompts['user_instr'].format(
                    fewshots=fewshots_str,
                    exercise_to_predict=user_data
                )
            )
        ]
        return messages

    def generate_single_turn_prediction(
        self,
        fewshots: List[str],
        user_data: str,
        prompts: dict,
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> Dict:
        """
        Single-turn generation: analysis + prediction + explanation in ONE API call.

        Returns:
            Dictionary containing:
            - history_analyses: List of analysis for each historical exercise
            - final_knowledge_state: Final knowledge state string
            - target_analysis: Analysis of target exercise
            - prediction: '0' or '1'
            - explanation: Explanation string
            - raw_response: Original LLM response
        """
        messages = self.create_single_turn_messages(fewshots, user_data, prompts)

        print(f"  [V2] Single-turn prediction, fewshots={len(fewshots)}", flush=True)

        raw_response = glm_chat_v2(
            self.name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Parse the structured response
        parsed = parse_single_turn_response(raw_response, len(fewshots))

        parsed['raw_response'] = raw_response

        # Validate
        if not validate_parsed_response(parsed):
            print(f"  [V2 WARN] Response validation failed, prediction={parsed['prediction']}", flush=True)

        return parsed

    def get_prompts(self, data_mode: str) -> dict:
        """Get V2 prompts."""
        return generic_get_prompts_v2("glm", data_mode)

    def create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode) -> str:
        """Create user data for prediction."""
        return generic_create_user_data_v2(student_info, test_exercise_info, extra_datas, data_mode)

    def create_fewshots(
        self,
        student_info,
        test_exercise_info,
        extra_datas,
        fewshots_num,
        fewshots_strategy,
        data_mode,
        prompts=None,  # V2 doesn't use prompts for fewshot generation, but keep for compatibility
        dataset_name=None,
        knowledge_graph_path=None,
    ) -> List[str]:
        """Create few-shots (V2: no pre-generated explanations)."""
        return generic_create_fewshots_v2(
            student_info,
            test_exercise_info,
            extra_datas,
            fewshots_num,
            fewshots_strategy,
            data_mode,
            dataset_name=dataset_name,
            knowledge_graph_path=knowledge_graph_path,
        )

    # ============== 兼容 V1 的方法（返回占位符）==============

    def generate_prediction(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> Union[List[str], str]:
        """V2 does not use this method separately. Use generate_single_turn_prediction instead."""
        raise NotImplementedError("V2 uses generate_single_turn_prediction() instead of separate generate_prediction()")

    def generate_analysis(self, fewshots: List[str], prompts: dict) -> Union[List[str], str]:
        """V2 does not use this method separately."""
        raise NotImplementedError("V2 uses generate_single_turn_prediction() which includes analysis")

    def generate_explaination(self, fewshots: List[str], prediction, prompts: dict) -> Union[List[str], str]:
        """V2 does not use this method separately."""
        raise NotImplementedError("V2 uses generate_single_turn_prediction() which includes explanation")


class GLM47V2(GLMChatV2):
    """GLM-4.7 V2 model."""
    def __init__(self):
        super().__init__(GLM_MODEL_NAME)


class GLM4V2(GLMChatV2):
    """GLM-4 V2 model."""
    def __init__(self):
        super().__init__("glm-4")


class GLM3V2(GLMChatV2):
    """GLM-3-turbo V2 model."""
    def __init__(self):
        super().__init__("glm-3-turbo")
