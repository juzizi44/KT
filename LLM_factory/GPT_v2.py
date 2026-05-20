"""
V2 GPT Implementation - Single-turn structured output
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
import openai
from openai import OpenAI
import os

# GPT API 配置
os.environ["OPENAI_API_KEY"] = "***REMOVED***"
os.environ["OPENAI_BASE_URL"] = "https://api.bianxie.ai/v1"

gpt_client = OpenAI()


@retry(wait=wait_random_exponential(min=60, max=180), stop=stop_after_attempt(6))
def gpt_chat_v2(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.2,
    num_comps=1,
) -> Union[List[str], str]:
    """
    GPT chat API call for V2.
    """
    start_ts = time.time()
    print(f"  [GPT V2 API] -> model={model}, msgs={len(messages)}, max_tokens={max_tokens}", flush=True)
    try:
        response = gpt_client.chat.completions.create(
            model=model,
            messages=[dataclasses.asdict(message) for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
        )
    except Exception as exc:
        elapsed = time.time() - start_ts
        print(f"  [GPT V2 API] !! request failed after {elapsed:.1f}s: {exc}", flush=True)
        raise

    elapsed = time.time() - start_ts
    print(f"  [GPT V2 API] <- done in {elapsed:.1f}s, choices={len(response.choices)}", flush=True)

    # Rate limiting
    time.sleep(0.1)

    if num_comps == 1:
        return response.choices[0].message.content

    return [choice.message.content for choice in response.choices]


class GPTChatV2(LLMModelBase):
    """
    V2 GPT Chat - Single-turn structured output.
    一次调用完成：历史分析 + 知识状态更新 + 预测 + 解释
    """

    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 8192,
        temperature: float = 0.2,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        return gpt_chat_v2(self.name, messages, max_tokens, temperature, num_comps)

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

        raw_response = gpt_chat_v2(
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
        return generic_get_prompts_v2("gpt", data_mode)

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


class GPT4V2(GPTChatV2):
    """GPT-4 V2 model."""
    def __init__(self):
        super().__init__("gpt-4")


class GPT35V2(GPTChatV2):
    """GPT-3.5-turbo V2 model."""
    def __init__(self):
        super().__init__("gpt-3.5-turbo")
