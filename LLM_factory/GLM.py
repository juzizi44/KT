from typing import List, Union
import dataclasses
import time

from LLM_factory.model import Message, LLMModelBase
from LLM_factory.prompt_factory import generic_get_prompts
from LLM_factory.fewshot_generator import (
    generic_create_user_data,
    generic_create_analysis_messages,
    generic_create_explaination_messages,
    generic_create_fewshots,
    generic_create_prediction_messages,
)

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

from openai import OpenAI
import os

# GLM 配置（通过 bianxie.ai 的 OpenAI 兼容接口调用）
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.bianxie.ai/v1"))
GLM_MODEL_NAME = os.getenv("GLM_MODEL_NAME", "glm-4.7")

if not GLM_API_KEY:
    raise ValueError("GLM_API_KEY environment variable is not set")

glm_client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


@retry(wait=wait_random_exponential(min=60, max=180), stop=stop_after_attempt(6))
def glm_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.2,
    num_comps=1,
) -> Union[List[str], str]:
    start_ts = time.time()
    print(f"  [GLM API] -> model={model}, msgs={len(messages)}, max_tokens={max_tokens}", flush=True)
    try:
        response = glm_client.chat.completions.create(
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
        print(f"  [GLM API] !! request failed after {elapsed:.1f}s: {exc}", flush=True)
        raise

    elapsed = time.time() - start_ts
    print(f"  [GLM API] <- done in {elapsed:.1f}s, choices={len(response.choices)}", flush=True)

    time.sleep(0.1)

    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


class GLMChat(LLMModelBase):
    """通过 bianxie.ai 的 OpenAI 兼容接口调用 glm 模型。"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.is_chat = True

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 4096,
        temperature: float = 0.2,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        return glm_chat(self.name, messages, max_tokens, temperature, num_comps)

    def generate_prediction(
        self,
        fewshots: List[str],
        user_data: str,
        prompts: dict,
        use_selected_fewshot: bool = True,
    ) -> Union[List[str], str]:
        prediction_messages = self.create_prediction_messages(
            fewshots, user_data, prompts, use_selected_fewshot
        )
        return glm_chat(self.name, prediction_messages, num_comps=1)

    def generate_analysis(self, fewshots: List[str], prompts: dict) -> Union[List[str], str]:
        analysis_messages = self.create_analysis_messages(fewshots, prompts)
        return glm_chat(self.name, analysis_messages, num_comps=1)

    def generate_explaination(
        self, fewshots: List[str], prediction, prompts: dict
    ) -> Union[List[str], str]:
        explaination_messages = self.create_explaination_messages(
            fewshots, prediction, prompts
        )
        return glm_chat(self.name, explaination_messages, num_comps=1)

    def create_prediction_messages(
        self,
        fewshots: List[str],
        user_data: str,
        prompts: dict,
        use_selected_fewshot: bool = True,
    ) -> List[Message]:
        return generic_create_prediction_messages(
            fewshots, user_data, prompts, use_selected_fewshot
        )

    def create_analysis_messages(self, fewshots: List[str], prompts: dict) -> List[Message]:
        return generic_create_analysis_messages(fewshots, prompts)

    def create_explaination_messages(
        self, fewshots: List[str], prediction, prompts: dict
    ) -> List[Message]:
        return generic_create_explaination_messages(fewshots, prediction, prompts)

    def create_fewshots(
        self,
        student_info,
        test_exercise_info,
        extra_datas,
        fewshots_num,
        fewshots_strategy,
        data_mode,
        prompts,
        dataset_name=None,
        knowledge_graph_path=None,
    ) -> List[str]:
        return generic_create_fewshots(
            student_info,
            test_exercise_info,
            extra_datas,
            fewshots_num,
            fewshots_strategy,
            data_mode,
            prompts,
            self.generate_analysis,
            self.generate_explaination,
            dataset_name=dataset_name,
            knowledge_graph_path=knowledge_graph_path,
        )

    def get_prompts(self, data_mode: str):
        return generic_get_prompts("glm", data_mode)

    def create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode) -> str:
        return generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode)


class GLM47(GLMChat):
    """glm-4.7，当前默认使用的模型。"""
    def __init__(self):
        super().__init__(GLM_MODEL_NAME)


class GLM4(GLMChat):
    def __init__(self):
        super().__init__("glm-4")


class GLM3(GLMChat):
    def __init__(self):
        super().__init__("glm-3-turbo")