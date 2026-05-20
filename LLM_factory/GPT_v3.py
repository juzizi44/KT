from typing import List

from LLM_factory.GPT import GPTChat, GPT4, GPT35
from LLM_factory.fewshot_generator_v3 import generic_create_fewshots_v3, generic_create_user_data_v3
from LLM_factory.prompt_factory_v3 import generic_get_prompts_v3


class GPTChatV3(GPTChat):
    def get_prompts(self, data_mode: str) -> dict:
        return generic_get_prompts_v3('gpt', data_mode)

    def create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode) -> str:
        return generic_create_user_data_v3(student_info, test_exercise_info, extra_datas, data_mode)

    def create_fewshots(
        self,
        student_info,
        test_exercise_info,
        extra_datas,
        fewshots_num,
        fewshots_strategy,
        data_mode,
        prompts=None,
        dataset_name=None,
        knowledge_graph_path=None,
    ) -> List[str]:
        return generic_create_fewshots_v3(
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


class GPT4V3(GPTChatV3):
    def __init__(self):
        super().__init__('gpt-4')


class GPT35V3(GPTChatV3):
    def __init__(self):
        super().__init__('gpt-3.5-turbo')
