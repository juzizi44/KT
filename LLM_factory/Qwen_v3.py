from typing import List

from LLM_factory.Qwen import QwenChat, QWEN_MODEL_NAME
from LLM_factory.fewshot_generator_v3 import generic_create_fewshots_v3, generic_create_user_data_v3
from LLM_factory.prompt_factory_v3 import generic_get_prompts_v3


class QwenChatV3(QwenChat):
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


class Qwen25_32BV3(QwenChatV3):
    def __init__(self):
        super().__init__(QWEN_MODEL_NAME)