from typing import List

from LLM_factory.GLM import GLMChat, GLM4, GLM3
from LLM_factory.fewshot_generator_v3 import generic_create_fewshots_v3, generic_create_user_data_v3
from LLM_factory.prompt_factory_v3 import generic_get_prompts_v3


class GLMChatV3(GLMChat):
    def get_prompts(self, data_mode: str) -> dict:
        return generic_get_prompts_v3('glm', data_mode)

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


class GLM47V3(GLMChatV3):
    def __init__(self):
        super().__init__('z-ai/glm-4.7')


class GLM4V3(GLMChatV3):
    def __init__(self):
        super().__init__('glm-4')


class GLM3V3(GLMChatV3):
    def __init__(self):
        super().__init__('glm-3-turbo')
