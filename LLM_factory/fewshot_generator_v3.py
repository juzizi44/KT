from typing import List

from LLM_factory.fewshot_generator import generic_create_fewshots, generic_create_user_data


def generic_create_user_data_v3(student_info, test_exercise_info, extra_datas, data_mode) -> str:
    return generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode)


def generic_create_fewshots_v3(
    student_info,
    test_exercise_info,
    extra_datas,
    fewshots_num,
    fewshots_strategy,
    data_mode,
    prompts=None,
    generate_analysis_func=None,
    generate_explanation_func=None,
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
        generate_analysis_func,
        generate_explanation_func,
        generate_analysis=generate_analysis_func is not None,
        generate_explanation=generate_explanation_func is not None,
        dataset_name=dataset_name,
        knowledge_graph_path=knowledge_graph_path,
    )
