"""
V2 Few-shot Generator - Simplified for single-turn output
不需要预先生成 explanation，只提供原始数据
"""

from typing import List
from utils import sample_fs_id


def generic_create_user_data_v2(student_info, test_exercise_info, extra_datas, data_mode) -> str:
    """
    Create prediction data for a test exercise to send to LLM.
    V2 version: simplified format for single-turn output.

    Args:
        student_info: Student information dict
        test_exercise_info: Test exercise information dict
        extra_datas: Extra data containing exercise info
        data_mode: 'onehot', 'sparse', 'moderate', or 'rich'

    Returns:
        Formatted user data string
    """
    user_data = ''
    if data_mode == "onehot":
        user_data += '<Exercise to Predict>\n'
        user_data += f"exercise_id: {test_exercise_info['exercise_id']}, knowledge concepts: {test_exercise_info['skill_ids']}\n"
    elif data_mode == "sparse":
        user_data += '<Exercise to Predict>\n'
        user_data += f"knowledge concepts: {test_exercise_info['skill_desc']}\n"
    elif data_mode == "moderate":
        user_data += '<Exercise to Predict>\n'
        user_data += f"exercise content: {test_exercise_info['exercise_desc']}\n"
        user_data += f"knowledge concepts: {test_exercise_info['skill_desc']}\n"
    elif data_mode == "rich":
        raise NotImplementedError("Rich mode not implemented for V2")
    else:
        raise ValueError(f"Invalid data_mode: {data_mode}")

    return user_data


def generic_create_fewshots_v2(student_info,
                               test_exercise_info,
                               extra_datas,
                               fewshots_num,
                               fewshots_strategy,
                               data_mode,
                               dataset_name=None,
                               knowledge_graph_path=None) -> List[str]:
    """
    Create few-shot examples for V2 single-turn output.
    Simplified: NO pre-generated explanations, only raw data.

    Args:
        student_info: Student information dict
        test_exercise_info: Test exercise information dict
        extra_datas: Extra data containing exercise info
        fewshots_num: Number of few-shot examples
        fewshots_strategy: Strategy for sampling few-shots
        data_mode: 'onehot', 'sparse', or 'moderate'
        dataset_name: Name of dataset (for certain strategies)
        knowledge_graph_path: Path to knowledge graph (for knowledge_graph strategy)

    Returns:
        List of few-shot strings (raw data only, no explanations)
    """
    if data_mode == "onehot":
        return generic_create_onehot_fewshots_v2(
            student_info, test_exercise_info, extra_datas,
            fewshots_num, fewshots_strategy, dataset_name, knowledge_graph_path
        )
    elif data_mode == "sparse":
        return generic_create_sparse_fewshots_v2(
            student_info, test_exercise_info, extra_datas,
            fewshots_num, fewshots_strategy, dataset_name, knowledge_graph_path
        )
    elif data_mode == "moderate":
        return generic_create_moderate_fewshots_v2(
            student_info, test_exercise_info, extra_datas,
            fewshots_num, fewshots_strategy, dataset_name, knowledge_graph_path
        )
    elif data_mode == "rich":
        raise NotImplementedError("Rich mode not implemented for V2")
    else:
        raise ValueError(f"Invalid data_mode: {data_mode}")


def generic_create_onehot_fewshots_v2(student_info,
                                      test_exercise_info,
                                      extra_datas,
                                      fewshots_num,
                                      fewshots_strategy,
                                      dataset_name=None,
                                      knowledge_graph_path=None) -> List[str]:
    """
    Create onehot format few-shots for V2.
    Format: exercise_id, knowledge concepts, is_correct (no explanation)
    """
    ret_fewshots = []
    fs_ex_ids = sample_fs_id(
        student_info['exercises_logs'], test_exercise_info, extra_datas,
        fewshots_num, fewshots_strategy, student_info, dataset_name, knowledge_graph_path
    )

    if fewshots_num == 0:
        return ret_fewshots

    other_ex_info = extra_datas['exercise_info']

    for i, ex_id in fs_ex_ids:
        skill_ids = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_ids'].values[0]
        is_correct = 'right' if student_info['is_corrects'][i] else 'wrong'

        fewshot = f"<Exercise {ex_id}>\n"
        fewshot += f"exercise_id: {ex_id}, knowledge concepts: {skill_ids}, is_correct: {is_correct}\n"
        fewshot += f"</Exercise {ex_id}>\n"

        ret_fewshots.append(fewshot)

    return ret_fewshots


def generic_create_sparse_fewshots_v2(student_info,
                                      test_exercise_info,
                                      extra_datas,
                                      fewshots_num,
                                      fewshots_strategy,
                                      dataset_name=None,
                                      knowledge_graph_path=None) -> List[str]:
    """
    Create sparse format few-shots for V2.
    Format: exercise_id, knowledge concepts description, is_correct (no explanation)
    """
    ret_fewshots = []
    fs_ex_ids = sample_fs_id(
        student_info['exercises_logs'], test_exercise_info, extra_datas,
        fewshots_num, fewshots_strategy, student_info, dataset_name, knowledge_graph_path
    )

    if fewshots_num == 0:
        return ret_fewshots

    other_ex_info = extra_datas['exercise_info']

    for i, ex_id in fs_ex_ids:
        skill_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_desc'].values[0]
        is_correct = 'right' if student_info['is_corrects'][i] else 'wrong'

        fewshot = f"<Exercise {ex_id}>\n"
        fewshot += f"exercise_id: {ex_id}, knowledge concepts: {skill_desc}, is_correct: {is_correct}\n"
        fewshot += f"</Exercise {ex_id}>\n"

        ret_fewshots.append(fewshot)

    return ret_fewshots


def generic_create_moderate_fewshots_v2(student_info,
                                        test_exercise_info,
                                        extra_datas,
                                        fewshots_num,
                                        fewshots_strategy,
                                        dataset_name=None,
                                        knowledge_graph_path=None) -> List[str]:
    """
    Create moderate format few-shots for V2.
    Format: exercise_id, exercise content, knowledge concepts, is_correct (no explanation)
    """
    ret_fewshots = []
    fs_ex_ids = sample_fs_id(
        student_info['exercises_logs'], test_exercise_info, extra_datas,
        fewshots_num, fewshots_strategy, student_info, dataset_name, knowledge_graph_path
    )

    if fewshots_num == 0:
        return ret_fewshots

    other_ex_info = extra_datas['exercise_info']

    for i, ex_id in fs_ex_ids:
        skill_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_desc'].values[0]
        ex_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['exercise_desc'].values[0]
        is_correct = 'right' if student_info['is_corrects'][i] else 'wrong'

        fewshot = f"<Exercise {ex_id}>\n"
        fewshot += f"exercise_id: {ex_id}\n"
        fewshot += f"exercise content: {ex_desc}\n"
        fewshot += f"knowledge concepts: {skill_desc}, is_correct: {is_correct}\n"
        fewshot += f"</Exercise {ex_id}>\n"

        ret_fewshots.append(fewshot)

    return ret_fewshots
