import random
from typing import List, Tuple, Any, Dict, Optional

from selection_strategies import create_selector


def check_response_format(response: str) -> str:
    # response if the response is '1' or '0'
    if response == '1' or response == '0':
        return response
    else:
        raise ValueError(f"Invalid response format: {response}, should be '1' or '0'")

def aggregate_data(id, extra_data, data_type):
    # data_type is exercise or student
    agg_data = {}
    if data_type == "exercise":
        for col in extra_data.columns:
            agg_data[col] = extra_data[extra_data['exercise_id'] == id][col].values[0]
    elif data_type == "student":
        for col in extra_data.columns:
            agg_data[col] = extra_data[extra_data['student_id'] == id][col].values[0]
    else:
        raise ValueError(f"Invalid data_type: {data_type}, when aggregating data")
    return agg_data


def sample_fs_id(sample_list, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy,
                  student_info=None, dataset_name=None, knowledge_graph_path=None) -> list:
    # return a list of tuples of (index in sample_list, exercise_id)
    fs_ex_ids = []
    if fewshots_num == 0:
        return fs_ex_ids
    # fewshots_strategy: "random", "first", "last", "concept_based", "knowledge_graph"
    if fewshots_strategy == "random":
        # randomly sample exercise_id from student_info to create fewshots
        # sample_list is a numpy array of str
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in random.sample(range(len(sample_list)), fewshots_num)]
    elif fewshots_strategy == "first":
        # use the first fewshot_num exercise_id from student_info to create fewshots
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in range(fewshots_num)]
    elif fewshots_strategy == "last":
        # use the last fewshot_num exercise_id from student_info to create fewshots
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list)-fewshots_num, len(sample_list))]
    elif fewshots_strategy == "concept_based" or fewshots_strategy == "knowledge_graph":
        # Use the new selection strategies
        fs_ex_ids = _select_with_strategy(
            sample_list, test_exercise_info, extra_datas, fewshots_num,
            fewshots_strategy, student_info, dataset_name, knowledge_graph_path
        )
    elif fewshots_strategy == "RAG":
        # TODO: implement Strategy
        raise NotImplementedError
    elif fewshots_strategy == "BLEU":
        # TODO: implement Strategy
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid fewshots_strategy: {fewshots_strategy}")
    return fs_ex_ids


def _select_with_strategy(
    sample_list,
    test_exercise_info,
    extra_datas,
    fewshots_num,
    strategy,
    student_info,
    dataset_name,
    knowledge_graph_path=None
) -> List[Tuple[int, str]]:
    """Helper function to use the new selection strategies.

    Args:
        sample_list: List of exercise_ids from student history
        test_exercise_info: Dict with test exercise info (skill_ids, skill_desc, etc.)
        extra_datas: Dict containing 'exercise_info' DataFrame
        fewshots_num: Number of few-shots to select
        strategy: 'concept_based' or 'knowledge_graph'
        student_info: Dict with student info (is_corrects list)
        dataset_name: Dataset name for knowledge graph path resolution
        knowledge_graph_path: Explicit path to knowledge graph JSON file

    Returns:
        List of (index, exercise_id) tuples
    """
    exercise_info = extra_datas.get('exercise_info')

    # Build records list for selector
    records = []
    is_corrects = student_info.get('is_corrects', []) if student_info is not None else []

    for idx, ex_id in enumerate(sample_list):
        ex_id_str = str(ex_id)
        # Get exercise info from DataFrame
        ex_row = exercise_info[exercise_info['exercise_id'] == ex_id_str]

        if ex_row.empty:
            continue

        record = {
            'exercise_id': ex_id_str,
            'skill_ids': list(ex_row['skill_ids'].values[0]),
            'index': idx,
            'is_correct': is_corrects[idx] if idx < len(is_corrects) else None,
        }

        # Add skill_desc if available
        if 'skill_desc' in ex_row.columns:
            record['skill_desc'] = list(ex_row['skill_desc'].values[0])

        records.append(record)

    # Build test record
    test_record = {
        'exercise_id': str(test_exercise_info.get('exercise_id', '')),
        'skill_ids': test_exercise_info.get('skill_ids', []),
        'skill_desc': test_exercise_info.get('skill_desc', []),
    }

    # Create and use selector
    selector = create_selector(strategy, dataset_name=dataset_name, graph_path=knowledge_graph_path)
    selected_records = selector.select(records, fewshots_num, test_record)

    # Convert back to (index, exercise_id) format
    fs_ex_ids = [(r['index'], r['exercise_id']) for r in selected_records]
    return fs_ex_ids


def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)