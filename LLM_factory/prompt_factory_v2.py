"""
V2 Prompt Factory - Single-turn structured output prompts
将多轮对话合并为单轮调用，保留递进式分析逻辑
每个历史题目都进行详细分析（类似V1预生成的explanation），最后汇总预测
"""

import re
from typing import Dict, List, Optional

# ============== V2 System Instructions (改进版) ==============

V2_ONEHOT_SYSTEM_INSTRUCTION = '''You are a knowledge tracing model that predicts whether a student will answer a new question correctly based on their exercise history.

CRITICAL: You must analyze EACH historical exercise in detail, then make a balanced prediction.

## Analysis Requirements for Each Historical Exercise

For each historical exercise, analyze:
1. What knowledge points are involved
2. Connection to previous exercises (is it similar or new?)
3. Why the student got it right/wrong (mastery vs guessing vs carelessness)
4. Update the student's knowledge state

## Output Format (STRICTLY follow this XML structure)

<Exercise_1_Analysis>
Knowledge points: [list the knowledge points]
Connection: [Similar to previous exercise X / New topic / Related to ...]
Student performance: The student got it [right/wrong], [likely because of mastery / possibly due to guessing / may be due to carelessness].
Knowledge state update: <knowledge point, mastery level (good/fair/bad)>
</Exercise_1_Analysis>

<Exercise_2_Analysis>
Knowledge points: ...
Connection: ...
Student performance: ... (analyze why, considering previous performance)
Knowledge state update: ...
</Exercise_2_Analysis>

... (continue for ALL historical exercises)

<Final_Knowledge_Summary>
After all historical exercises:
- Strong knowledge points: [list with mastery level]
- Weak knowledge points: [list with mastery level]
- Patterns observed: [e.g., student struggles with X type of problems, good at Y]
- Potential issues: [guessing, carelessness, knowledge gaps]
</Final_Knowledge_Summary>

<Target_Exercise_Analysis>
Knowledge points in target: [list]
Connection to history: [similar to exercise X / new combination / ...]
Student's readiness: [well-prepared / partially prepared / weak preparation]
Risk factors: [identify potential issues]
</Target_Exercise_Analysis>

<Prediction>
[0 or 1 ONLY]
</Prediction>

<Reasoning>
[Predict 0 or 1 based on: 1) knowledge mastery, 2) pattern analysis, 3) risk assessment]
</Reasoning>
'''

V2_SPARSE_SYSTEM_INSTRUCTION = '''You are a knowledge tracing model that predicts whether a student will answer a new question correctly based on their exercise history and knowledge concepts.

CRITICAL: You must analyze EACH historical exercise in detail, then make a balanced prediction.

## Analysis Requirements for Each Historical Exercise

For each historical exercise, analyze:
1. What knowledge concepts are involved
2. Connection to previous exercises (is it similar or new?)
3. Why the student got it right/wrong (mastery vs guessing vs carelessness)
4. Update the student's knowledge state

## Output Format (STRICTLY follow this XML structure)

<Exercise_1_Analysis>
Knowledge concepts: [list the knowledge concepts]
Connection: [Similar to previous exercise X / New topic / Related to ...]
Student performance: The student got it [right/wrong], [likely because of mastery / possibly due to guessing / may be due to carelessness].
Knowledge state update: <knowledge concept, mastery level (good/fair/bad)>
</Exercise_1_Analysis>

<Exercise_2_Analysis>
Knowledge concepts: ...
Connection: ...
Student performance: ... (analyze why, considering previous performance)
Knowledge state update: ...
</Exercise_2_Analysis>

... (continue for ALL historical exercises)

<Final_Knowledge_Summary>
After all historical exercises:
- Strong knowledge concepts: [list with mastery level]
- Weak knowledge concepts: [list with mastery level]
- Patterns observed: [e.g., student struggles with X type of problems, good at Y]
- Potential issues: [guessing, carelessness, knowledge gaps]
</Final_Knowledge_Summary>

<Target_Exercise_Analysis>
Knowledge concepts in target: [list]
Connection to history: [similar to exercise X / new combination / ...]
Student's readiness: [well-prepared / partially prepared / weak preparation]
Risk factors: [identify potential issues]
</Target_Exercise_Analysis>

<Prediction>
[0 or 1 ONLY]
</Prediction>

<Reasoning>
[Predict 0 or 1 based on: 1) knowledge mastery, 2) pattern analysis, 3) risk assessment]
</Reasoning>
'''

V2_MODERATE_SYSTEM_INSTRUCTION = '''You are a knowledge tracing model that predicts whether a student will answer a new question correctly based on their exercise history, exercise content, and knowledge concepts.

CRITICAL: You must analyze EACH historical exercise in detail, then make a balanced prediction.

## Analysis Requirements for Each Historical Exercise

For each historical exercise, analyze:
1. What knowledge concepts are involved and what the exercise content is about
2. Connection to previous exercises (is it similar or new?)
3. Why the student got it right/wrong (mastery vs guessing vs carelessness)
4. Update the student's knowledge state

## Output Format (STRICTLY follow this XML structure)

<Exercise_1_Analysis>
Knowledge concepts: [list the knowledge concepts]
Exercise content: [brief summary]
Connection: [Similar to previous exercise X / New topic / Related to ...]
Student performance: The student got it [right/wrong], [likely because of mastery / possibly due to guessing / may be due to carelessness].
Knowledge state update: <knowledge concept, mastery level (good/fair/bad)>
</Exercise_1_Analysis>

<Exercise_2_Analysis>
Knowledge concepts: ...
Exercise content: ...
Connection: ...
Student performance: ... (analyze why, considering previous performance)
Knowledge state update: ...
</Exercise_2_Analysis>

... (continue for ALL historical exercises)

<Final_Knowledge_Summary>
After all historical exercises:
- Strong knowledge concepts: [list with mastery level]
- Weak knowledge concepts: [list with mastery level]
- Patterns observed: [e.g., student struggles with X type of problems, good at Y]
- Potential issues: [guessing, carelessness, knowledge gaps]
</Final_Knowledge_Summary>

<Target_Exercise_Analysis>
Knowledge concepts in target: [list]
Exercise content: [brief summary]
Connection to history: [similar to exercise X / new combination / ...]
Student's readiness: [well-prepared / partially prepared / weak preparation]
Risk factors: [identify potential issues]
</Target_Exercise_Analysis>

<Prediction>
[0 or 1 ONLY]
</Prediction>

<Reasoning>
[Predict 0 or 1 based on: 1) knowledge mastery, 2) pattern analysis, 3) risk assessment]
</Reasoning>
'''


# ============== V2 User Instructions ==============

V2_USER_INSTRUCTION = '''
## Student's Exercise History
{fewshots}

## Exercise to Predict
{exercise_to_predict}

Based on the exercise history above, analyze EACH exercise step by step following the XML format, then predict whether the student will answer the target exercise correctly.

IMPORTANT:
1. Analyze every single historical exercise with detailed reasoning
2. Consider both mastery AND potential issues (guessing, carelessness, gaps)
3. Make balanced predictions - do NOT always predict 1
'''


# ============== Prompt Dictionaries ==============

model_v2_sys_instr = {
    'glm': V2_ONEHOT_SYSTEM_INSTRUCTION,
    'gpt': V2_ONEHOT_SYSTEM_INSTRUCTION
}

model_v2_user_instr = {
    'glm': V2_USER_INSTRUCTION,
    'gpt': V2_USER_INSTRUCTION
}

model_v2_sp_sys_instr = {
    'glm': V2_SPARSE_SYSTEM_INSTRUCTION,
    'gpt': V2_SPARSE_SYSTEM_INSTRUCTION
}

model_v2_mo_sys_instr = {
    'glm': V2_MODERATE_SYSTEM_INSTRUCTION,
    'gpt': V2_MODERATE_SYSTEM_INSTRUCTION
}


def generic_get_prompts_v2(model_name: str, data_mode: str) -> dict:
    """
    Get V2 prompts for single-turn structured output.
    """
    if data_mode == 'onehot':
        return {
            'sys_instr': model_v2_sys_instr[model_name],
            'user_instr': model_v2_user_instr[model_name]
        }
    elif data_mode == 'sparse':
        return {
            'sys_instr': model_v2_sp_sys_instr[model_name],
            'user_instr': model_v2_user_instr[model_name]
        }
    elif data_mode == 'moderate':
        return {
            'sys_instr': model_v2_mo_sys_instr[model_name],
            'user_instr': model_v2_user_instr[model_name]
        }
    elif data_mode == 'rich':
        raise NotImplementedError("Rich mode not implemented for V2")
    else:
        raise ValueError(f"Invalid data_mode: {data_mode}")


def parse_single_turn_response(response: str, fewshot_count: int) -> Dict:
    """
    Parse single-turn response to extract all analysis components.
    Supports both old format (History_Analysis_N) and new format (Exercise_N_Analysis).
    """
    result = {
        'history_analyses': [],
        'final_knowledge_state': '',
        'target_analysis': '',
        'prediction': None,
        'explanation': ''
    }

    # Try new format first: <Exercise_N_Analysis>
    for i in range(1, fewshot_count + 1):
        pattern = rf'<Exercise_{i}_Analysis>(.*?)</Exercise_{i}_Analysis>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result['history_analyses'].append(match.group(1).strip())

    # If no matches, try old format: <History_Analysis_N>
    if not result['history_analyses']:
        for i in range(1, fewshot_count + 1):
            pattern = rf'<History_Analysis_{i}>(.*?)</History_Analysis_{i}>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                result['history_analyses'].append(match.group(1).strip())

    # If still no matches, try generic pattern
    if not result['history_analyses']:
        pattern_generic = rf'<(?:Exercise|History)_\d+_Analysis>(.*?)</(?:Exercise|History)_\d+_Analysis>'
        matches = re.findall(pattern_generic, response, re.DOTALL | re.IGNORECASE)
        result['history_analyses'] = [m.strip() for m in matches]

    # Extract other components (support both old and new format)
    patterns = {
        'final_knowledge_state': r'<Final_Knowledge_Summary>(.*?)</Final_Knowledge_Summary>',
        'target_analysis': r'<Target_Exercise_Analysis>(.*?)</Target_Exercise_Analysis>',
        'prediction': r'<Prediction>\s*([01])\s*</Prediction>',
        'explanation': r'<Reasoning>(.*?)</Reasoning>'
    }

    # Also try old format patterns
    old_patterns = {
        'final_knowledge_state': r'<Final_Knowledge_State>(.*?)</Final_Knowledge_State>',
        'explanation': r'<Explanation>(.*?)</Explanation>'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            if key == 'prediction':
                result[key] = match.group(1).strip()
            else:
                result[key] = match.group(1).strip()
        elif key in old_patterns:
            # Try old format
            match = re.search(old_patterns[key], response, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()

    # Validate prediction
    if result['prediction'] not in ['0', '1']:
        # Try alternative extraction
        pred_match = re.search(r'\b([01])\b', response)
        if pred_match:
            result['prediction'] = pred_match.group(1)

    return result


def validate_parsed_response(parsed: Dict) -> bool:
    """
    Validate that the parsed response has all required fields.
    """
    if parsed['prediction'] not in ['0', '1']:
        return False
    # Allow empty final_knowledge_state and explanation for more flexibility
    return True
