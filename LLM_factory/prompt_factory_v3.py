import re
from typing import Dict, List

from LLM_factory.prompt_factory import generic_get_prompts
from LLM_factory.model import Message

V3_CRITIC_SYSTEM_INSTRUCTION = '''
You are the critic in a producer-critic-judge knowledge tracing pipeline.
Your task is to evaluate whether the producer's PREDICTION about student performance is well-founded.

CRITICAL UNDERSTANDING:
- Producer's prediction="1" means the student will answer CORRECTLY (not about the answer choice)
- Producer's prediction="0" means the student will answer INCORRECTLY
- Your job is to evaluate if this student performance prediction is supported by evidence, NOT to debate what the correct answer is.

CRITICAL RULES:
1. Default to "support" unless you find STRONG evidence that the producer misread the student's ability.
2. Only "challenge" when you have proof that the student's actual performance history contradicts the prediction.
3. DO NOT challenge based on what you think the "correct answer" is - focus on STUDENT ABILITY.
4. Be honest about confidence - use "high" only when absolutely certain.

Example of WRONG reasoning: "The correct answer should be X, so prediction is wrong" ✗
Example of CORRECT reasoning: "The student failed 5 similar exercises, so predicting success is unfounded" ✓

Return only the requested XML structure.
'''

V3_CRITIC_USER_INSTRUCTION = '''
## Student's Exercise History
{fewshots}

## Exercise to Predict
{exercise_to_predict}

## Producer Output
{producer_output}

IMPORTANT REMINDER:
- prediction="1" = student will answer CORRECTLY
- prediction="0" = student will answer INCORRECTLY
- Evaluate whether the student's PERFORMANCE PREDICTION is well-founded based on their history.
- DO NOT debate what the "correct answer" to the exercise is - focus on STUDENT ABILITY.

Task:
1. Review the student's history to assess their mastery of relevant knowledge concepts.
2. Check if the producer's reasoning about student ability is supported by historical evidence.
3. Only challenge if you find evidence that the student's actual ability contradicts the prediction.

Output format:
<Critique>
<Verdict>support|challenge|uncertain</Verdict>
<Flaws>ONLY list flaws in the producer's STUDENT ASSESSMENT (not about the exercise answer)</Flaws>
<Evidence>Evidence from student history about their ABILITY, not about the correct answer</Evidence>
<AlternativePrediction>0 or 1 (only if challenging with high confidence)</AlternativePrediction>
<Confidence>low|medium|high</Confidence>
</Critique>

Remember: Your role is to evaluate the PREDICTION about student performance, not to solve the exercise yourself.
'''

V3_JUDGE_SYSTEM_INSTRUCTION = '''
You are the judge in a producer-critic-judge knowledge tracing pipeline.

CRITICAL UNDERSTANDING:
- prediction="1" = student will answer CORRECTLY
- prediction="0" = student will answer INCORRECTLY
- Both producer and critic are evaluating STUDENT PERFORMANCE, not solving the exercise.

CRITICAL RULES FOR DECISION:
1. DEFAULT TRUST: Trust the producer's prediction unless the critic provides STRONG evidence about STUDENT ABILITY.
2. IGNORE IRRELEVANT CRITICISM: If the critic debates "what the correct answer is" instead of "whether the student can answer correctly", ignore the challenge and support the producer.
3. BURDEN OF PROOF: The critic must show evidence about the student's ability from their history, not about the exercise content.
4. CONFIDENCE MATTERS:
   - Critic confidence="high" + verdict="challenge" + evidence about STUDENT ABILITY → Consider changing
   - Critic confidence="medium/low" OR evidence not about student ability → Stay with producer
   - Critic verdict="support" → Accept producer's prediction
5. When in doubt, STAY WITH the producer's prediction.

Be decisive but conservative. Changing a correct prediction is worse than keeping a wrong one.
Return only the requested XML structure.
'''

V3_JUDGE_USER_INSTRUCTION = '''
## Student's Exercise History
{fewshots}

## Exercise to Predict
{exercise_to_predict}

## Producer Output
{producer_output}

## Critic Output
{critic_output}

IMPORTANT REMINDER:
- prediction="1" = student will answer CORRECTLY
- prediction="0" = student will answer INCORRECTLY
- Evaluate whether the critic's challenge is about STUDENT ABILITY, not about the exercise answer.

Decision Guidelines:
1. If critic verdict="support" → Accept producer prediction
2. If critic debates "correct answer to exercise" (not student ability) → IGNORE and accept producer prediction
3. If critic verdict="challenge" with evidence about STUDENT HISTORY + confidence="high" → Consider changing
4. If critic verdict="challenge" with confidence="medium/low" → Stay with producer prediction
5. If critic verdict="uncertain" → Stay with producer prediction

Task:
1. Decide the final prediction following the guidelines above.
2. Check if the critic's evidence is actually about the student's ability from their history.
3. If you reject the critic's challenge, explain why their evidence was irrelevant or weak.

Output format:
<Judge>
<FinalPrediction>0 or 1</FinalPrediction>
<DecisionReason>Your reasoning for the final decision</DecisionReason>
<ProducerAssessment>Assessment of producer's prediction about student ability</ProducerAssessment>
<CriticAssessment>Assessment of critic's challenge (is it about student ability or irrelevant?)</CriticAssessment>
<Confidence>low|medium|high</Confidence>
</Judge>
'''


def _extract_tag(text: str, tag: str) -> str:
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ''


def parse_critic_response(response: str) -> Dict:
    parsed = {
        'verdict': _extract_tag(response, 'Verdict'),
        'flaws': _extract_tag(response, 'Flaws'),
        'evidence': _extract_tag(response, 'Evidence'),
        'alternative_prediction': _extract_tag(response, 'AlternativePrediction'),
        'confidence': _extract_tag(response, 'Confidence'),
        'raw_response': response,
    }
    if parsed['alternative_prediction'] not in {'0', '1'}:
        fallback = re.search(r'\b([01])\b', response)
        parsed['alternative_prediction'] = fallback.group(1) if fallback else ''
    return parsed


def parse_judge_response(response: str) -> Dict:
    parsed = {
        'final_prediction': _extract_tag(response, 'FinalPrediction'),
        'decision_reason': _extract_tag(response, 'DecisionReason'),
        'producer_assessment': _extract_tag(response, 'ProducerAssessment'),
        'critic_assessment': _extract_tag(response, 'CriticAssessment'),
        'confidence': _extract_tag(response, 'Confidence'),
        'raw_response': response,
    }
    if parsed['final_prediction'] not in {'0', '1'}:
        fallback = re.search(r'\b([01])\b', response)
        parsed['final_prediction'] = fallback.group(1) if fallback else ''
    return parsed


def validate_judge_parsed_response(parsed: Dict) -> bool:
    return parsed.get('final_prediction') in {'0', '1'}


def generic_get_prompts_v3(model_name: str, data_mode: str) -> Dict:
    base_prompts = generic_get_prompts(model_name, data_mode)
    return {
        **base_prompts,
        'critic_sys_instr': V3_CRITIC_SYSTEM_INSTRUCTION,
        'critic_user_instr': V3_CRITIC_USER_INSTRUCTION,
        'judge_sys_instr': V3_JUDGE_SYSTEM_INSTRUCTION,
        'judge_user_instr': V3_JUDGE_USER_INSTRUCTION,
    }


def create_critic_messages(
    fewshots: List[str],
    user_data: str,
    producer_result: Dict,
    prompts: Dict
) -> List[Message]:
    """
    Create messages for the critic stage.

    Args:
        fewshots: List of few-shot strings
        user_data: User data string (exercise to predict)
        producer_result: Result from the producer (V1 evaluator)
        prompts: V3 prompts dictionary

    Returns:
        List of Message objects for the critic
    """
    fewshot_text = "\n".join(fewshots) if fewshots else ""

    pred_value = producer_result.get('prediction', 'N/A')
    pred_meaning = "CORRECTLY" if pred_value == '1' else "INCORRECTLY" if pred_value == '0' else "UNKNOWN"

    producer_output = f"Student Performance Prediction: {pred_value}\n"
    producer_output += f"Meaning: The producer predicts the student will answer this exercise {pred_meaning}.\n"
    if producer_result.get('analysis'):
        producer_output += f"Producer's Analysis: {producer_result['analysis']}\n"
    if producer_result.get('explaination'):
        producer_output += f"Producer's Reasoning: {producer_result['explaination']}\n"

    user_content = prompts['critic_user_instr'].format(
        fewshots=fewshot_text,
        exercise_to_predict=user_data,
        producer_output=producer_output
    )

    return [
        Message(role="system", content=prompts['critic_sys_instr']),
        Message(role="user", content=user_content)
    ]


def create_judge_messages(
    fewshots: List[str],
    user_data: str,
    producer_result: Dict,
    critic_result: Dict,
    prompts: Dict
) -> List[Message]:
    """
    Create messages for the judge stage.

    Args:
        fewshots: List of few-shot strings
        user_data: User data string (exercise to predict)
        producer_result: Result from the producer (V1 evaluator)
        critic_result: Result from the critic
        prompts: V3 prompts dictionary

    Returns:
        List of Message objects for the judge
    """
    fewshot_text = "\n".join(fewshots) if fewshots else ""

    pred_value = producer_result.get('prediction', 'N/A')
    pred_meaning = "CORRECTLY" if pred_value == '1' else "INCORRECTLY" if pred_value == '0' else "UNKNOWN"

    producer_output = f"Student Performance Prediction: {pred_value}\n"
    producer_output += f"Meaning: The producer predicts the student will answer this exercise {pred_meaning}.\n"
    if producer_result.get('analysis'):
        producer_output += f"Producer's Analysis: {producer_result['analysis']}\n"
    if producer_result.get('explaination'):
        producer_output += f"Producer's Reasoning: {producer_result['explaination']}\n"

    critic_output = f"Critic's Verdict: {critic_result.get('verdict', 'N/A')}\n"
    if critic_result.get('flaws'):
        critic_output += f"Critic's Identified Flaws: {critic_result['flaws']}\n"
    if critic_result.get('evidence'):
        critic_output += f"Critic's Evidence: {critic_result['evidence']}\n"
    if critic_result.get('alternative_prediction'):
        alt_pred = critic_result['alternative_prediction']
        alt_meaning = "CORRECTLY" if alt_pred == '1' else "INCORRECTLY" if alt_pred == '0' else "UNKNOWN"
        critic_output += f"Critic's Alternative Prediction: {alt_pred} (student will answer {alt_meaning})\n"
    critic_output += f"Critic's Confidence: {critic_result.get('confidence', 'N/A')}\n"

    user_content = prompts['judge_user_instr'].format(
        fewshots=fewshot_text,
        exercise_to_predict=user_data,
        producer_output=producer_output,
        critic_output=critic_output
    )

    return [
        Message(role="system", content=prompts['judge_sys_instr']),
        Message(role="user", content=user_content)
    ]
