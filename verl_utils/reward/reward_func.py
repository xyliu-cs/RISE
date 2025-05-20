try:
    from .openmathinst_utils import extract_answer, math_equal
except:
    from verl_utils.reward.openmathinst_utils import extract_answer, math_equal

def reward_func(data_source, solution_str, ground_truth, extra_info) -> float:
    extracted_answer = extract_answer(solution_str, extract_from_boxed=True)
    if extracted_answer is None: # formatting error
        return -1.0
    else:
        if math_equal(extracted_answer, ground_truth, check_antlr_version=False):
            return 1.0
        else:
            return -0.5

def ver_reward_func(data_source, solution_str, ground_truth, extra_info) -> float:
    extracted_answer = extract_answer(solution_str, extract_from_boxed=True)
    if extracted_answer is None: # formatting error
        return -1.0
    if len(solution_str) < 800:
        return -1.0
    else:
        if math_equal(extracted_answer, ground_truth, check_antlr_version=False):
            return 1.0
        else:
            return -0.5
