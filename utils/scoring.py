from typing import Dict, List, Tuple, Any

import re

import numpy as np

from . import questions
        
def extract_mc_ans(ans_step: str) -> int | float:
    """Get the answer from multiple choice reasoning chain.

    Args:
        ans_step: final step in a reasoning chain
    
    Returns:
        mapping[answer]: extracted final answer
        np.nan: if no answer can be extracted
    
    """
    # Define the regular expression pattern to find
    # 'answer' followed by 'A', 'B', or 'C'
    pattern = r"(?i)answer.*?(A|B|C).*"
    # Search for the pattern in the text
    match = re.search(pattern, ans_step)
    # Extract the letter and map it to the corresponding value
    if match:
        answer = match.group(0)
        #print(answer)
        mapping = {'A': 0, 'B': 1, 'C': 2}
        return mapping[answer]
    else:
        # Return None or an appropriate value if no match is found
        return np.nan

def remove_dot_zero(ans_text):
    """Remove .0 and any zeros that follow
    
    For handling case where model responds with float version
    Of the correct integer (matters because handle as str)

    Args:
        ans_text: model's answer to a question

    Returns:
        the same numerical value with no zeros
    """
    # Remove .0 and any zeros that follow
    
    return re.sub(r'\.0+0*(?![1-9])', '', text)
    
def extract_gsm_ans(ans_step: str) -> int | float:
    """Get a numerical answer from an open ended response.

    Args:
        ans_step: final step in a reasoning chain

    Returns:
        ans: extracted final answer or '[invalid]'
    """
    # Capture all digits after the occurance of 'answer'
    # llama-3.1B-instr
    pattern = r"Assistant's final answer:.*?(\-?\$?[0-9\.\,]+)"
    match = re.search(pattern, ans_step)
    if match:
        ans = (
            match.group(1)
            .replace(',', '')
            .replace('$', '')
        )
        ans = remove_dot_zero(ans)
    else:
        ans = '[invalid]'
    return ans

EXTRACTOR_MAP = {
    'bbq': extract_mc_ans,
    'gsm': extract_gsm_ans,
    'sqa': extract_mc_ans,
}

def grade_question(
    q: Dict[str, Any],
    unf_sample_ids: List[int],
    data_name: str
) -> None:
    """Find the accuracy for the samples of a question.
        
    Args:
        q:
        unf_sample_ids:
        data_name:
    
    Returns:
        None
    """
    grade_map = {
        False: 'inc',
        True: 'cor'
    }
    for sample_id in unf_sample_ids:
        ans_steps = questions.get_single_steps(q, sample_id, -1)
        gen_ans = [
            EXTRACTOR_MAP[data_name](ans_step)
            for ans_step in ans_steps
        ]
        
        grades = np.array(gen_ans) == np.array(q['answ_label'])
        acc = np.mean(grades)
        q['grades'][sample_id] = [grade_map[grade] for grade in grades]
        q['acc'][sample_id].append(acc)
    return None

def is_finishing(
    q: Dict[str, Any],
    n_samples: int,
    data_name: str,
    fin_thresh: float
) -> Tuple[Dict[str, bool], Dict[str, List[int]]]:
    """Determine if a sample should be finished.

    Args:
        q:
        n_samples:
        data_name:
        fin_thresh:
    
    Returns:
        finishing_map: which samples have an answer(s) as the next step
        potential_ans: for each sample, generation ids which contain an answer
    """
    finishing_map = {}
    potential_ans = {}
    for sample_id in range(n_samples):
        next_steps = questions.get_single_steps(q, sample_id, 0)
        gen_ans = np.array([
            EXTRACTOR_MAP[data_name](next_step)
            for next_step in next_steps
        ])
        is_fin = gen_ans != '[invalid]'
        finishing_map[sample_id] = True if is_fin.sum() >= fin_thresh else False
        ans = np.where(is_fin)[0]
        potential_ans[sample_id] = ans if ans.size > 0 else np.array([np.nan])
    return finishing_map, potential_ans

def best_of_n(
    q: Dict[str, Any],
    data_name: str,
) -> bool:
    """Get best of N vote.

    Args:
        q:
        data_name: dataset being scored
    
    Returns:
        accuracy: BoN accuracy
    """
    next_steps = questions.get_single_steps(q, 0, -1)
    gen_ans = np.array([
        EXTRACTOR_MAP[data_name](next_step)
        for next_step in next_steps
    ])
    ans_votes = {}
    for ans in gen_ans:
        if ans in ans_votes:
            ans_votes[ans] += 1
        else:
            ans_votes[ans] = 1
    ans_mode = max(ans_votes, key=ans_votes.get)
    accuracy = ans_mode == q['answ_label']

    return accuracy