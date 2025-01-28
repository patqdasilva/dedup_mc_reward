import json
import jsonlines
from typing import Generator, List, Dict, Any

from transformers import AutoTokenizer

from . import monitors, prompts, step_selection as ss

class QuestionReader:
    def __init__(self, fp: str, restart_idx: int):
        """Initialize file path for the questions."""
        self.fp = fp
        self.restart_idx = restart_idx

    def get_quesiton(self, batch_size: int = 1) -> Generator[List[Dict[str, Any]], None, None]:
        """Generator to yield questions."""
        try:
            with jsonlines.open(self.fp) as reader:
                batch = []
                batch_start_idx = None
                for idx, line in enumerate(reader):
                    if idx < self.restart_idx:
                        continue # Skip line, restart loop
                    if batch_start_idx is None:
                        batch_start_idx = idx
                    batch.append(line)
                    if len(batch) == batch_size:
                        yield (batch, batch_start_idx, idx)
                        batch = []
                        batch_start_idx = None
                if batch: # if remaining lines < batch_size
                    yield (batch, batch_start_idx, idx)         
        except FileNotFoundError:
            print(f"Error: The file {self.fp} was not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from the file {self.fp}.")

def format_prompt(qs: List[Dict[str, Any]]) -> None:
    """Prepare a batch of the dataset for inference/grading.

    Args:
        qs: questions and metadata being generated

    Returns:
        None
    """
    # Format prompts for Multiple Choice or Open Answer
    prompt_templates = prompts.PROMPTS
    for i, q in enumerate(qs):
        is_mc = q['answ_opt']
        if is_mc:
            label_answ = zip(
                q['answ_opt'],
                ['A: ', 'B: ', 'C: ']
            )
            options = (
                ' from the following options:'
                + ',\n'.join(
                    f'{label} "{answ}"'
                    for answ, label in label_answ
                )
            )
        else:
            options = '.'
        prompt = prompt_templates['prompt'].format(
            question=q['question'],
            options=options
        )
        # Apply chat template
        chat_prompt = prompt_templates['llama_chat'].format(prompt=prompt)
        qs[i]['base_prompt'] = chat_prompt
    return None

def init_qs(
    qs: List[Dict[str, Any]],
    n_samples: int
) -> None:
    """Create data structure to hold the keys below.

    Args:
        qs: questions and metadata being generated
        n_samples: number of samples per seed prompt
    
    Returns:
        None    
    """
    # Can return to make this a defaultdict for brevity
    keys = [
        'gen_tmp', 'prompts_tmp',
        'chosen_steps',
        'acc', 'grades',
        'sim_mean'
    ]
    for q in qs:
        for key in keys:
            q[key] = [[] for _ in range(n_samples)]
    return None

def update_prompts(
    qs: List[Dict[str, Any]],
    n_samples: int,
) -> None:
    """Append chosen steps to the prompt for each sample.

    Args:
        qs: questions and metadata being generated
        n_samples: number of samples per seed prompt
    
    Returns:
        None
    """
    for q in qs:
        for sample_id in range(n_samples):
            q['prompts_tmp'][sample_id] = (
                q['base_prompt']
                + "\n\nThe start of Assistant's step by step breakdown\n"
                + '\n'.join(q['chosen_steps'][sample_id])
                + '\n'
            )
    return None

def wipe_tmp_vals(qs):
    """Reset temporary values from question batch to free space.

    Args:
        qs: questions and metadata being generated
    
    Returns:
        None    
    """
    keys = [
        'gen_tmp', 'prompts_tmp',
        'grades',
        'sim_mean',
    ]
    for q in qs:
        for key in keys:
            del q[key]
    return None

def get_single_steps(
    q: Dict[str, Any],
    sample_id: int,
    step_idx: int
) -> List[str]:
    """Return a step from each generation based on an index.

    Args:
        q: data from the samples/generations of a single prompt
        sample_id: the index of a sample for a prompt
        step_idx:
            index from the reasoning trace from which to extract a step
            e.g. step_idx=0 means first step, step_idx=-1 means last step
    Returns:
        steps: a step for each generation
    """
    steps = [
        gens[step_idx]
        if gens else ''
        for gens in q['gen_tmp'][sample_id]
    ]
    return steps

def get_all_steps(
    q: Dict[str, Any],
    n_samples: int,
    unf_sample_ids: List[str]
) -> List[List[str]]:
    """Get all first steps and banned steps for a question.

    Args:
        q: data from the samples/generations of a single prompt
        n_samples: number of samples per seed prompt
        unf_sample_ids: which
        
    Returns:
        all_steps: all first steps and banned steps for a question
    """
    all_steps = []
    for sample_id in range(n_samples):
        if sample_id in unf_sample_ids:
            first_steps = get_single_steps(q, sample_id, 0)
        else:
            first_steps = []
        all_steps.append(first_steps)
    # Get all previously chosen steps too
    # Loop thorugh sample_ids again because I want
    # chosen steps to be placed after all currently generated steps
    for sample_id in range(n_samples):
        if q['chosen_steps'][sample_id]:
            all_steps.append(q['chosen_steps'][sample_id])
    return all_steps

def write_jsonl(
    data: List[Dict[str, Any]],
    fp: str
) -> None:
    """Save data into a jsonline file.

    Args:
        data: line corresponding to a single completed quesiton
        fp: where to save data
    
    Returns:
        None    
    """
    with jsonlines.open(fp, mode='a') as writer:
        writer.write_all(data)
    return None

def check_input_len(
    qs: List[Dict[str, Any]],
    all_unf_ids: List[List[int]],
    all_completed_map: List[Dict[int, bool]],
    tokenizer: AutoTokenizer,
) -> None:
    """Check all generations against max model len (512).
    
    Make any samples that do "complete" and remove them from all_unf_ids
    (Basically the CoT got too long before answering the question 
    or running out of steps)

    Args:
        qs: questions and metadata being generated
        all_unf_ids: samples which still need generation (across all questions)
        all_completed_map: which questions and samples are complete
        tokenizer: LM tokenizer

    Returns:
        None
    """
    for qid in range(len(qs)):
        for sample_id in all_unf_ids[qid]:
            prompt = qs[qid]['prompts_tmp'][sample_id]
            tokens = tokenizer.encode(prompt)
            is_too_long = len(tokens) > 512
            if is_too_long:
                all_completed_map[qid][sample_id] = True
        all_unf_ids[qid] = ss.remove_complete_samples(all_completed_map[qid])
    return None
