from typing import List, Dict, Set, Tuple, Any

from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

from . import monitors, questions, scoring

def calc_step_sim(
    sent_sim_model: SentenceTransformer,
    step: str,
    others: List[str]
) -> torch.Tensor:
    """Calculate the similarity between one reasoning step and others.

    Args:
        sent_sim_model: sentence embedding model (all-MiniLM-L6-v2)
        step: a reasoning step
        others: all other reasoning steps to compare against
    
    Returns:
        similarity_scores:
            pairwise similarity scores between step and others
    """
    # Remove 'Step #:' and 'ки' from comparison
    # step = step[8:-3]
    # others = [other[8:-3] for other in others]
    
    # Remove '#.' from comparison
    step = step[2:]
    others = [other[2:] for other in others]
    # print(step)
    # print(others)
    step_emb = sent_sim_model.encode(step, convert_to_tensor=True)
    others_emb = sent_sim_model.encode(others, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(others_emb, step_emb)
    monitors.manage_mem(step_emb, others_emb)
    return similarity_scores

def get_nodes(
        n_samples: int,
        unf_sample_ids: List[int],
        all_steps: List[List[str]],
) -> List[str]:
    """Get all possible nodes based on samples and generations.

    Args:
        all_steps:
        unf_sample_ids:
        n_samples:
    Returns:
        nodes: all nodes
    """
    n_chosen = len(all_steps) - n_samples
    banned_sample_ids = (
        unf_sample_ids
        + list(range(n_samples, n_samples + n_chosen))
    )
    n_gen_tmp = [len(steps) for steps in all_steps if len(steps) > 0]
    nodes = [
        f'{sample_id}_{gen_id}'
        for i, sample_id in enumerate(banned_sample_ids)
        for gen_id in range(n_gen_tmp[i])
    ]
    return nodes

def init_sim_graph(
    n_samples: int,
    n_gen: int,
    unf_sample_ids: List[int]
) -> Dict[str, List[str]]:
    """Initialize adjacency list for similarity graph.

    Only create keys for samples being generated.

    Args:
        n_gen:
        unf_sample_ids:
    
    Returns:
        sim_graph:
    """
    n_gens = [['']*n_gen for _ in range(len(unf_sample_ids))]
    all_keys = get_nodes(n_samples, unf_sample_ids, n_gens)
    sim_graph = {key: [] for key in all_keys}
    return sim_graph

def init_banned(
    n_samples: int,
    iter_idx: int
) -> Set[str]:
    """Hold nodes banned from selection.

    All potential nodes, some may not exist.
    Does not negatively impact disjoint calc.
    
    Args:
        n_gen:
        n_samples:
        n_incomplete:
        iter_idx:
    
    Returns:
        banned:
    """
    banned_ids = range(n_samples, n_samples + n_samples)
    banned = set([
        f'{sample_id}_{gen_id}'
        for sample_id in banned_ids
        for gen_id in range(iter_idx - 1)
    ])
    return banned

def remove_complete_samples(
    completed_map: Dict[int, bool]
) -> List[int]:
    """Remove completed samples to stop generating data.

    Args:
        completed_map:
    
    Returns:
        unf_sample_ids: samples that still need data collection
    """
    unf_sample_ids = [
        sample_id
        for sample_id, is_fin in completed_map.items()
        if is_fin is False
    ]
    return unf_sample_ids

def gen_sim_graph(
    q: Dict[str, Any],
    n_samples: int,
    n_gen: int,
    unf_sample_ids: List[int],
    sent_sim_model: SentenceTransformer,
    sim_thresh: float
) -> Tuple[List[List[str]], Dict[str, List[str]]]:
    """Calculate global similarity and generate similarity graph.
        
    Args:
        q:
        n_samples:
        n_gen:
        unf_sample_ids:
        n_incomplete:
        sent_sim_model:
        sim_thresh:

    Returns:
        all_steps:
        sim_graph:
    """
    sim_graph = init_sim_graph(n_samples, n_gen, unf_sample_ids)
    all_steps = questions.get_all_steps(q, n_samples, unf_sample_ids)
    for sample_id in unf_sample_ids:
        sim_means = []
        for gen_id in range(n_gen):
            # Get a candidate step
            cand_step = all_steps[sample_id].pop(gen_id)
            # Flatten remaining candidates/banned steps
            other_cand = np.concatenate(all_steps).tolist()
            # Add the candidate step back into the correct spot
            all_steps[sample_id].insert(gen_id, cand_step)
            similarities = calc_step_sim(sent_sim_model, cand_step, other_cand)
            # Build similarity graph
            cand_node = f'{sample_id}_{gen_id}'
            compare_nodes = get_nodes(n_samples, unf_sample_ids, all_steps)
            compare_nodes.remove(cand_node)
            sim_map = dict(zip(compare_nodes, similarities.view(-1).tolist()))
            # Add a node to the sim adj list if it is sufficiently similar to the candidate step
            for node, sim in sim_map.items():
                if sim >= sim_thresh:
                    sim_graph[cand_node].append(node)
            # Save sumarry metrics
            sim_means.append(torch.mean(similarities).item())
        q['sim_mean'][sample_id] = sim_means
    return all_steps, sim_graph

def choose_next_steps(
        q: Dict[str, Any], 
        n_gen: int,
        unf_sample_ids: List[int],
        all_steps: List[List[str]],
        sim_graph: Dict[str, List[str]],
        banned: Set[str],
        sent_sim_model: SentenceTransformer,
        potential_ans: Dict[int, List[int]],
        completed_map: Dict[int, bool],
        rand_step_select: bool,
        data_name: str
    ) -> None:
    """Choose the next step for each sample in a quesiton.
    
    Args:
        q:
        n_gen:
        unf_sample_ids:
        all_steps:
        sim_graph:
        banned:
        sent_sim_model:
        finishing_map: which samples have an answer(s) as the next step
        potential_ans: for each sample, generation ids which contain an answer
        completed_map:
        rand_step_select: controls whether or not to select steps randomly
    Returns:
        None
    """
    banned_steps = [step for steps in q['chosen_steps'] for step in steps]
    # print('\nchoose_next_steps: ----- START OF QUESTION -----\n')
    # print('choose_next_steps: all_steps', all_steps)
    for sample_id in unf_sample_ids:
        # print('\nchoose_next_steps: ----- new sample_id', sample_id)
        # print('choose_next_steps: banned_steps', banned_steps)
        # print('choose_next_steps: banned', banned)
        if rand_step_select:
            cand_choice_id = np.random.choice(range(n_gen))
        # If no steps are banned yet (none chosen for a question)
        elif not banned_steps:
            # print('choose_next_steps: no steps chosen yet')
            # Choose lowest within & between sample similarity
            cand_choice_id = np.argmin(q['sim_mean'][sample_id])
        else:
            # Assemble all non-banned candidate-step indices
            non_dupes = []
            for gen_id in range(n_gen):
                cand_node = f'{sample_id}_{gen_id}'
                cand_edges = sim_graph[cand_node]
                cand_net = set([cand_node] + cand_edges)
                if cand_net.isdisjoint(banned):
                    non_dupes.append(gen_id)
            if len(non_dupes) == 0:
                completed_map[sample_id] = True
                # print('choose_next_steps: no non-dupes')
                # print('choose_next_steps: completed_map', completed_map)
                continue # all candidates are dupes, skip whole sample
            else:
                # Choose candidate step least similar to banned steps
                sims_mean = []
                # print('choose_next_steps: candidate gen_ids', non_dupes)
                for gen_id in non_dupes:
                    cand_step = all_steps[sample_id][gen_id]
                    similarities = calc_step_sim(
                        sent_sim_model, cand_step, banned_steps
                    )
                    sims_mean.append(torch.mean(similarities).item())
                cand_choice_id = non_dupes[np.argmin(sims_mean)]
                monitors.manage_mem(sims_mean)
        next_step = all_steps[sample_id][cand_choice_id]
        # print('choose_next_steps: next_step', next_step)
        # Set the next chosen step for a sample
        q['chosen_steps'][sample_id].append(next_step)
        # ban current candidate step and all related steps
        banned_steps.append(next_step)
        cand_node = f'{sample_id}_{cand_choice_id}'
        cand_edges = sim_graph[cand_node]
        cand_net = set([cand_node] + cand_edges)
        # print('choose_next_steps: cand_node', cand_node)
        # print('choose_next_steps: cand_edges', cand_edges)
        banned.update(cand_net)
        if cand_choice_id in potential_ans[sample_id]:
            # print('choose_next_steps: chose a final answer')
            completed_map[sample_id] = True
            # The reward of an answer step is 1 or 0
            # Depending on it if answers correctly
            step_ans = scoring.EXTRACTOR_MAP[data_name](next_step)
            q['acc'][sample_id].append(
                float(step_ans == q['answ_label'])
            )
    return None

