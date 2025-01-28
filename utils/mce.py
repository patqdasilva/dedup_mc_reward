from pprint import pprint
from logging import Logger

from vllm import LLM
from transformers import AutoTokenizer

from . import questions, step_selection as ss, generation, scoring, monitors


def perform_mce(
    q_reader: questions.QuestionReader,
    batch_size: int,
    n_samples: int,
    n_gen: int,
    rand_step_select: bool,
    llm: LLM,
    sent_sim_model: ,
    tokenizer: AutoTokenizer,
    sim_thresh: float,
    data_name: str,
    fin_thresh: float,
    max_iter_idx: int,
    save_fp: str,
    log: Logger,
) -> None:
    """Perform Deduplicated Monte Carlo Reward Sampling.

    Args:
        q_reader: reads in batches of questions from a JSONL file
        batch_size: how many questions to pass through LLM at once
        n_samples: number of samples per seed prompt
        n_gen: number of rollouts per sample
        rand_step_select: whether to choose the next step randomly
        llm: a vLLM LLM
        sent_sim_model: sentence-transformers/all-MiniLM-L6-v2
            maps sentences & paragraphs to a 384 dimensional dense vector space
            and can be used for tasks like clustering or semantic search.
        tokenizer: LM tokenizer
        sim_thresh:
            embedding similarity threshold for steps to be considered equal
        data_name:
            dataset options = {GSM8k, BBQ, SQA?}
        fin_thresh:
            How many of the steps need to have found the final answer
            before being considered finsihed
        max_iter_idx: upper limit of steps to generate
        save_fp: where to save outputs
        log: tracks program performance

    Returns:
        None
    """
    for qs, start_id, end_id in q_reader.get_quesiton(batch_size=batch_size):
        log.info(f'starting batch with idxs\t{start_id}\t{end_id}')
        questions.format_prompt(qs)
        questions.init_qs(qs, n_samples)
        # Initialize for start of qs
        all_completed_map = [
            {sample_id: False for sample_id in range(n_samples)}
            for _ in range(len(qs))
        ]
        all_unf_ids = [
            ss.remove_complete_samples(all_completed_map[qid])
            for qid in range(len(qs))
        ]
        for iter_idx in range(1, max_iter_idx):
            questions.update_prompts(qs, n_samples)
            questions.check_input_len(
                qs, all_unf_ids, all_completed_map, tokenizer
            )
            generation.perform_rollouts(
                llm, qs,
                n_gen, all_unf_ids,
                log
            )
            for qid, q in enumerate(qs):
                if all_unf_ids[qid]:
                    # Do not want certain things to happen on first step init round
                    if iter_idx > 1:
                        scoring.grade_question(
                            q,
                            all_unf_ids[qid],
                            data_name
                        )
                    finishing_map, potential_ans = scoring.is_finishing(
                        q, n_samples,
                        data_name, fin_thresh
                    )
                    all_steps, sim_graph = ss.gen_sim_graph(
                        q,
                        n_samples, n_gen, all_unf_ids[qid],
                        sent_sim_model, sim_thresh
                    )
                    banned = ss.init_banned(n_samples, iter_idx)
                    ss.choose_next_steps(
                        q,
                        n_gen, all_unf_ids[qid],
                        all_steps, sim_graph, banned,
                        sent_sim_model,
                        potential_ans, all_completed_map[qid],
                        rand_step_select, data_name
                    )
                    all_unf_ids[qid] = ss.remove_complete_samples(all_completed_map[qid])
                    monitors.manage_mem(all_steps, sim_graph, banned)
        questions.wipe_tmp_vals(qs)
        questions.write_jsonl(qs, save_fp)
        monitors.manage_mem(qs, force_gc=True)
        log.info(monitors.get_RAM_usage('fin batch'))
        log.info(monitors.get_VRAM_usage('fin batch'))
        log.info(f'complete batch with idxs\t{start_id}\t{end_id}')
        # Optional early exit for debugging
        # if end_id == 59:
        #     break
        # break