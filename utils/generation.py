import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from logging import Logger
from typing import List, Dict, Any

from . import monitors

MODEL_FP = '/users/PCON0381/dasi06/nlp/models/Meta-Llama-3.1-8B-Instruct/'

def perform_rollouts(
    llm: LLM,
    qs: List[Dict[str, Any]],
    n_gen: int,
    all_unf_ids: List[int], # this might be List[List[int]]
    log: Logger,
) -> None:
    """Complete LLM prompt for a batch of questions.

    Args:
        llm: language model
        qs: questions and metadata being generated
        n_gen: number of rollouts per question
        all_unf_ids: samples which still need generation (across all questions)
        log: tracks program performance
    
    Returns:
        None    
    """
    sampling_params = SamplingParams(
        n=n_gen, temperature=1.0, top_p=0.9, max_tokens=512
    )
    # Format promtps from all questions
    prompts = []
    for q, unf_sample_id in zip(qs, all_unf_ids):
        for sample_id in unf_sample_id:
            prompts.append(q['prompts_tmp'][sample_id])
    # Generate responses and format properly
    all_gen = llm.generate(prompts, sampling_params)
    #log.info(monitors.get_RAM_usage('after step gen'))
    log.info(monitors.get_VRAM_usage('after step gen'))
    # Loops: Question -> Sample -> Rollouts
    for q, unf_sample_id in zip(qs, all_unf_ids):
        for sample_id in unf_sample_id:
            clean_gens = []
            for gen_id in range(n_gen):
                gen_text = all_gen[0].outputs[gen_id].text
                # Split generation into clean steps
                clean_steps = [
                    step.strip() for step in gen_text.split('\n')
                    if step.strip() != ''
                ]
                clean_gens.append(clean_steps)
            # Free memory
            del all_gen[0]
            torch.cuda.empty_cache()
            q['gen_tmp'][sample_id] = clean_gens
    monitors.manage_mem()
    #log.info(monitors.get_RAM_usage('cleared step gen'))
    log.info(monitors.get_VRAM_usage('cleared step gen'))
    return None

def load_model() -> LLM:
    """Load LLM into memory with vllm.

    Args:
        None
    
    Returns:
        llm: vllm LLM    
    """
    # Check this link https://docs.vllm.ai/_/downloads/en/v0.4.3/pdf/
    # for the error about preemption
    # do I need more RAM/CPU?
    llm = LLM(
        model=MODEL_FP,
        dtype='float16',
        max_model_len=512,
        max_num_batched_tokens=4096,
        enforce_eager=True,
        # gpu_memory_utilization=0.90,
        # kv_cache_dtype="fp8",
        # quantization="bitsandbytes",
        # load_format="bitsandbytes",
        # quantization="awq",
    )
    return llm

def load_tokenizer() -> AutoTokenizer:
    """Load tokenizer.

    Args:
        None

    Returns:
        tokenizer: LM tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FP, use_fast=True)
    return tokenizer