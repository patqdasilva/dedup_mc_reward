print('starting python stuff')
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import torch

import argparse
parser = argparse.ArgumentParser(
    prog='gen.py',
    description='Generate Synthetic Data for PRM Training',
)
parser.add_argument('-t', '--trial')
parser.add_argument('-n', '--n_eval_steps')
args = parser.parse_args()
print(args)

data_name = 'gsm'

# Load the dataset (jsonl so load in two files for trian/test)
data_train_fp = f'./format_questions/prm_train/{data_name}/{args.trial}.jsonl'
data_test_fp = f'./format_questions/prm_train/{data_name}/eval.jsonl'
data = load_dataset("json", data_files={'train': data_train_fp, 'test': data_test_fp})

print('load model')
#model_fp = '/fs/ess/PAS2570/short_ans_feedback/models/Mistral-7B-Instruct-v0.2'
# model_fp = '/users/PCON0381/dasi06/nlp/models/math-shep-mistral-7b-sft'
model_fp = '/users/PCON0381/dasi06/nlp/models/Meta-Llama-3.1-8B-Instruct/'
adapter_fp = f'/users/PCON0381/dasi06/nlp/models/dedup/checkpoints/{data_name}_{args.trial}/checkpoint-3018_first'
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_fp,
    device_map="auto",
    trust_remote_code=False,
    quantization_config=nf4_config
)

tokenizer = AutoTokenizer.from_pretrained(model_fp, use_fast=True)

print('set tokenizer configs')
tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def formatting_prompts_func(samples):
    output_texts = []
    for prompt, completion in zip(samples['prompt'], samples['completion']):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{completion}"
        output_texts.append(text)
    return output_texts
response_template_with_context = "<|end_header_id|>"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=f'/users/PCON0381/dasi06/nlp/models/dedup/checkpoints/{data_name}_{args.trial}/',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_train_epochs=1,
    optim="adamw_bnb_8bit",
    # Save
    save_strategy='epoch',
    save_safetensors=True,
    # Eval
    eval_strategy='steps',
    per_device_eval_batch_size = 1,
    eval_accumulation_steps=8,
    eval_steps=int(args.n_eval_steps),
    # Log steps
    logging_strategy='steps',
    logging_steps=int(args.n_eval_steps),
    logging_first_step=True,
    logging_dir=f'./logs/training/tensorboard/{data_name}_{args.trial}',  # Directory to store logs for TensorBoard
    report_to='tensorboard'  # Enable TensorBoard reporting
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    formatting_func=formatting_prompts_func,
    train_dataset=data['train'],
    eval_dataset=data['test'],  # Add test dataset for evaluation
    peft_config=lora_config,
    max_seq_length=1024,
)
print('start training')
trainer.train() # resume_from_checkpoint = True