print('starting python side')
from sentence_transformers import SentenceTransformer

from utils import monitors, questions, generation, mce

import argparse

parser = argparse.ArgumentParser(
    prog='gen.py',
    description='Generate Synthetic Data for PRM Training',
)
parser.add_argument('-t', '--trial')
parser.add_argument('-r', '--restart_idx')
args = parser.parse_args()
print(args)

# Initialize
n_samples = 10
n_gen = 8
max_iter_idx = 10 # range(1, max_iter_idx)
sim_thresh = 0.85
fin_thresh = 1
data_name = 'gsm'
batch_size = 30

if args.trial == 'random':
    rand_step_select = True
else:
    rand_step_select = False
is_test = 'train'

if args.trial == 'eval':
    rand_step_select = True
    is_test = 'test'
    n_samples = 1
    batch_size = 100

fp = f'./format_questions/questions/{data_name}_qs_{is_test}.jsonl'
save_fp = f'./synth_samples/{data_name}/{args.trial}.jsonl'
data_gen_log_fp = f'./logs/data_gen/{data_name}_{args.trial}.csv'
q_reader = questions.QuestionReader(fp, int(args.restart_idx))
log = monitors.make_logger(data_gen_log_fp)

log.info(f'BEGINNING SESSION')
log.info(monitors.get_RAM_usage('baseline'))
log.info(monitors.get_VRAM_usage('baseline'))
tokenizer = generation.load_tokenizer()
sent_sim_model = SentenceTransformer('/users/PCON0381/dasi06/nlp/models/all-MiniLM-L6-v2')
llm = generation.load_model()
log.info(monitors.get_RAM_usage('model loaded'))
log.info(monitors.get_VRAM_usage('model loaded'))

mce.perform_mce(
    q_reader,
    batch_size,
    n_samples,
    n_gen,
    rand_step_select,
    llm,
    sent_sim_model,
    tokenizer,
    sim_thresh,
    data_name,
    fin_thresh,
    max_iter_idx,
    save_fp,
    log
)
