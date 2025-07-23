import argparse
from tqdm import tqdm
import jsonlines
from transformers import T5Tokenizer
from pathlib import Path
from FiDT5 import FiDT5
import random
from beir_eval import run_direct_rerank_eval
from beir_length_mapping import BEIR_LENGTH_MAPPING
import time
from deepspeed.profiling.flops_profiler import FlopsProfiler

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def read_jsonl(path):
    data = []
    with jsonlines.open(path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

class ListT5Evaluator():
    def __init__(self, args):
        self.args = args
        self.tok = T5Tokenizer.from_pretrained(self.args.model_path)

        if not os.path.isdir(self.args.input_path):
            self.test_file = read_jsonl(self.args.input_path)
            print(f"Input path: {self.args.input_path}")
        self.model = self.load_model()
        self.num_forward = 0

    def write_jsonl_file(self, path, data):
        if self.args.measure_flops:
            self.prof.stop_profile()
            self.flops = self.prof.get_total_flops()
        else:
            self.flops = 0
        print(f"Flops: {self.flops}!")
        with jsonlines.open(path, 'w') as writer:
            writer.write_all(data)
        print(f"Writing to {path} done!")

    def load_model(self):
        start = time.time()
        print("Loading model..")
        print(f"Loading fid model from {self.args.model_path}")
        model = FiDT5.from_pretrained(
            self.args.model_path,
            n_passages = self.args.topk,
            n_special_tokens=self.args.n_special_tokens,
            tokenizer=self.tok).to('cuda')
        end = time.time()
        
        print(f"Done! took {end-start} second")
        model.eval()
        if self.args.measure_flops:
            self.prof = FlopsProfiler(model)
            self.prof.start_profile()        
        return model

    def make_input_tensors(self, texts):
        raw = self.tok(
            texts,
            return_tensors='pt',
            padding=self.args.padding,
            max_length=self.args.max_input_length,
            truncation=True).to('cuda')
        input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0),
                'attention_mask': raw['attention_mask'].unsqueeze(0)}
        return input_tensors

    def make_listwise_text(self, question, ctxs, sep='|'):
        out = []
        for i in range(len(ctxs)):
            if self.args.n_special_tokens >= 1:
                special_str = "".join([f"<extra_id_{x}>" for x in range(0, self.args.n_special_tokens)])
                text = f"{special_str} | Query: {question} | Context: {ctxs[i]}"
            else:
                text = f"<extra_id_0> | Query: {question} | Context: {ctxs[i]}"            
            out.append(text)
        return out

    def run_inference(self, input_tensors):
        output = self.model.generate_by_single_logit(
            **input_tensors,
            max_length = self.args.max_gen_length,
            return_dict=False),
        self.num_forward += 1
        
        return output[0]
 
    def get_rel_index(self, output, mode='default', k=-1):
        if k == -1:
            k = self.args.out_k
        
        gen_out = None
        topk_possible = [str(x) for x in range(1, k+1)]
        
        if mode=='default':
            gen_out = self.tok.batch_decode(output.sequences, skip_special_tokens=True)
            gen_out = gen_out[0].split(' ')
        elif mode=='logit':
            topk_logits = output.scores[0].topk(k + 10).indices
            gen_out = [x.split() for x in self.tok.batch_decode(topk_logits, skip_special_tokens=True)][0]
        
        print("Model output: ", gen_out)        
        out_rel_indexes = []
        for x in enumerate(gen_out):
            if x in topk_possible:
                out_rel_indexes.append(x)
                topk_possible.remove(x)
        
        if len(out_rel_indexes) < k:
            if 'rev' in self.args.model_path:
                out_rel_indexes = out_rel_indexes + topk_possible
            else:    
                out_rel_indexes = topk_possible[::-1] + out_rel_indexes

        return out_rel_indexes

    def direct_rerank(self, question, ctxs, k=-1):
        full_input_texts = self.make_listwise_text(question, ctxs)
        try:
            input_tensors = self.make_input_tensors(full_input_texts)
        except:
            import IPython;
            IPython.embed()
            exit()
        output = self.run_inference(input_tensors)

        out_k_rel_index = [str(x+1) for x in output[0]]

        return out_k_rel_index
    
    def run_direct_rerank(self):
        reranked_instances = []
        len_question = []
        for instance in tqdm(self.test_file):
            question = instance[self.args.question_text_key]
            items = instance[self.args.firststage_result_key][:self.args.topk]

            if self.args.initial == 'origin':
                pass
            elif self.args.initial == 'reverse':
                items = items[::-1]
            elif self.args.initial == 'random':
                random.shuffle(items)
            
            topk_ctxs = [f"{x[self.args.title_key]} {x[self.args.text_key]}".strip() for x in items]
            self.model.n_passages = len(topk_ctxs)
            len_question.append(len(question))

            if len(topk_ctxs) > 0:
                index = self.direct_rerank(question, topk_ctxs, k=self.args.topk)
            else:
                index = []

            reranked_items = []

            for i, pid in enumerate(index):
                pid = int(pid) - 1
                template  = items[pid]
                template['orig_'+self.args.score_key] = template[self.args.score_key]
                template[self.args.score_key] = 100000 - i                
                reranked_items.append(template)

            instance[self.args.firststage_result_key] = reranked_items
            reranked_instances.append(instance)
        self.write_jsonl_file(self.args.output_path, reranked_instances)
        ndcg_k, scores = run_direct_rerank_eval(self.args.output_path, k=self.args.topk)

        return ndcg_k, scores

def run_reranker(args):
    module = ListT5Evaluator(args)

    start = time.time()
    ndcg_10, scores = module.run_direct_rerank()
    end = time.time()
    print(f"Total elapsed time: {end-start}")    
    print("Elasped time per query: ", (end-start)/len(module.test_file))
    if args.measure_flops:
        flops = module.flops
        num_forward = module.num_forward
        print(f"Total number of forward passes: {num_forward}")
        print(f"Total flops: {flops}")
    else:
        flops = 0
        num_forward = 0

    return ndcg_10, scores, flops, num_forward


# IF input_path is a folder, run reranker for all files in the folder
def run_reranker_all(args):
    module = ListT5Evaluator(args)

    ndcg_10 = {}
    scores = {}
    import glob
    files = glob.glob(f'{module.args.input_path}*.jsonl')
    output_path = args.output_path
    for file in files:
        
        if module.args.max_input_length == -1:
            input_path = file.split('/')[-1]
        for name in BEIR_LENGTH_MAPPING:
            if name in input_path:
                module.args.max_input_length = BEIR_LENGTH_MAPPING[name]
        if args.max_input_length == -1:
            print(f"Could not find automatic max_input_length assignment from the following dataset keys: {BEIR_LENGTH_MAPPING.keys()}. Please modify the input_length data name or specify max input length by giving it by arguments.")
            raise Exception

        
        print(f"Processing {file}")
            
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        
        module.test_file = read_jsonl(file)
        file_name = file.split("/")[-1]
        key = file_name.split('.')[0]
        module.args.output_path = f'{output_path}/mvt5-{file_name}'
        ndcg_k, score = module.run_direct_rerank()
        ndcg_10[key] = ndcg_k
        scores[key] = score
        module.args.max_input_length = -1
    
    return ndcg_10, scores, 0, 0

def main():
    parser = argparse.ArgumentParser()
    # default key setup
    parser.add_argument('--question_text_key', default='q_text', type=str)
    parser.add_argument('--firststage_result_key', default='bm25_results', type=str)
    parser.add_argument('--score_key', default='bm25_score', type=str)
    parser.add_argument('--text_key', default='text', type=str)
    parser.add_argument('--title_key', default='title', type=str)
    
    # default model setup
    parser.add_argument('--model_path', default='bulbna/MVP-base', type=str)
    parser.add_argument('--input_path', type=str, default='./eval_data/dl19.jsonl')
    parser.add_argument('--output_path', type=str, default='./outputs/dl19.jsonl')
    parser.add_argument('--topk', default=100, type=int, help='number of initial candidate passages to consider') 
    parser.add_argument('--max_input_length', type=int, default=-1) # depends on each individual data setup
    parser.add_argument('--padding', default='max_length', type=str)
    parser.add_argument('--n_special_tokens', default=1, type=int)

    # Position bias setup
    parser.add_argument('--initial', default='origin', type=str)
    parser.add_argument('--seed', default=0, type=int)
    
    # profiling setup
    parser.add_argument('--measure_flops', action='store_true')

    args = parser.parse_args()
    res = {}
    random.seed(args.seed)
    args.max_gen_length = args.topk + 1

    if args.max_input_length == -1:
        input_path = args.input_path.split('/')[-1]
        for name in BEIR_LENGTH_MAPPING:
            if name in input_path:
                args.max_input_length = BEIR_LENGTH_MAPPING[name]
        if args.max_input_length == -1:
            print(f"Could not find automatic max_input_length assignment from the following dataset keys: {BEIR_LENGTH_MAPPING.keys()}. Please modify the input_length data name or specify max input length by giving it by arguments.")
            raise Exception
        
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    ndcg_10, scores, flops, num_forwards = run_reranker(args)
    
    return res



if __name__ == '__main__':
    main()
