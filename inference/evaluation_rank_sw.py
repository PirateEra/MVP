import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
import argparse
from tqdm import tqdm

import pandas as pd
import pickle
import jsonlines
import json

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import sys
from pathlib import Path
# from FiDT5 import FiDT5
from FiDT5 import FiDT5
import random
from beir_eval import run_direct_rerank_eval
from beir_length_mapping import BEIR_LENGTH_MAPPING

from deepspeed.profiling.flops_profiler import FlopsProfiler

import copy

sys.setrecursionlimit(10000)

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)



def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    data = []
    with jsonlines.open(path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

class ListT5Evaluator():
    def __init__(self, args):
        self.idx = 0
        self.imsi = []
        self.args = args
        local_files_only = False
        if 'jeong' in self.args.model_path:
            local_files_only = True
        self.tok = T5Tokenizer.from_pretrained(self.args.model_path, local_files_only=local_files_only, legacy=False)
        
        # For Evaluate all datasets (Using Folder as input_path)
        if not os.path.isdir(self.args.input_path):
            self.test_file = read_jsonl(self.args.input_path)
            print(f"Input path: {self.args.input_path}")
        # self.idx2tokid = self.tok.encode(' '.join([str(x) for x in range(1, self.args.listwise_k+1)]))[:-1]
        self.model = self.load_model()
            
        self.num_forward = 0

    def write_json_file(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Writing to {path} done!")

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
        print(f"Pooling type: {self.args.pooling_type}")
        model = FiDT5.from_pretrained(self.args.model_path,n_passages = self.args.topk, pooling_type=self.args.pooling_type,
                                       n_special_tokens=self.args.n_special_tokens, tokenizer=self.tok).to('cuda')

        end = time.time()
        print(f"Done! took {end-start} second")
        model.eval()
        if self.args.measure_flops:
            self.prof = FlopsProfiler(model)
            self.prof.start_profile()
        return model

    def make_input_tensors(self, texts):
        raw = self.tok(texts, return_tensors='pt',
                padding=self.args.padding, max_length=self.args.max_input_length,
                truncation=True).to('cuda')
        input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0),
                'attention_mask': raw['attention_mask'].unsqueeze(0)}
        return input_tensors
    
    def make_listwise_text(self, question, ctxs, sep='|'):

        if self.args.pooling_type == 'rv':
            for i in range(len(ctxs)):
                if self.args.n_special_tokens > 1:
                    special_str = "".join([f"<Relevance_{x}>" for x in range(1, 1+self.args.n_special_tokens)])
                    text = f"{special_str} | Query: {question} | Context: {ctxs[i]}"                    
                else:
                    text = f"<Relevance> | Query: {question} | Context: {ctxs[i]}"
                out.append(text)
        else:
            for i in range(len(ctxs)):
                if self.n_special_tokens > 1:
                    special_str = "".join([f"<extra_id_{x}>" for x in range(0, self.args.n_special_tokens)])
                    text = f"{special_str} | Query: {question} | Context: {ctxs[i]}"
                # text = f"<extra_id_17>, Query: {question}, Context: {ctxs[i]}"
                
                out.append(text)
        return out


    def run_inference(self, input_tensors):
        output = self.model.generate_by_single_logit(**input_tensors,
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
        for i, x in enumerate(gen_out):
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
        # print(question, len(ctxs))
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
    
    def sliding_window(self, question, items):
        # start_pos, end_pos = rank_end - window_size, rank_end
        window_size = self.args.window_size
        stride = self.args.step_size
        rerank_topk = self.args.rerank_topk

        reranked_result = copy.deepcopy(items)
        reranked_list = list(range(len(items)))
        # check_dup = set(range(len(items)))
        window_size = self.args.window_size or len(items)

        # if rerank_topk is larager than window-size, Doing multiple reranking by changing the rank_start point.
        # Caluclate the number of reranking
        if self.args.num_iter == -1:
            num_iter = int(np.ceil(self.args.rerank_topk / (window_size - stride)))
        else:
            num_iter = self.args.num_iter
        rank_start = 0

        # print(f"Number of reranking: {num_iter}. Window size: {window_size}. Stride: {stride}")
        for _ in range(num_iter):
            rank_end = len(items)
            start_pos, end_pos = rank_end - window_size, rank_end
            while rank_end > rank_start and start_pos + stride != rank_start:
                
                start_pos = max(start_pos, rank_start)
                self.model.n_passages = end_pos - start_pos
                topk_ctxs = [x[self.args.text_key] for x in reranked_result[start_pos:end_pos]]
                permutation = [(int(x) - 1) for x in self.direct_rerank(question, topk_ctxs, k = end_pos-start_pos)]
                # print("Permutation: ", permutation)
                cut_range = copy.deepcopy(reranked_result[start_pos:end_pos])
                temp = copy.deepcopy(reranked_list[start_pos:end_pos])
                # import IPython;
                # IPython.embed()
                # exit()
                for local_rank, index in enumerate(permutation):
                    reranked_result[start_pos+local_rank] = cut_range[index]
                    reranked_list[start_pos+local_rank] = temp[index]
                # print(reranked_list)
                excepted = list(range(len(items)))

                assert sorted(reranked_list) == excepted, f"reranked_list does not contain all values from 0 to {len(excepted)}: {start_pos}"
                # import IPython;
                # IPython.embed()
                start_pos, end_pos = start_pos - stride, end_pos - stride
            rank_start = rank_start + (window_size - stride)

        return [str(x+1) for x in reranked_list]
    
    def run_direct_rerank(self):
        reranked_instances = []
        len_question = []
        for instance in tqdm(self.test_file):
            
            question = instance[self.args.question_text_key]
            items = instance[self.args.firststage_result_key][:self.args.topk]
            topk_ctxs = [x[self.args.text_key] for x in items]
            qrels = instance[self.args.qrels_key]
            self.model.n_passages = len(topk_ctxs)
            len_question.append(len(question))
            
            if len(topk_ctxs) > 0:
                index = self.sliding_window(question, items)
                # print(index)
            else:
                # If no candidate passages are available, skip the instance
                index = []
            
            reranked_items = []
            for i, pid in enumerate(index):
                pid = int(pid) - 1
                template  = items[pid]
                template['orig_'+self.args.score_key] = template[self.args.score_key]
                if 'rev' in self.args.model_path:
                    template[self.args.score_key] = 100000 - i                
                else:
                    template[self.args.score_key] = 100000 + i
                reranked_items.append(template)
            instance[self.args.firststage_result_key] = reranked_items
            reranked_instances.append(instance)
         
        self.write_jsonl_file(self.args.output_path, reranked_instances)
        ndcg_k, scores = run_direct_rerank_eval(self.args.output_path, k=self.args.topk)
        
        return ndcg_k, scores

def run_reranker(args):
    module = ListT5Evaluator(args)

    he = time.time()
    ndcg_10, scores = module.run_direct_rerank()
    hehe = time.time()
    print(f"Total elapsed time: {hehe-he}")    
    print("Elasped time per query: ", (hehe-he)/len(module.test_file))
    if args.measure_flops:
        flops = module.flops
        num_forward = module.num_forward
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
        module.args.output_path = f'{output_path}/listt5-{file_name}'
        ndcg_k, score = module.run_direct_rerank()
        ndcg_10[key] = ndcg_k
        scores[key] = score
        module.args.max_input_length = -1
    
    return ndcg_10, scores, 0, 0

def main():
    parser = argparse.ArgumentParser()
    # Dataset key setup
    parser.add_argument('--firststage_result_key', default='bm25_results', type=str)
    parser.add_argument('--docid_key', default='docid', type=str)
    parser.add_argument('--pid_key', default='pid', type=str)
    parser.add_argument('--qrels_key', default='qrels', type=str)
    parser.add_argument('--score_key', default='bm25_score', type=str)
    parser.add_argument('--question_text_key', default='q_text', type=str)
    parser.add_argument('--text_key', default='text', type=str)
    parser.add_argument('--title_key', default='title', type=str)
    parser.add_argument('--pooling_type', default=None, type=str)
    parser.add_argument('--n_special_tokens', default=1, type=int)
    parser.add_argument('--store_result', default=False, type=bool)

    parser.add_argument('--model_path', default='Soyoung97/ListT5-base', type=str)
    parser.add_argument('--topk', default=100, type=int, help='number of initial candidate passages to consider') 
    parser.add_argument('--score_mode', default='default', type=str, help='default or logit')
    # or 1000
    parser.add_argument('--max_input_length', type=int, default=-1) # depends on each individual data setup
    parser.add_argument('--padding', default='max_length', type=str)
    parser.add_argument('--rerank_topk', default=10, type=int)
    parser.add_argument('--decoding_strategy', default='single', type=str)
    parser.add_argument('--target_seq', default='token', type=str)

    # Sliding window
    parser.add_argument('--window-size', default=20, type=int)
    parser.add_argument('--step-size', default=10, type=int)
    parser.add_argument('--wrap-encoder-batch', default=100, type=int) # Because of the memory issue, we need to Devide the input into small batch size. (max_input_length -> wrap_encoder_batch. 256 -> 100, 512 -> 50. 1024 -> 25 in 24GB gpu)
    parser.add_argument('--num_iter', default=-1, type=int)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--input_path', type=str, default='./trec-covid.jsonl')
    parser.add_argument('--output_path', type=str, default='./outputs/trec-covid.jsonl')
    parser.add_argument('--special_loc', default=0, type=int)
    parser.add_argument('--topk_selection', default=1, type=int)
    parser.add_argument('--encoder_batch_size', default=100, type=int)

    # profiling setup
    parser.add_argument('--measure_flops', action='store_true')
    parser.add_argument('--skip_no_candidate', action='store_true', help='skip instances with no gold qrels included at first-stage retrieval for faster inference, only works when gold qrels are available')
    parser.add_argument('--skip_issubset', action='store_true', help='skip the rest of reranking when the gold qrels is a subset of reranked output for faster inference, only works when gold qrels are available')
    args = parser.parse_args()
    if args.measure_flops:
        from deepspeed.profiling.flops_profiler import FlopsProfiler
    res = {}
    random.seed(args.seed)
    args.max_gen_length = args.topk + 1
    
    print(args)
    # Kwon JUN
    # 만약에 input_path가 폴더 이름일 경우에는 모든 데이터셋에 대해서 rerank를 수행한다.
    if os.path.isdir(args.input_path):
        ndcg_10, scores, flops, num_forwards = run_reranker_all(args)
        print("========= RESULT =========")
        print(f"Model path: {args.model_path}")
        print(args.model_path)
        for key in ndcg_10:
            print(f"{key}\t{ndcg_10[key]}")
        print("========= RESULT =========")
        return
    
    if args.max_input_length == -1:
        input_path = args.input_path.split('/')[-1]
        for name in BEIR_LENGTH_MAPPING:
            if name in input_path:
                args.max_input_length = BEIR_LENGTH_MAPPING[name]
        if args.max_input_length == -1:
            print(f"Could not find automatic max_input_length assignment from the following dataset keys: {BEIR_LENGTH_MAPPING.keys()}. Please modify the input_length data name or specify max input length by giving it by arguments.")
            raise Exception
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    start_time = time.time()
    ndcg_10, scores, flops, num_forwards = run_reranker(args)
    res['flops'] = flops
    res['num_forwards'] = num_forwards
    res[args.output_path] = scores
    res['ndcg@10'] = ndcg_10
    end_time = time.time()
    res['time_duration'] = end_time - start_time
    print("========= RESULT =========")
    print(args.model_path)
    print(args.pooling_type)
    print(res)
    print("========= RESULT =========")
    return res



if __name__ == '__main__':
    main()
