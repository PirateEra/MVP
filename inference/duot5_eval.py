import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
import pickle
import itertools
import numpy as np
# from genre.trie import Trie
import time
import argparse
from tqdm import tqdm
import jsonlines
import json
from pprint import pprint
import pandas as pd
# from knockknock import slack_sender
import torch
import math
import glob
import wandb
import pickle
from transformers import pipeline
import sys
from pathlib import Path
sys.path.append('../q2q_train_code/')
sys.path.append('../scripts/eval/')
from beir_eval import run_rerank_eval
from FiDT5 import FiDT5
from deepspeed.profiling.flops_profiler import FlopsProfiler
import random

sys.setrecursionlimit(10**7)


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

class Pangu():
    def __init__(self, args):
        self.idx = 0
        self.args = args
        try:
            #if self.args.measure_flops:
            #    self.tok = LlamaTokenizer.from_pretrained('castorini/rank_vicuna_7b_v1')
            #    self.tok.padding_side = 'left'
            #else:
            self.tok = T5Tokenizer.from_pretrained(self.args.model_path, legacy=False)
        except:
            print(f"No tokenizer found for {self.args.model_path}. Backoffing from t5-base")
            self.tok = T5Tokenizer.from_pretrained('t5-base', legacy=False)
        self.test_file = read_jsonl(self.args.test_path)
        if self.args.shuffle:
            new = []
            for instance in self.test_file:
                bm25_res = instance['bm25_results']
                random.shuffle(bm25_res)
                instance['bm25_results'] = bm25_res
                new.append(instance)
            self.test_file = new
        #if self.args.debug:
        #    self.test_file = self.test_file[59:]
        print(self.args.test_path)
        self.idx2tokid = self.tok.encode(' '.join([str(x) for x in range(1, self.args.listwise_k+1)]))[:-1]
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
        print(f"Loading baseline {self.args.sub_mode} model from {self.args.model_path}")
        model = T5ForConditionalGeneration.from_pretrained(self.args.model_path).to('cuda')#, torch_dtype=torch.float32)
        end = time.time()
        print(f"Done! took {end-start} second")
        model.eval()
        if self.args.measure_flops:
            self.prof = FlopsProfiler(model)
            self.prof.start_profile()
        return model

    def make_input_tensors(self, texts):
        raw = self.tok(texts, return_tensors='pt',
                padding=self.args.padding, max_length=self.args.max_length,
                truncation=True).to('cuda')
        input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0),
                'attention_mask': raw['attention_mask'].unsqueeze(0)}
        return input_tensors

    def run_inference(self, input_tensors):
        #if self.args.measure_flops:
        #    output = self.model.generate(input_tensors['input_ids'],
        #            max_new_tokens=92, return_dict_in_generate=True)
        with torch.no_grad():
            if self.args.beam_size == -1:
                output = self.model.generate(**input_tensors,
                    max_length = self.args.max_gen_length,
                    return_dict_in_generate=True, output_scores=True)
            else:
                output = self.model.generate(**input_tensors, num_return_sequences=1,
                        num_beams=self.args.beam_size,
                    max_length = self.args.max_gen_length,
                    return_dict_in_generate=True, output_scores=True)
            self.num_forward += 1
        return output

    def group2chunks(self, l, n=5):
        for i in range(0, len(l), n):
            yield l[i:i+n]
    def run_subeval(self, i, output):
        if i % 500 == 0:
            string = run_rerank_eval(output, combined=True)
            ndcg_10 = get_ndcg_from_string(string)
            print(f"Iter: {i}, ndcg@10: {ndcg_10}")
        return

    def run_baseline(self):
        temp = []
        for i, instance in tqdm(enumerate(self.test_file), total=len(self.test_file)):
            question = instance['q_text']
            topk_ctxs = [f"{x['title']} {x['text']}".strip() for x in instance['bm25_results']]
            if len(topk_ctxs) >= self.args.topk:
                topk_ctxs = topk_ctxs[:self.args.topk]
            full_input_texts = []
            full_input_text_ij_mapping = []
            for i in range(len(topk_ctxs)):
                for j in range(len(topk_ctxs)):
                    if i == j:
                        continue
                    full_input_texts.append(f"Query: {question} Document0: {topk_ctxs[i]} Document1: {topk_ctxs[j]} Relevant:")
                    full_input_text_ij_mapping.append(tuple([i,j]))


            grouped_input_texts = list(self.group2chunks(full_input_texts, n=self.args.bsize))
            scores_holder = []
            for batch_input_texts in tqdm(grouped_input_texts):
                input_tensors = self.tok(batch_input_texts, return_tensors='pt',
                        padding=self.args.padding, max_length=self.args.max_length,
                        truncation=True).to('cuda')
                outputs = self.run_inference(input_tensors)
                del input_tensors
                scores = torch.stack(outputs.scores)
                """
                true_score = scores[0][:, 1176]
                false_score = scores[0][:, 6136]
                symsum_score = (true_score + (1-false_score)).tolist()
                """
                tf_scores = torch.nn.functional.log_softmax(scores[0][:, [1176, 6136]], dim=1)[:, 0].tolist()
                scores_holder += tf_scores
            if len(full_input_text_ij_mapping) != len(scores_holder):
                import pdb; pdb.set_trace()
            full_mapping = {x: y for x,y in zip(full_input_text_ij_mapping, scores_holder)}
            # do aggregation
            agg_scores = {x: 0 for x in range(len(topk_ctxs))}
            for i in range(len(topk_ctxs)):
                for j in range(len(topk_ctxs)):
                    if i == j:
                        continue
                    #score = np.exp(full_mapping[(i,j)])
                    score = full_mapping[(i,j)]
                    agg_scores[i] += score
                    agg_scores[j] += (1-score)
            all_scores_tensor = torch.tensor(list(agg_scores.values()))
            rank = torch.argsort(all_scores_tensor).tolist()
            rank.reverse()
            reranked_instances = []
            for rank_id in rank:
                template = instance['bm25_results'][rank_id]
                reranked_scores = agg_scores[rank_id]
                template['bm25_score'] = reranked_scores
                reranked_instances.append(template)
            instance['bm25_results'] = reranked_instances
            temp.append(instance)
            self.run_subeval(i, temp)
        self.write_jsonl_file(self.args.output_path, temp)
        ndcg_k, scores = run_rerank_eval(self.args.output_path, k=100)
        return ndcg_k

def run_reranker(args):
    #run = wandb.init(project="q2q_pangu", config=vars(args), entity='soyoung97',name=args.output_path)
    module = Pangu(args)
    module.run_baseline()
    flops = module.flops
    num_forward = module.num_forward
    ndcg_k = run_rerank_eval(args.output_path)
    return scores, flops, num_forward

def get_ndcg_from_string(string):
    ndcg_at_10 = float(string.strip().split()[8].replace(',',''))
    return ndcg_at_10

# @slack_sender(webhook_url=get_webhook_url(), channel=get_channel())
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="castorini/duot5-base-msmarco", type=str)
    parser.add_argument('--topk', default=100, type=int) #or 1000
    parser.add_argument('--beam_size', default=-1, type=int)
    #parser.add_argument('--gold_dir', default='../dataset/beir_bm25/final/', type=str)
    #parser.add_argument('--gold_dir', default='../dataset/beir_bm25/final_fromindex/', type=str)
    parser.add_argument('--gold_dir', default='./castorini/monot5-base-msmarco-10k/official/', type=str)
    parser.add_argument('--dataname', required=True, type=str)
    parser.add_argument('--outname', required=True, type=str)
    parser.add_argument('--max_gen_length', default=256, type=int)
    parser.add_argument('--bsize', default=128, type=int)
    parser.add_argument('--padding', default='max_length', type=str) # longest is recommended
    parser.add_argument('--listwise_k', default=20, type=int)
    parser.add_argument('--rerank_topk', default=10, type=int)
    parser.add_argument('--out_k', default=1, type=int)
    parser.add_argument('--no_dummy', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--encoder_output_k', default=-1, type=int)
    parser.add_argument('--sub_mode', default='', type=str)
    parser.add_argument('--measure_flops', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--g80', action='store_true')
    parser.add_argument('--shuffle_local', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test_path', default='', type=str)
    parser.add_argument('--output_path', default='', type=str)
    args = parser.parse_args()
    res = {}
    random.seed(args.seed)
    if args.shuffle:
        print(f"Shuffle input data with seed : {args.seed}")
    # args.test_path = f"{args.gold_dir}/{args.dataname}_output.jsonl"
    # args.output_path = f"./baseline_model_outputs/{args.sub_mode}/{args.model_path}/{args.outname}/{args.dataname}_output.jsonl"

    data2len = {'msmarco': 256, 'dl19': 256, 'dl20': 256, 'trec-covid': 512, 'nfcorpus': 512,
            'bioasq': 512,
            'nq': 256, 'hotpotqa': 256, 'fiqa': 512, 'signal': 256, 'news': 1024, 'robust04': 1024,
            'arguana': 1024, 'touche': 1024, 'cqadupstack': 512, 'quora': 256, 'dbpedia-entity': 256,
            'scidocs': 512, 'fever': 256, 'climate-fever': 256, 'msmarco_top1000': 256,
            'scifact': 512}
    max_length = data2len.get(args.dataname)
    if not 'nobsize' in args.sub_mode:
        args.bsize = {256:384*5, 512:160*5, 1024: 48*5, 1280: 24*5}[max_length]
        if args.g80:
            args.bsize *= 2
    if max_length == None:
        print(f"No mapping for dataname {args.dataname}!!!!!")
        raise Exception
    else:
        print(f"Max length: {max_length} for dataname: {args.dataname}")
    pprint(args)
    args.max_length = max_length
    #if args.measure_flops:
    #    args.max_length=4096
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    start_time = time.time()
    ndcg_k, flops, num_forwards = run_reranker(args)
    res['flops'] = flops
    res['num_forwards'] = num_forwards
    # res[args.output_path] = scores
    # ndcg_at_10 = get_ndcg_from_string(scores)
    res['ndcg@10'] = ndcg_k
    end_time = time.time()
    data_name = './duot5_eval_result.txt'
    log_str = ""
    log_str += f"FILE : {args.test_path}\n"
    log_str += f"NDCG@10 : {ndcg_k}\n"
    log_str += "==============================\n"
    with open(data_name, 'a', encoding='utf-8') as f:
        f.write(log_str)

    res['time_duration'] = end_time - start_time
    if args.shuffle or args.shuffle_local:
        res['shuffle_true_and_seed_is'] = args.seed
    print(res)
    return res



if __name__ == '__main__':
    main()
