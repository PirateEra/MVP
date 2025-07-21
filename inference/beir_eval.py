import argparse
import jsonlines
from collections import defaultdict
import json
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import pandas as pd
import logging
import pathlib, os
import glob

def write_json_file(file_path, res):
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Wrote json file to: {file_path}!")

def read_jsonl_file(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def cleanup_id(id_text):
    if type(id_text) == int:
        return str(id_text)
    elif '_' in id_text:
        return id_text.split('_')[-1]
    else:
        raise NotImplementedError

def convert_to_result_format(data):
    output = {}
    for instance in data: # to do: inspect more
        qid = str(instance['qid'])
        input_q = instance['q_text']
        psg_ids = [str(x['pid']) for x in instance['bm25_results']]
        scores = [x['bm25_score'] for x in instance['bm25_results']]
        output[qid] = {}
        for psgid, score in zip(psg_ids, scores):
            output[qid][psgid] = float(score)
    return output

def check_dup(data, key='q_text'):
    new = []
    qs = set()
    for d in data:
        if d['q_text'] not in qs:
            new.append(d)
            qs.add(d['q_text'])
    if len(data) != len(new):
        print(f"Original data len {len(data)}, removed dup to {len(new)}")
    return new
    #return new
def check_100(data):
    new = []
    for instance in data:
        if len(instance['bm25_results']) > 100:
            print(f"Shortening to 100 from {len(instance['bm25_results'])}")
            instance['bm25_results'] = instance['bm25_results'][:100]
        new.append(instance)
    return new

def setup():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
    #### /print debug information to stdout

def format_res_for_print(k, acc, ndcg, _map, recall, pre, mrr):
    out = ''
    k_values = [1,5,10,20,50,100]
    k_values = [x for x in k_values if x <= k]
    
    acc_str = "Accuarcy@" + "/".join([str(x) for x in k_values]) + ": " + ', '.join([str(acc[f"Accuracy@{x}"]) for x in k_values])
    ndcg_str = "NDCG@" + "/".join([str(x) for x in k_values]) + ": " + ', '.join([str(ndcg[f"NDCG@{x}"]) for x in k_values])
    mrr_str = "MRR@" + "/".join([str(x) for x in k_values]) + ": " + ', '.join([str(mrr[f"MRR@{x}"]) for x in k_values])
    recall_str = "Recall@" + "/".join([str(x) for x in k_values]) + ": " + ', '.join([str(recall[f"Recall@{x}"]) for x in k_values])
    pre_str = "Precision@" + "/".join([str(x) for x in k_values]) + ": " + ', '.join([str(pre[f"P@{x}"]) for x in k_values])
    map_str = "MAP@" + "/".join([str(x) for x in k_values]) + ": " + ', '.join([str(_map[f"MAP@{x}"]) for x in k_values])
    
    out = '\n' + acc_str + '\n' + ndcg_str + '\n' + mrr_str + '\n' + recall_str + '\n' + pre_str + '\n' + map_str
    
    return out

def make_dummy_results(corpus, queries):
    query_keys = list(queries.keys())
    corpus_keys = list(corpus.keys())
    out = {}
    for key in query_keys:
        out[key] = {ck: 0.5 for ck in corpus_keys}
    return out

def remove_nan(results):
    new_res = {}
    for query_key in results.keys():
        out = {}
        for i, corpus_key in enumerate(results[query_key].keys()):
            out[corpus_key] = 100 - i
        new_res[query_key] = out
    return new_res

def do_evaluation(queries, qrels, corpus, results=None, k=10, mode='ours'):
    k_values = [1,5,10,20,50,100]
    k_values = [x for x in k_values if x <= k]
    retriever = EvaluateRetrieval()
    if mode != 'ours':
        from beir.reranking.models import CrossEncoder, MonoT5
        from beir.reranking import Rerank
        if 'monot5' in mode:
            cross_encoder_model = MonoT5(mode, token_false='▁false', token_true='▁true')
        else:
            cross_encoder_model = CrossEncoder(mode)
        orig_results = results
        print(f"Loading cross-encoder model from: {cross_encoder_model.model.config._name_or_path}")
        reranker = Rerank(cross_encoder_model, batch_size=700)
        results = reranker.rerank(corpus, queries, results, top_k=100)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
    mrr = retriever.evaluate_custom(qrels, results, k_values, metric='mrr')
    hits = retriever.evaluate_custom(qrels, results, k_values, metric='top_k_accuracy')
    
    ndcg_k = None
    if k <= 10:
        ndcg_k = ndcg[f'NDCG@{str(k)}']
    else:
        ndcg_k = ndcg[f'NDCG@10']
    out_string = format_res_for_print(k, hits, ndcg, _map, recall, precision, mrr)
    return ndcg_k, out_string

def make_corpus(data):
    res = {}
    for line in data:
        ctxs = line['bm25_results']
        for subline in ctxs:
            res[str(subline['pid'])] = {'text': subline['text'], 'title': 'none'}
    return res

def run_rerank_eval(data_path, mode='ours', combined=False):
    if combined:
        data = data_path
    else:
        data = read_jsonl_file(data_path)
    data = check_dup(data)
    data = check_100(data)
    results = convert_to_result_format(data)
    corpus = make_corpus(data)
    queries = {}
    qrels = {}
    ## making queries and qrels
    for line in data:
        id_text = str(line['qid'])
        queries[id_text] = line['q_text']
        qrels[id_text] = line['qrels']
    ndcg_k, out_string = do_evaluation(queries, qrels, corpus, results=results, mode=mode)
    
    return ndcg_k, out_string

def run_direct_rerank_eval(data_path, k=10, mode='ours', combined=False):
    if combined:
        data = data_path
    else:
        data = read_jsonl_file(data_path)
    data = check_dup(data)
    data = check_100(data)
    results = convert_to_result_format(data)
    corpus = make_corpus(data)
    queries = {}
    qrels = {}
    ## making queries and qrels
    for line in data:
        id_text = str(line['qid'])
        queries[id_text] = line['q_text']
        qrels[id_text] = line['qrels']
    ndcg_k, out_string = do_evaluation(queries, qrels, corpus, results=results, k=k, mode=mode)
    print("====================================")
    if not combined:
        print(f"For {data_path}")
    print(f"Evaluation results :")
    print(out_string)
    if k <= 10:
        print(f"NDCG@{k}: ")
    else:
        print(f"NDCG@10: ")
    print(ndcg_k)
    print("====================================")
    
    return ndcg_k, out_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../../dataset/beir_bm25/fiqa_test.json')
    parser.add_argument('--mode', default='ours', type=str)
    parser.add_argument('--combine', action='store_true')
    args = parser.parse_args()
    setup()
    if args.combine:
        paths = glob.glob(f"{args.path}/*.jsonl")
        print(f'Combine = True, paths: {paths}')
        full_data = []
        for path in paths:
            data = read_jsonl_file(path)
            full_data += data
        print(f"Combined full len: {len(full_data)}")
        run_rerank_eval(full_data, mode=args.mode, combined=True)
    else:
        run_rerank_eval(args.path, mode=args.mode, combined=False)
    # making corpus out of data
