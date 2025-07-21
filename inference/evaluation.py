from beir_eval import run_direct_rerank_eval

ndcg_k, scores = run_direct_rerank_eval('outputs/duot5-dl20.jsonl', k=100)