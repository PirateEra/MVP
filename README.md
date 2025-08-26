# MVP


This repository contains the official implementation for our EMNLP 2025 paper:

> **Multi-view-guided Passage Reranking with Large Language Models**  
> Jeongwoo Na*, Jun Kwon*, Eunseong Choi, Jongwuk Lee (* : equal contribution)  
> *Accepted to EMNLP 2025*

## Overview
Recent advances in large language models (LLMs) have shown impressive performance in passage reranking tasks. Despite their success, LLM-based methods still face challenges in efficiency and sensitivity to external biases. (i) Existing models rely mostly on autoregressive generation and sliding window strategies to rank passages, which incurs heavy computational overhead as the number of passages increases. (ii) External biases, such as positional or semantic bias, hinder the model’s ability to accurately represent passages and the input-order sensitivity. To address these limitations, we introduce a novel passage reranking model, called Multi-View-guided Passage Reranking (MVP). MVP is a non-generative LLM-based reranking method that encodes query–passage information into diverse view embeddings without being influenced by external biases. For each view, it combines query-aware passage embeddings to produce a distinct anchor vector, used to directly compute relevance scores in a single decoding step. Besides, it employs an orthogonal loss to make the views more distinctive. Extensive experiments demonstrate that MVP, with just 220M parameters, matches the performance of much larger 7B-scale fine-tuned models while achieving a 100× reduction in inference latency. Notably, the 3B-parameter variant of MVP achieves state-of-the-art performance on both in-domain and out-of-domain benchmarks.   


### Running Environment
```
conda env create -f mvp.yaml
```

### How to Use
TBU