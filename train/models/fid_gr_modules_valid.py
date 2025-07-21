import os
import sys
import json
import copy
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from itertools import chain
import utils.format as format_utils
import utils.io_modules as io_utils
import jsonlines
import pandas as pd
from torch.utils.data import Dataset
import json
from models.shared_modules import SharedDataset, SharedModel
from pprint import pprint
from torchmetrics import BLEUScore
from pathlib import Path
from itertools import chain
from transformers import T5ForConditionalGeneration
import math
import time
import random
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval


import sys
import pdb
class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class FiDGRDatasetForTest(SharedDataset):
    def __init__(self, tokenizer, split, args):
        super().__init__(tokenizer, split, args)
        self.debugi = 0
        self.split = split
        self.dataset = self.format_msmarco_data(self.dataset, split)
        self.len = len(self.dataset)
        self.dup = []

    def format_msmarco_data(self, dataset, split):
        res = []
        # data = [x for x in dataset if len(x['ret']) == self.args.listwise_k]
        data = dataset
        # print(f"Orig: {len(dataset)}, After: {len(data)}")
        del self.dataset
        return data

    def __len__(self):
        return self.len
    
    def print_io(self, idx, input_, output_):
        print_str = "=" * 80
        print_str += f"\n({idx}) {self.split}\ninput: {input_}\n"
        print_str += f"\noutput: {output_}\n"
        print_str += "=" * 80
        print(print_str)

    def convert_listwise_to_features(self, idx): ## 여기다!!
        # import IPython;
        # IPython.embed()
        # exit
        raw = self.dataset[idx]
        shuffled_psgs = raw['bm25_results']

        if self.args.use_special_tokens:
            if self.args.n_special_tokens > 1:
                special_tokens = [f'<extra_id_{i}>' for i in range(0, self.args.n_special_tokens)]
                special_string = ''.join(special_tokens)
                input_texts = [special_string + f" | Query: {raw['q_text']} | Context: {x['text']}" for i, x in enumerate(shuffled_psgs)]
            else:
                special_string = f'<extra_id_{self.args.extra_id}>'
                input_texts = [f"{special_string} | Query: {raw['q_text']} | Context: {x['text']}" for i, x in enumerate(shuffled_psgs)]
        else:
            input_texts = [f"Query: {raw['q_text']} | Context: {x['text']}" for i, x in enumerate(shuffled_psgs)]

        source = self.tokenizer(input_texts, padding='max_length', max_length=self.args.max_input_length, truncation=True, return_tensors='pt')
        # Original PID
        pid = [x['pid'] for x in shuffled_psgs]
        qid = raw['qid']
        qrels = raw['qrels']

        return source, pid, qid, qrels

    def __getitem__(self, idx):
        if 'listwise' in self.args.sub_mode:
            source, pid, qid, qrels = self.convert_listwise_to_features(idx)
        elif 'split' in self.args.sub_mode:
            source, target = self.convert_split_to_features(idx)
        else:
            source, target = self.convert_to_features(idx)
        #print(f"Source shape: {source['input_ids'].shape}")
        res = {
            'idx': idx,
            "source_ids": source['input_ids'],
            # "target_ids": target['input_ids'],
            "source_mask": source['attention_mask'],
            # "target_mask": target['attention_mask'],
            # "target_dist": target_dist
            'pid': pid,
            'qid': qid,
            'qrels': qrels
        }
        return res
