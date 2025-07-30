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
import numpy as np
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

class FiDSortDataset(SharedDataset):
    def __init__(self, tokenizer, split, args):
        super().__init__(tokenizer, split, args)
        self.debugi = 0
        self.split = split
        self.dataset = self.format_msmarco_data(self.dataset, split)
        self.len = len(self.dataset)

    def format_msmarco_data(self, dataset, split):
        data = [x for x in dataset if len(x['bm25_results']) >= self.args.listwise_k]
        print(f"Orig: {len(dataset)}, After: {len(data)}")
        return data

    def __len__(self):
        return self.len

    def print_io(self, idx, input_, output_):
        print_str = "=" * 80
        print_str += f"\n({idx}) {self.split}\ninput: {input_}\n"
        print_str += f"\noutput: {output_}\n"
        print_str += "=" * 80
        print(print_str)

    def convert_listwise_to_features(self, idx):
        raw = self.dataset[idx]
        pos_idx = random.choice(range(self.args.listwise_k)) + 1
        if 'sorted' in self.args.sub_mode:
            neg_texts = raw['bm25_results'][:self.args.listwise_k - 1]
        else:
            neg_texts = random.sample(raw['bm25_results'], k=(self.args.listwise_k - 1))
        full_texts = neg_texts[:pos_idx - 1] + [raw['pos_psg']] + neg_texts[pos_idx - 1:]
        if len(full_texts) != self.args.listwise_k:
            print('Error!!')
            ForkedPdb().set_trace()
        input_texts = [f"Query: {raw['q_text']}, Index: {i+1}, Context: {x['text']}" for i, x in enumerate(full_texts)]
        source = self.tokenizer(input_texts, padding='max_length', max_length=self.args.max_input_length, truncation=True, return_tensors='pt')
        if 'top1' in self.args.sub_mode:
            converted = str(pos_idx)
        else:
            bm25_scores = [x['bm25_score'] for x in full_texts]
            sort_list = [str(x+1) for x in np.argsort(bm25_scores)]
            if 'pos_first' in self.args.sub_mode:
                sort_list.reverse()
            converted = ' '.join(sort_list)
        target = self.tokenize_t5(converted, max_length=self.args.max_output_length)
        if self.should_print() or self.args.debug:
            self.print_io(idx, '\n>>> '.join(input_texts), f"\nPos index: {pos_idx}, output: {converted}\n")
        return source, target

    def __getitem__(self, idx):
        source, target = self.convert_listwise_to_features(idx)
        res = {
            'idx': idx,
            "source_ids": source['input_ids'],
            "target_ids": target['input_ids'],
            "source_mask": source['attention_mask'],
             "target_mask": target['attention_mask'],
        }
        return res

class FiDSortModel(SharedModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if torch.cuda.current_device() == 0:
            self.print = True
        else:
            self.print = False
        # If in testing mode, load ckpt for inference
        if not self.args.do_train:
            self.test_input_list = []
            self.test_pred_list = []
            self.test_score_list = []
        self.gen_time = 0
        self.text_len = 0
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.idx2tokid = self.tokenizer.encode(' '.join([str(x) for x in range(1, self.args.listwise_k + 1)]))[:-1]

    def get_dataset(self, split):
        dataset = FiDSortDataset(
            tokenizer=self.tokenizer, split=split, args=self.args
        )
        return dataset

    def forward(self, input_ids, attention_mask, lm_labels, decoder_attention_mask):
        #ForkedPdb().set_trace()
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _loss(self, batch):
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=self.encode_label_ids({'input_ids': batch['target_ids']})['input_ids'],
            decoder_attention_mask=batch["target_mask"],
        )
        if 'listwise' in self.args.sub_mode:
            lm_logits = outputs[1] # [bsize, max_len, vocab_size]
            lm_logits = lm_logits[:, 0, :]
            target_logits = lm_logits[:, self.idx2tokid]
            target_labels = batch['target_ids'][:, 0]
            target_labels_converted = torch.tensor([self.idx2tokid.index(x) for x in target_labels]).to('cuda')
            listwise_loss = self.criterion(target_logits, target_labels_converted)
            #if listwise_loss.item() > outputs[0].item():
            #    import pdb; pdb.set_trace()
            return listwise_loss
            #aself.idx2tokid
            #ForkedPdb().set_trace()
        loss = outputs[0]
        #loss = loss + listwise_loss
        return loss

    def training_step(self, batch, batch_idx):
        self.log(
                'global_step',
                self.global_step,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True)
        loss = self._loss(batch)
        self.log(
            "train loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def _val_step(self, batch, return_elem=False):
        loss = self._loss(batch)

        self.log(
                'val_loss',
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True)
        return loss.item()


    def validation_step(self, batch, batch_idx):
        return self._val_step(batch)

    def _gather_object(self, obj):
        if self.print:
            print(f'## Gathering list from {self.args.n_gpu} process!')
        gathered = [None for _ in range(self.args.n_gpu)]
        dist.all_gather_object(gathered, obj)
        return gathered
