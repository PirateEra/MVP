# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.file_utils import ModelOutput
from transformers import T5Tokenizer

@dataclass
class RankingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    ranking: Optional[torch.LongTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    passage_embed: Optional[torch.FloatTensor] = None


class PointNetwork(nn.Module):
    def __init__(self, input_dim=None, output_dim=1):
        super(PointNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class FiDT5(transformers.T5ForConditionalGeneration):
    # 이 부분 Parameter는 Train Code에 따라 달라질 예정.
    def __init__(self, config, pooling_type=None, n_passages=5, softmax_temp=1, n_special_tokens=0, local_weight=0.5, tokenizer=None):
        super().__init__(config)
        ### jeongwoo
        self.topk_attn_indices = []
        self.pooling_type = pooling_type
        self.n_passages = n_passages
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.softmax_temp = softmax_temp
        self.local_weight = local_weight
        self.n_special_tokens = n_special_tokens
        self.wrap_encoder(pooling_type=pooling_type, n_special_tokens=self.n_special_tokens)

        if 'weighted' in pooling_type:
            # MAKE self.d_model * self.n_special_tokens -> self.n_special_tokens Gating MLP
            self.gating_mlp = nn.Linear(self.n_special_tokens, 1)
        else:
            self.gating_mlp = None
        
        
        
        
    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.

    def orthogonal_loss(self, hidden_states, bsz):

        hidden_states = hidden_states.view(hidden_states.size(0), -1).view(bsz, self.n_special_tokens, -1)
        last_hidden_states_norm = F.normalize(hidden_states, p=2, dim=-1)
        dot_product = torch.bmm(last_hidden_states_norm, last_hidden_states_norm.transpose(1, 2)) # (bsz, n_special_tokens, n_special_tokens)
        
        # Ignore the diagonal
        I = torch.eye(self.n_special_tokens, device=hidden_states.device).unsqueeze(0)
        I = I.repeat(bsz, 1, 1)
        diff = dot_product - I

        loss = (diff ** 2).sum(dim=(1, 2)).mean() 

        return loss
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        label_dist: Optional[torch.FloatTensor] = None,
        **kwargs
    )->RankingOutput:
        r'''
        kwargs
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        '''

        
        # 1. resize input_ids and attention_mask and get encoder_outputs
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
                self.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        if kwargs.get('return_dict') is None:
            kwargs['return_dict'] = False
        
        # 2. get encoder_outputs if not provided
        if kwargs.get('encoder_outputs') is None:
            kwargs['output_attentions'] = False
            self.encoder.n_passages = self.n_passages

            encoder_kwargs = {
                "input_ids": input_ids, 
                "attention_mask": attention_mask,
                "output_attentions": False,
                "return_dict": False
            }
            encoder_outputs, attention_mask = self.encoder(**encoder_kwargs)
            kwargs["encoder_outputs"] = encoder_outputs
        # update encoder_outputs and attention_mask after pooling in EncoderWrapper if provided
        else:
            encoder_outputs, attention_mask = kwargs["encoder_outputs"]
            kwargs["encoder_outputs"] = encoder_outputs
        
        # 3. get decoder input embeds with encoder_outputs and labels
            # always start with pad token
            # (hsz) -> (bsz, k, hsz)
            # single_token_decoding -> Only generate one token in FiD list 
            # autoregressive decoding -> shift t
        bsz = input_ids.size(0)

        # Reshape encoder Outputs that can be used in the decoder
        # Special Tokens which are in the same position will be concatenated
        reshaped = encoder_outputs[0].view(bsz, self.n_passages, self.n_special_tokens, -1)
        transposed = reshaped.permute(0, 2, 1, 3)
        passage_embed = transposed.reshape(bsz * self.n_special_tokens, self.n_passages, -1)
        attention_mask = torch.ones(passage_embed.size(0), passage_embed.size(1), dtype=torch.long).to(attention_mask.device)

        if labels is None:
            pad_token_embed = self.shared.weight[self.pad_token_id]
            decoder_input_embed = pad_token_embed.view(1, 1, -1).expand(passage_embed.size(0), 1, -1)

            with torch.no_grad():
                decoder_outputs = self.decoder(
                    inputs_embeds=decoder_input_embed,
                    encoder_hidden_states=passage_embed,
                    encoder_attention_mask=attention_mask,
                    return_dict=kwargs['return_dict'],
                )
        
        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            # Decoder input ids shape : (bsz, 1).
            # Expand it to (bsz * self.n_special_tokens, 1).
            decoder_input_ids = decoder_input_ids.unsqueeze(1).expand(-1, self.n_special_tokens, -1).reshape(-1, decoder_input_ids.size(-1)) 
            # if auto-regressive decoding, use except for the last token
            if labels.size(1) != 1:
                decoder_input_ids = decoder_input_ids[:, :-1]
                decoder_attention_mask = decoder_attention_mask[:, :-1]

            # Call decoder based on decoder input type (nl token embeddin or ranking vector)
            decoder_input_embed = self.shared(decoder_input_ids)
            # passage_embed = encoder_outputs[0]
            # lookahead pooling

            decoder_outputs = self.decoder(
                inputs_embeds=decoder_input_embed,
                encoder_hidden_states=passage_embed,
                encoder_attention_mask=attention_mask,
                return_dict=kwargs['return_dict']
            )
        
        # generate when pad token is the only token
        last_hidden_state = decoder_outputs[0].to(decoder_outputs[0].device)                
        
        # 4. get logits by dot product between 1) last hidden state and 2) encoder outputs(ranking vectors)

        # Calculate COSINE SIMILARITY BETWEEN EACH LAST HIDDEN STATES (THERE ARE 4 LAST HIDDEN STATES)
        
        # lhs_norm = F.normalize(last_hidden_state, p=2, dim=-1)


        logits = torch.einsum('bsh,bph->bsp', last_hidden_state, passage_embed)
        logits = logits.view(bsz, self.n_special_tokens, self.n_passages)
        # logits_temp = logits
        logits = logits.mean(dim=1)


        # 이부분은 Training Code가 다 만들어진 다음에 수정될 예정.
        # Step 1 에서는 ListNet을 사용하고, Step 2에서는 RankNet을 사용함.
        # 이 두 함수를 Training Code 완성되고 난 다음에 가져와야함.
        loss = 0
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            if label_dist is not None:
                logits /= self.softmax_temp
                logits = torch.nn.functional.softmax(logits, dim=-1)
                list_loss = loss_fct(logits.view(-1, logits.size(-1)), label_dist.view(-1, label_dist.size(-1)))
                orthogonal_loss = self.orthogonal_loss(last_hidden_state, bsz)

            loss = list_loss + self.local_weight * orthogonal_loss
        else:

            loss = 0
            
        return RankingOutput(
            loss=loss,
            logits=logits,
            # ranking = logits_temp,
            # passage_embed=passage_embed,
            # last_hidden_state=last_hidden_state,
            # ranking=last_hidden_state,
        )
    

    # Calculate the topk indices based on the logits
    # Only Generate 1 token
    def generate_by_single_logit(self, input_ids=None, attention_mask=None, **kwargs):

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if outputs.logits.dim() == 3:
            logits = outputs.logits.squeeze(1).to(input_ids.device)
        else:
            logits = outputs.logits.to(input_ids.device)

        topk_indices = logits.topk(logits.size(-1), dim=1).indices.tolist()

        return topk_indices
    
    def wrap_encoder(self, pooling_type, use_checkpoint=False, n_special_tokens=0):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, pooling_type=pooling_type, use_checkpoint=use_checkpoint, n_special_tokens=n_special_tokens)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder(self.pooling_type, n_special_tokens=self.n_special_tokens)
            
    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, pooling_type=None, use_checkpoint=False, n_special_tokens=0):
        super().__init__()
        self.main_input_name = encoder.main_input_name
        self.encoder = encoder
        self.pooling_type = pooling_type
        self.n_special_tokens = n_special_tokens
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def set_input_embeddings(self, new_embeddings):
        """
        This method allows the embedding layer to be updated
        with the new resized embedding.
        """
        self.encoder.set_input_embeddings(new_embeddings)
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):

        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)

        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask, **kwargs)

        # Extract the last hidden state of the special tokens
        last_hidden_state = outputs[0][:, :self.n_special_tokens, :]
        last_hidden_state = last_hidden_state.contiguous().view(bsz, self.n_passages*self.n_special_tokens, -1).to(outputs[0].device)

        attention_mask = torch.ones(bsz, last_hidden_state.size(1), dtype=torch.long).to(attention_mask.device)
        
        if kwargs.get('return_dict'):
            outputs.last_hidden_state = last_hidden_state
        else:
            outputs = (last_hidden_state,) + outputs[1:]
        return outputs, attention_mask
        
class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output
