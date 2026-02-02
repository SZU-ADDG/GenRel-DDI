import math
import logging
import collections
import os
import re
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer
import yaml

logger = logging.getLogger(__name__)

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


class BertConfig:
    def __init__(self, vocab_size, n_embd, n_layer, n_head, kv_heads=4,
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, max_len=1024):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.kv_heads = kv_heads
        self.max_len = max_len


class BidirectionalSelfFlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv_split = (config.n_embd, config.n_embd, config.n_embd)
        self.to_qkv = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.combine_heads = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, attention_mask, current_idx=None, kv_cache=False):
        B, T, C = x.size()
        head_dim = self.head_dim

        q, k, v = self.to_qkv(x).split(self.qkv_split, dim=-1)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        dropout_p = self.attn_drop.p if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=False,
        )

        y = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.combine_heads(y))
        return y, None


class Block(nn.Module):
    def __init__(self, config, _):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfFlashAttention(config)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attention_mask):
        y, attn = self.attn(self.ln1(x), attention_mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_len, config.n_embd)
        self.seg_emb = nn.Embedding(2, config.n_embd)

        self.ln = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids, token_type_ids):
        b, t = input_ids.size()

        token_embeddings = self.tok_emb(input_ids)
        position_ids = torch.arange(t, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.pos_emb(position_ids)
        segment_embeddings = self.seg_emb(token_type_ids)

        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.ln(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        x = hidden_states[:, 0]
        x = self.dense(x)
        x = self.activation(x)
        return x


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.n_embd)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.blocks = nn.Sequential(*[Block(config, _) for _ in range(config.n_layer)])
        self.pooler = BertPooler(config)
        self.mlm_head = BertLMPredictionHead(config, self.embeddings.tok_emb.weight)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

        if attention_mask is None:
            attention_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0

        x = self.embeddings(input_ids, token_type_ids)

        attn_maps = []
        for layer in self.blocks:
            x, attn = layer(x, attention_mask)
            attn_maps.append(attn)

        sequence_output = x
        pooled_output = self.pooler(sequence_output)

        loss = None
        if masked_lm_labels is not None:
            prediction_scores = self.mlm_head(sequence_output)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        return sequence_output, pooled_output, loss, attn_maps


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class SmilesTokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = "", **kwargs):
        super().__init__(vocab_file, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocab file at path '{vocab_file}'.")

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())


class BasicSmilesTokenizer(object):
    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        return [token for token in self.regex.findall(text)]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return {} if obj is None else obj
