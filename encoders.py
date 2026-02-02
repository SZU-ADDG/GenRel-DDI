import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from local_bert import SmilesTokenizer, load_yaml, BertConfig, BERT


def _set_smiles_special(tok):
    tok.pad_token = "[PAD]"
    tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
    tok.unk_token = "[UNK]"
    tok.unk_token_id = tok.convert_tokens_to_ids("[UNK]")
    tok.cls_token = "[CLS]"
    tok.cls_token_id = tok.convert_tokens_to_ids("[CLS]")
    tok.sep_token = "[SEP]"
    tok.sep_token_id = tok.convert_tokens_to_ids("[SEP]")
    tok.mask_token = "[MASK]"
    tok.mask_token_id = tok.convert_tokens_to_ids("[MASK]")
    return tok


class LocalBertAsHF(nn.Module):
    """Wrap the self-pretrained BERT so that Project-2's model code can stay unchanged.

    It mimics HuggingFace API:
      - exposes `.config.hidden_size`
      - forward returns an object with `.last_hidden_state`
    """

    def __init__(self, base: str):
        super().__init__()

        tok = SmilesTokenizer(os.path.join(base, "vocab.txt"))
        self.tokenizer = _set_smiles_special(tok)

        enc_cfg = load_yaml(os.path.join(base, "encoder.yaml"))
        ckpt = torch.load(os.path.join(base, "checkpoint.pt"), map_location="cpu", weights_only=False)

        cfg = BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_layer=int(enc_cfg["n_layer"]),
            n_head=int(enc_cfg["n_head"]),
            n_embd=int(enc_cfg["n_embd"]),
        )
        self.model = BERT(cfg)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

        self.config = SimpleNamespace(hidden_size=int(enc_cfg["n_embd"]))

        # match Project-1 behavior: keep pooler/mlm head frozen
        for p in self.model.pooler.parameters():
            p.requires_grad = False
        for p in self.model.mlm_head.parameters():
            p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        seq, _, _, _ = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return SimpleNamespace(last_hidden_state=seq)


def build_tokenizer(spec: dict):
    kind = spec["kind"]
    path = spec["path"]

    if kind == "hf":
        return AutoTokenizer.from_pretrained(path, trust_remote_code=spec.get("trust_remote_code", False))
    if kind == "local_bert":
        tok = SmilesTokenizer(os.path.join(path, "vocab.txt"))
        return _set_smiles_special(tok)
    raise ValueError(kind)


def build_encoder(spec: dict):
    kind = spec["kind"]
    path = spec["path"]

    if kind == "hf":
        return AutoModel.from_pretrained(path, trust_remote_code=spec.get("trust_remote_code", False))
    if kind == "local_bert":
        return LocalBertAsHF(path)
    raise ValueError(kind)


def pick_spec(specs: list[dict], slot: str) -> dict:
    for s in specs:
        if s.get("slot") == slot:
            return s
    raise ValueError(f"missing encoder spec for slot={slot}")
