
import torch
import torch.nn as nn

from encoders import build_encoder, pick_spec

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, q, kv, kv_mask=None):
        q = self.q_proj(q)
        kv = self.kv_proj(kv)
        key_padding_mask = (kv_mask == 0) if kv_mask is not None else None
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        x = self.norm1(q + out)
        return self.norm2(x + self.ffn(x))

class FusionModule(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim, direction):
        super().__init__()
        self.direction = direction
        if direction in ["left_to_right", "bidirectional"]:
            self.a12 = CrossAttentionBlock(dim1, dim2, hidden_dim)
        if direction in ["right_to_left", "bidirectional"]:
            self.a21 = CrossAttentionBlock(dim2, dim1, hidden_dim)

    def forward(self, x1, x2, m1=None, m2=None):
        outs = []
        if self.direction in ["left_to_right", "bidirectional"]:
            outs.append(self.a12(x1, x2, m2))
        if self.direction in ["right_to_left", "bidirectional"]:
            outs.append(self.a21(x2, x1, m1))
        return outs

class DDIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_spec = pick_spec(config.encoder_specs, "bert")
        mol_spec = pick_spec(config.encoder_specs, "mol")
        self.chemberta = build_encoder(bert_spec)
        self.molformer = build_encoder(mol_spec)
        freeze_bert = bool(bert_spec.get("freeze", False))
        freeze_mol = bool(mol_spec.get("freeze", False))

        if freeze_bert:
            for p in self.chemberta.parameters():
                p.requires_grad = False
        if freeze_mol:
            for p in self.molformer.parameters():
                p.requires_grad = False

        bert_dim = int(self.chemberta.config.hidden_size)
        mol_dim = int(self.molformer.config.hidden_size)

        self.intra = FusionModule(bert_dim, mol_dim, config.hidden_dim, config.intra_attn_direction)
        intra_dim = config.hidden_dim * 2 if config.intra_attn_direction == "bidirectional" else config.hidden_dim

        self.inter = FusionModule(intra_dim, intra_dim, config.hidden_dim, config.inter_attn_direction)
        final_dim = config.hidden_dim * 2 if config.inter_attn_direction == "bidirectional" else config.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels)
        )

    def encode(self, ids_b, mask_b, ids_m, mask_m):
        b = self.chemberta(ids_b, attention_mask=mask_b).last_hidden_state
        m = self.molformer(ids_m, attention_mask=mask_m).last_hidden_state
        outs = self.intra(b, m, mask_b, mask_m)
        pooled = [o.mean(1) for o in outs]
        return torch.cat(pooled, -1) if len(pooled) > 1 else pooled[0]

    def forward(self, batch):
        a = self.encode(batch["a_bert_ids"], batch["a_bert_mask"],
                        batch["a_mol_ids"], batch["a_mol_mask"])
        b = self.encode(batch["b_bert_ids"], batch["b_bert_mask"],
                        batch["b_mol_ids"], batch["b_mol_mask"])

        outs = self.inter(a.unsqueeze(1), b.unsqueeze(1))
        feats = [o.squeeze(1) for o in outs]
        feat = torch.cat(feats, -1) if len(feats) > 1 else feats[0]
        return self.classifier(feat)
