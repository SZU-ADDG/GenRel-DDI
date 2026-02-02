
import pandas as pd
import torch
from torch.utils.data import Dataset

class DDIDataset(Dataset):
    def __init__(
        self,
        csv_path,
        bert_tokenizer,
        mol_tokenizer,
        max_len_bert=128,
        max_len_mol=128,
    ):
        self.data = pd.read_csv(csv_path)
        self.bert_tokenizer = bert_tokenizer
        self.mol_tokenizer = mol_tokenizer
        self.max_len_bert = int(max_len_bert)
        self.max_len_mol = int(max_len_mol)

        self.drug_a = self.data.iloc[:, 0].astype(str).tolist()
        self.drug_b = self.data.iloc[:, 1].astype(str).tolist()
        self.labels = self.data.iloc[:, 2].astype(int).tolist()

    def __len__(self):
        return len(self.data)

    def tokenize(self, smiles, tokenizer, max_len):
        return tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        a = self.tokenize(self.drug_a[idx], self.bert_tokenizer, self.max_len_bert)
        a_m = self.tokenize(self.drug_a[idx], self.mol_tokenizer, self.max_len_mol)
        b = self.tokenize(self.drug_b[idx], self.bert_tokenizer, self.max_len_bert)
        b_m = self.tokenize(self.drug_b[idx], self.mol_tokenizer, self.max_len_mol)

        return {
            "a_bert_ids": a["input_ids"].squeeze(0),
            "a_bert_mask": a["attention_mask"].squeeze(0),
            "a_mol_ids": a_m["input_ids"].squeeze(0),
            "a_mol_mask": a_m["attention_mask"].squeeze(0),
            "b_bert_ids": b["input_ids"].squeeze(0),
            "b_bert_mask": b["attention_mask"].squeeze(0),
            "b_mol_ids": b_m["input_ids"].squeeze(0),
            "b_mol_mask": b_m["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
