import os
import torch


class Config:
    """Project-2 config (Project-1 style).

    - tasks: dataset definitions (train/val/test CSVs under data_dir)
    - encoder_specs: per-slot encoder setup (slot in {"bert", "mol"}, with kind and path)
    """

    def __init__(self):
        root = os.path.dirname(os.path.abspath(__file__))

        # =====================
        # Datasets / tasks
        # =====================
        self.tasks = [
            {
                "name": "UnseenDDIs",
                "data_dir": "./data/UnseenDDIs",
                "train_csv": "train.csv",
                "val_csvs": {
                    "val": "val.csv",
                },
                "test_csvs": {
                    "test": "test.csv",
                },
                "num_labels": 4,
            },
            {
                "name": "Unseendrugs_onedrug",
                "data_dir": "./data/Unseendrugs",
                "train_csv": "train.csv",
                "val_csvs": {
                    "val": "val_dataset_unseen_onedrug.csv",
                },
                "test_csvs": {
                    "test": "val_dataset_unseen_onedrug.csv",
                },
                "num_labels": 4,
            },
            {
                "name": "Unseendrugs_twodrugs",
                "data_dir": "./data/Unseendrugs",
                "train_csv": "train.csv",
                "val_csvs": {
                    "val": "val_dataset_unseen_twodrugs.csv",
                },
                "test_csvs": {
                    "test": "val_dataset_unseen_twodrugs.csv",
                },
                "num_labels": 4,
            }
        ]


        # kind:
        # - "hf": HuggingFace AutoModel/AutoTokenizer
        # - "local_bert": self-pretrained BERT (vocab.txt + encoder.yaml + checkpoint.pt)
        self.encoder_specs = [
            {
                "slot": "bert",
                "name": "chemberta",
                "kind": "hf",
                "path": "./pretrained/chemberta-77",
                "trust_remote_code": False,
                "freeze": True,
                "max_len": 128,
            },
            {
                "slot": "mol",
                "name": "molformer",
                "kind": "hf",
                "path": "./pretrained/MolFormer_10pct",
                "trust_remote_code": True,
                "freeze": False,
                "max_len": 128,
            },
        ]

        self.seeds = [42,43,44,45,46]
        self.num_workers = 4

        self.batch_size = 256
        self.epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5

        self.dropout = 0.1
        self.hidden_dim = 256
        self.intra_attn_direction = "bidirectional"
        self.inter_attn_direction = "bidirectional"

        self.num_labels = 4

        self.model_save_dir = os.path.join(root, "checkpoints")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
