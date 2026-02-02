# GenRel-DDI
Rethinking Drug–Drug Interaction Modeling as Generalizable Relation Learning

## 1. Installation

Recommended:
- Python 3.10
- PyTorch built for your CUDA/driver stack

Install the minimal Python dependencies:
```bash
pip install -r requirements.txt
```

Notes:
- `torch` wheels are CUDA-specific; install it the same way you install PyTorch on your cluster.
- The remaining packages are pure Python and can be installed via pip.

---

## 2. Data format

The code expects CSV files with three columns:

1. drug A (SMILES or any string accepted by the tokenizer)
2. drug B
3. label

---

## 3. Training

### 3.1 Default training (run all tasks × all seeds)

```bash
python main.py --mode train
```

The training loop iterates:
- all entries in `config.py -> tasks`
- all entries in `config.py -> seeds`

### 3.2 Train a single task

`--task` is the index in `config.py -> tasks` (starting from 0):
```bash
python main.py --mode train --task 0
```

### 3.3 Train a single seed

```bash
python main.py --mode train --task 0 --seed 42
```

---

## 4. Evaluation

Evaluation is performed for a single task:

```bash
python main.py --mode eval --task 0 --ckpt /path/to/checkpoint.pth
```

Behavior:
- uses `task["test_csvs"]` when provided
- if `test_csvs` is empty, it falls back to `val_csvs`

---

## 5. Configuration (config.py)

### 5.1 `tasks`: dataset/task definitions

`tasks` is a list. Each item describes one dataset setup.

Common fields:
- `name`: used in logs and output directories
- `data_dir`: directory containing the CSV files
- `train_csv`: training CSV file name (relative to `data_dir`)
- `val_csvs`: validation CSV mapping, e.g. `{"val": "val.csv"}`
- `test_csvs`: test CSV mapping, e.g. `{"test": "test.csv"}`
- `num_labels`: number of classes (e.g. 4)

Example:
```python
self.tasks = [
    {
        "name": "ddinter_s1",
        "data_dir": "/path/to/data",
        "train_csv": "train.csv",
        "val_csvs": {"val": "val.csv"},
        "test_csvs": {"test": "test.csv"},
        "num_labels": 4,
    },
]
```

### 5.2 `encoder_specs`: two encoders (bert / mol)

`encoder_specs` must contain exactly two entries:

- `slot="bert"`: drives the `Anchor Role` inputs
- `slot="mol"`: drives the `Adapter Role` inputs

Common fields:
- `slot`: `"bert"` or `"mol"`
- `kind`:
  - `"hf"`: load via `transformers` AutoModel/AutoTokenizer
  - `"local_bert"`: load your self-pretrained model from a local directory
- `path`: model name or local directory path
- `freeze`: whether to freeze this encoder during training
- `max_len`: max token length for this encoder input

Example:
```python
self.encoder_specs = [
    {"slot": "bert", "kind": "local_bert", "path": "/path/to/1B",   "freeze": True,  "max_len": 128},
    {"slot": "mol",  "kind": "local_bert", "path": "/path/to/200M", "freeze": False, "max_len": 128},
]
```

#### `local_bert` directory requirements

The directory specified by `path` should contain:
- `vocab.txt`
- `encoder.yaml`
- `checkpoint.pt` (expected to contain `model_state_dict`)
