import os
import argparse

import torch
from torch.utils.data import DataLoader

from config import Config
from encoders import build_tokenizer, pick_spec
from dataset import DDIDataset
from model import DDIModel
from train_eval import train_epoch, evaluate
from utils import seed_everything


def build_loaders(task, bert_tok, mol_tok, bert_len, mol_len, cfg):
    data_dir = task["data_dir"]

    train_ds = DDIDataset(
        os.path.join(data_dir, task["train_csv"]),
        bert_tok,
        mol_tok,
        max_len_bert=bert_len,
        max_len_mol=mol_len,
    )
    val_ds = {
        k: DDIDataset(
            os.path.join(data_dir, v),
            bert_tok,
            mol_tok,
            max_len_bert=bert_len,
            max_len_mol=mol_len,
        )
        for k, v in task.get("val_csvs", {}).items()
    }
    test_ds = {
        k: DDIDataset(
            os.path.join(data_dir, v),
            bert_tok,
            mol_tok,
            max_len_bert=bert_len,
            max_len_mol=mol_len,
        )
        for k, v in task.get("test_csvs", {}).items()
    }

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_dl = {
        k: DataLoader(
            d,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        for k, d in val_ds.items()
    }
    test_dl = {
        k: DataLoader(
            d,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        for k, d in test_ds.items()
    }
    return train_dl, val_dl, test_dl


def train_one(cfg: Config, task: dict, seed: int):
    seed_everything(seed)

    bert_spec = pick_spec(cfg.encoder_specs, "bert")
    mol_spec = pick_spec(cfg.encoder_specs, "mol")

    bert_tok = build_tokenizer(bert_spec)
    mol_tok = build_tokenizer(mol_spec)
    bert_len = int(bert_spec.get("max_len", 128))
    mol_len = int(mol_spec.get("max_len", 128))

    train_dl, val_dl, test_dl = build_loaders(task, bert_tok, mol_tok, bert_len, mol_len, cfg)

    cfg.num_labels = int(task.get("num_labels", cfg.num_labels))
    task_name = task["name"]

    out_dir = os.path.join(cfg.model_save_dir, task_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    model = DDIModel(cfg).to(cfg.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best = {k: {"auc": -1.0, "path": None, "metrics": None} for k in val_dl.keys()}

    for ep in range(1, cfg.epochs + 1):
        tr_loss = train_epoch(model, train_dl, optim, cfg.device)

        msg = [f"[{task_name}] seed {seed} ep {ep:03d} | train_loss {tr_loss:.4f}"]
        for tag, ld in val_dl.items():
            v_loss, v_acc, v_auc, v_aupr = evaluate(model, ld, cfg.device, cfg.num_labels)
            msg.append(f"{tag} loss {v_loss:.4f} acc {v_acc:.4f} auc {v_auc:.4f} aupr {v_aupr:.4f}")

            if v_auc > best[tag]["auc"]:
                best[tag]["auc"] = v_auc
                best[tag]["metrics"] = (v_loss, v_acc, v_auc, v_aupr)
                p = os.path.join(out_dir, f"best_model_{tag}.pth")
                torch.save(model.state_dict(), p)
                best[tag]["path"] = p

        print(" | ".join(msg), flush=True)

    if not test_dl:
        test_dl = val_dl

    for tag, ld in test_dl.items():
        pick = None
        if tag in best and best[tag]["path"] is not None:
            pick = best[tag]["path"]
        else:
            for v in best.values():
                if v["path"] is not None:
                    pick = v["path"]
                    break

        if pick is None:
            print(f"[{task_name}] seed {seed} | skip test {tag} (no ckpt)")
            continue

        model.load_state_dict(torch.load(pick, map_location=cfg.device, weights_only=False))
        t_loss, t_acc, t_auc, t_aupr = evaluate(model, ld, cfg.device, cfg.num_labels)
        print(
            f"[{task_name}] seed {seed} TEST {tag} | "
            f"acc {t_acc:.4f} auc {t_auc:.4f} aupr {t_aupr:.4f}",
            flush=True,
        )


def eval_only(cfg: Config, task: dict, ckpt_path: str):
    bert_spec = pick_spec(cfg.encoder_specs, "bert")
    mol_spec = pick_spec(cfg.encoder_specs, "mol")

    bert_tok = build_tokenizer(bert_spec)
    mol_tok = build_tokenizer(mol_spec)
    bert_len = int(bert_spec.get("max_len", 128))
    mol_len = int(mol_spec.get("max_len", 128))

    _, val_dl, test_dl = build_loaders(task, bert_tok, mol_tok, bert_len, mol_len, cfg)
    if not test_dl:
        test_dl = val_dl

    cfg.num_labels = int(task.get("num_labels", cfg.num_labels))
    model = DDIModel(cfg).to(cfg.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=False))

    for tag, ld in test_dl.items():
        t_loss, t_acc, t_auc, t_aupr = evaluate(model, ld, cfg.device, cfg.num_labels)
        print(
            f"[{task['name']}] eval {tag} | "
            f"loss {t_loss:.4f} acc {t_acc:.4f} auc {t_auc:.4f} aupr {t_aupr:.4f}",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--task", type=str, default="all", help='task index in cfg.tasks, or "all"')
    p.add_argument("--seed", type=int, default=-1, help="if set (>=0), run only this seed")
    p.add_argument("--ckpt", type=str, default="", help="required for --mode eval")
    args = p.parse_args()

    cfg = Config()
    os.makedirs(cfg.model_save_dir, exist_ok=True)

    if args.mode == "eval":
        if not args.ckpt:
            raise SystemExit("--ckpt is required for --mode eval")
        if args.task == "all":
            raise SystemExit('--mode eval requires a single --task index (not "all")')
        task = cfg.tasks[int(args.task)]
        eval_only(cfg, task, args.ckpt)
        return

    tasks = cfg.tasks if args.task == "all" else [cfg.tasks[int(args.task)]]
    seeds = [args.seed] if args.seed >= 0 else [int(s) for s in cfg.seeds]

    for task in tasks:
        for seed in seeds:
            train_one(cfg, task, seed)


if __name__ == "__main__":
    main()
