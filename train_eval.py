import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import calculate_metrics
from sklearn.metrics import roc_auc_score, average_precision_score

def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0
    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader, device, num_classes):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"]

            logits = model(batch)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0) 
    all_labels = np.concatenate(all_labels, axis=0) 


    preds = np.argmax(all_probs, axis=1)
    acc = (preds == all_labels).mean()

    if num_classes == 2:
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        aupr = average_precision_score(all_labels, all_probs[:, 1])
    else:
        labels_onehot = np.eye(num_classes)[all_labels]
        auroc = roc_auc_score(
            labels_onehot,
            all_probs,
            average="micro",
            multi_class="ovr"
        )
        aupr = average_precision_score(
            labels_onehot,
            all_probs,
            average="micro"
        )

    return total_loss / len(loader), acc, auroc, aupr