
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_probs, num_classes):
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)

    if num_classes == 2:
        auroc = roc_auc_score(y_true, y_probs[:, 1])
        aupr = average_precision_score(y_true, y_probs[:, 1])
    else:
        y_true_oh = np.eye(num_classes)[y_true]
        auroc = roc_auc_score(y_true_oh, y_probs, average="micro", multi_class="ovr")
        aupr = average_precision_score(y_true_oh, y_probs, average="micro")
    return acc, auroc, aupr
