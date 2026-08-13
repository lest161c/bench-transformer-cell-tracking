"""Probe training loop and metrics computation.

Provides the training step (forward + BCE loss + Adam optimizer) and
the early-stopping protocol used by both feature-based probes and
the end-to-end CNN+probe model. The training loop restores the
best-validation state before returning metrics.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from src.edge_probing.harness.edge_datasets import _gather_labels, make_balanced_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("edge_probing.training")


def compute_metrics(scores, target):
    """Compute all metrics from logits and binary targets."""
    pred = (scores > 0.0).float()
    target_np = target.cpu().numpy()
    pred_np = pred.cpu().numpy()
    bal_acc = balanced_accuracy_score(target_np, pred_np)
    f1 = f1_score(target_np, pred_np, zero_division=0)
    prec = precision_score(target_np, pred_np, zero_division=0)
    rec = recall_score(target_np, pred_np, zero_division=0)
    return bal_acc, f1, prec, rec


def train_probe(probe, train_loader, val_loader, epochs=200, lr=1e-3,
                patience=10, eval_every=1, is_e2e=False):
    """
    Train a probe (or CNN+probe) on pair-level data.
    Returns dict of metrics.
    """
    probe = probe.to(device)
    if is_e2e:
        # End-to-end: train all params (CNN + probe)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_bal_acc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        probe.train()
        train_losses = []
        for feats_teacher, feats_student, target in train_loader:
            feats_teacher = feats_teacher.to(device)
            feats_student = feats_student.to(device)
            target = target.to(device)

            scores = probe(feats_teacher, feats_student)
            loss = criterion(scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate
        if epoch % eval_every == 0 or epoch == epochs - 1:
            probe.eval()
            val_losses = []
            val_metrics_list = []
            with torch.no_grad():
                for feats_teacher, feats_student, target in val_loader:
                    feats_teacher = feats_teacher.to(device)
                    feats_student = feats_student.to(device)
                    target = target.to(device)

                    scores = probe(feats_teacher, feats_student)
                    loss = criterion(scores, target)
                    val_losses.append(loss.item())
                    bal_acc, f1, prec, rec = compute_metrics(scores, target)
                    val_metrics_list.append({
                        "bal_acc": bal_acc, "f1": f1,
                        "precision": prec, "recall": rec,
                    })

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss = float(np.mean(val_losses))
            avg_val_bal_acc = float(np.mean([m["bal_acc"] for m in val_metrics_list]))
            avg_val_f1 = float(np.mean([m["f1"] for m in val_metrics_list]))

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_bal_acc": avg_val_bal_acc,
                "val_f1": avg_val_f1,
            })

            # Early stopping
            if avg_val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = avg_val_bal_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"      Early stopping at epoch {epoch}")
                    break

    # Restore best state
    if best_state is not None:
        probe.load_state_dict(best_state)

    # Final evaluation
    probe.eval()
    final_metrics_list = []
    with torch.no_grad():
        for feats_teacher, feats_student, target in val_loader:
            feats_teacher = feats_teacher.to(device)
            feats_student = feats_student.to(device)
            target = target.to(device)

            scores = probe(feats_teacher, feats_student)
            bal_acc, f1, prec, rec = compute_metrics(scores, target)
            final_metrics_list.append({
                "bal_acc": bal_acc, "f1": f1,
                "precision": prec, "recall": rec,
            })

    result = {
        "final_bal_acc": float(np.mean([m["bal_acc"] for m in final_metrics_list])),
        "final_f1": float(np.mean([m["f1"] for m in final_metrics_list])),
        "final_precision": float(np.mean([m["precision"] for m in final_metrics_list])),
        "final_recall": float(np.mean([m["recall"] for m in final_metrics_list])),
        "history": history,
    }
    return result
