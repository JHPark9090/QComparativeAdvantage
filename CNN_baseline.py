#!/usr/bin/env python3
"""
CNN Baseline — Classical Convolutional Neural Network (Spatial Domain, Local Expert)
===================================================================================
3-layer CNN with 3x3 kernels for image classification.

The locality mechanism: Conv2d with small (3x3) kernels imposes a local receptive
field.  When features are shuffled, adjacent pixels become uncorrelated and the
3x3 kernels see noise — destroying the spatial inductive bias.

Input: flat 1D tensor (batch, flat_dim) from Load_Image_Datasets.py
       — already feature-shuffled when shuffle_pct > 0.

Datasets: MNIST (784-dim, 10 classes), CIFAR-10 (3072-dim, 10 classes).
"""

import os
import random
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

from Load_Image_Datasets import load_data, DatasetResult


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """
    3-layer CNN with 3x3 kernels — the classical "Local Expert".

    Architecture
    ------------
    Input:  (batch, flat_dim) — from load_data
    Reshape: (batch, C_in, H, W)
    Conv2d(C_in, B, 3, padding=1) + ReLU   -> (batch, B, H, W)
    MaxPool2d(2)                            -> (batch, B, H/2, W/2)
    Conv2d(B, 2B, 3, padding=1) + ReLU     -> (batch, 2B, H/2, W/2)
    MaxPool2d(2)                            -> (batch, 2B, H/4, W/4)
    Conv2d(2B, 4B, 3, padding=1) + ReLU    -> (batch, 4B, H/4, W/4)
    AdaptiveAvgPool2d(2, 2)                 -> (batch, 4B, 2, 2)
    Flatten                                 -> (batch, 16B)
    Linear(16B, num_classes)                -> (batch, 10)
    """

    # Dataset-specific input shapes
    INPUT_SHAPES = {
        'mnist':   (1, 28, 28),
        'cifar10': (3, 32, 32),
    }

    def __init__(self, dataset: str = 'mnist', base_channels: int = 8,
                 num_classes: int = 10):
        super().__init__()

        if dataset not in self.INPUT_SHAPES:
            raise ValueError(f"Unknown dataset: {dataset}. "
                             f"Choose from {list(self.INPUT_SHAPES)}")

        self.input_shape = self.INPUT_SHAPES[dataset]
        c_in = self.input_shape[0]
        B = base_channels

        self.features = nn.Sequential(
            nn.Conv2d(c_in, B, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(B, 2 * B, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * B, 4 * B, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.classifier = nn.Linear(4 * B * 2 * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, flat_dim) — flattened image features
        Returns:
            logits: (batch, num_classes)
        """
        # Reshape flat input back to image
        x = x.view(-1, *self.input_shape)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        adjusted = score if self.mode == 'max' else -score
        if self.best_score is None:
            self.best_score = adjusted
            self.best_epoch = epoch
        elif adjusted < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = adjusted
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for data, target in tqdm(loader, desc="  Train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for data, target in tqdm(loader, desc="  Eval ", leave=False):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        probs = torch.softmax(output, dim=1)
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_probs.append(probs.cpu())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # ROC-AUC (one-vs-rest)
    try:
        all_probs_np = torch.cat(all_probs).numpy()
        auc = roc_auc_score(all_labels, all_probs_np, multi_class='ovr')
    except Exception:
        auc = float('nan')

    return avg_loss, accuracy, auc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(args.seed)

    # --- Output dirs ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset config ---
    ds_config = {
        'mnist':   {'input_dim': 784,  'num_classes': 10},
        'cifar10': {'input_dim': 3072, 'num_classes': 10},
    }
    if args.dataset not in ds_config:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from {list(ds_config)}")
    cfg = ds_config[args.dataset]

    # --- Load data ---
    result = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_valtest=args.n_valtest,
        shuffle_pct=args.shuffle_pct,
        shuffle_seed=args.shuffle_seed,
    )
    if isinstance(result, DatasetResult):
        train_loader = result.train_loader
        val_loader = result.val_loader
        test_loader = result.test_loader
    else:
        train_loader, val_loader, test_loader = result

    print(f"{'=' * 65}")
    print(f"CNN Baseline Training — {args.dataset.upper()}")
    print(f"{'=' * 65}")
    print(f"  Device        : {device}")
    print(f"  Seed          : {args.seed}")
    print(f"  Shuffle %     : {args.shuffle_pct}")
    print(f"  Shuffle seed  : {args.shuffle_seed}")
    print(f"  Base channels : {args.base_channels}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  LR            : {args.lr}")
    print(f"  Epochs        : {args.n_epochs}")
    print(f"  Patience      : {args.patience}")

    # --- Model ---
    model = CNN(
        dataset=args.dataset,
        base_channels=args.base_channels,
        num_classes=cfg['num_classes'],
    ).to(device)

    n_params = count_parameters(model)
    print(f"  Parameters    : {n_params:,}")
    print(f"{'=' * 65}\n")

    # --- Optimizer / scheduler / criterion ---
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # --- Resume ---
    tag = f"CNN_{args.dataset}_shuf{args.shuffle_pct}_seed{args.seed}"
    best_path = output_dir / f"{tag}_best.pt"
    ckpt_path = ckpt_dir / f"{tag}_checkpoint.pt"
    start_epoch, best_val_acc, history = 0, 0.0, []

    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        history = ckpt['history']
        print(f"  Resuming from epoch {start_epoch}, best val acc {best_val_acc:.4f}")

    # --- Training ---
    best_epoch = start_epoch
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d}/{args.n_epochs} | "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:.4f}  AUC {val_auc:.4f} | "
            f"LR {lr_now:.2e} | {elapsed:.1f}s"
        )

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'lr': lr_now,
        })

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'seed': args.seed,
            }, best_path)
            print(f"  -> New best val acc: {val_acc:.4f}")

        # Checkpoint (every epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
        }, ckpt_path)

        if early_stopping(val_acc, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    # --- Test best model ---
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 65}")
    print(f"Test Results  (best epoch {best_epoch})")
    print(f"  Loss     : {test_loss:.4f}")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  ROC-AUC  : {test_auc:.4f}")
    print(f"{'=' * 65}")

    # --- Save history & summary ---
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / f"{tag}_history.csv", index=False)

    summary = {
        'model': 'CNN',
        'dataset': args.dataset,
        'shuffle_pct': args.shuffle_pct,
        'shuffle_seed': args.shuffle_seed,
        'seed': args.seed,
        'base_channels': args.base_channels,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_auc': test_auc,
    }
    summary_path = output_dir / f"{tag}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"\nResults saved to {output_dir}/")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="CNN Baseline — Spatial Domain Local Expert (3x3 kernels)"
    )

    # Model
    p.add_argument('--base-channels', type=int, default=8,
                   help='Base channel width B; layers use B, 2B, 4B')

    # Data
    p.add_argument('--dataset', type=str, default='mnist',
                   choices=['mnist', 'cifar10'])
    p.add_argument('--shuffle-pct', type=int, default=0,
                   help='Feature shuffle percentage (0-100)')
    p.add_argument('--shuffle-seed', type=int, default=42,
                   help='RNG seed for feature permutation')
    p.add_argument('--n-train', type=int, default=50000)
    p.add_argument('--n-valtest', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)

    # Training
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--resume', action='store_true', default=False)

    # Output
    p.add_argument('--output-dir', type=str, default='./results/cnn')
    p.add_argument('--job-id', type=str, default='')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
