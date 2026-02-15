#!/usr/bin/env python3
"""
TCN Baseline — Temporal Convolutional Network (Temporal Domain, Local Expert)
=============================================================================
Causal TCN with dilated Conv1d for time-series classification and regression.

The locality mechanism: Conv1d with small kernel_size (default 3) slides across
the time axis.  Each kernel only sees `kernel_size` consecutive timesteps.
Dilations (1, 2, 4, ...) expand the receptive field hierarchically but each
individual convolution operation is local.  When features are shuffled across
time, the causal/temporal inductive bias is destroyed.

Input: 3D tensor (batch, n_channels, n_timesteps) from Load_TimeSeries_Datasets.py
       — already feature-shuffled when shuffle_pct > 0.

Datasets:
    - EEG (64 ch, classification, 2 classes)
    - ETTh1 (7 ch, regression, 1 output)
    - Weather (21 ch, regression, 1 output)
"""

import os
import random
import argparse
import math
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

from Load_TimeSeries_Datasets import load_ts_data, DatasetResult


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
class CausalConv1d(nn.Module):
    """Conv1d with left-only causal padding (no future information leak)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        # Chop off the right side to enforce causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Two causal convolutions + residual connection.

    CausalConv1d -> ReLU -> Dropout -> CausalConv1d -> ReLU -> Dropout
    + Residual (1x1 conv if channel dims differ, else identity)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection if needed
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        return out + self.residual(x)


class TCN(nn.Module):
    """
    Temporal Convolutional Network — the classical "Local Expert" for
    the temporal domain.

    Architecture
    ------------
    Input: (batch, n_channels, n_timesteps)
    TemporalBlock x n_blocks (dilations 1, 2, 4, ...)
    GlobalAvgPool over time -> (batch, hidden_channels)
    Linear(hidden_channels, output_dim)
    """

    def __init__(self, n_channels, hidden_channels=16, n_blocks=2,
                 kernel_size=3, dropout=0.1, output_dim=2):
        super().__init__()

        blocks = []
        for i in range(n_blocks):
            in_ch = n_channels if i == 0 else hidden_channels
            dilation = 2 ** i
            blocks.append(TemporalBlock(in_ch, hidden_channels, kernel_size,
                                        dilation, dropout))
        self.network = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_channels, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timesteps)
        Returns:
            (batch, output_dim) for classification
            (batch,) for regression (output_dim == 1)
        """
        out = self.network(x)            # (batch, H, T)
        out = out.mean(dim=2)            # Global avg pool -> (batch, H)
        out = self.fc(out)               # (batch, output_dim)
        if self.output_dim == 1:
            out = out.squeeze(-1)        # (batch,) for regression
        return out


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
DS_CONFIG = {
    'eeg':     {'task': 'classification', 'output_dim': 2},
    'etth1':   {'task': 'regression',     'output_dim': 1},
    'weather': {'task': 'regression',     'output_dim': 1},
}


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
# Train / Evaluate — classification
# ---------------------------------------------------------------------------
def train_epoch_cls(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for data, target in tqdm(loader, desc="  Train", leave=False):
        data, target = data.to(device), target.to(device).long()
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
def evaluate_cls(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for data, target in tqdm(loader, desc="  Eval ", leave=False):
        data, target = data.to(device), target.to(device).long()
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        probs = torch.softmax(output, dim=1)
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_probs.append(probs.cpu())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    try:
        all_probs_np = torch.cat(all_probs).numpy()
        auc = roc_auc_score(all_labels, all_probs_np, multi_class='ovr')
    except Exception:
        auc = float('nan')

    return avg_loss, accuracy, auc


# ---------------------------------------------------------------------------
# Train / Evaluate — regression
# ---------------------------------------------------------------------------
def train_epoch_reg(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for data, target in tqdm(loader, desc="  Train", leave=False):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(output.detach().cpu().numpy())
        all_labels.extend(target.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    rmse = math.sqrt(np.mean((np.array(all_preds) - np.array(all_labels)) ** 2))
    return avg_loss, rmse


@torch.no_grad()
def evaluate_reg(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for data, target in tqdm(loader, desc="  Eval ", leave=False):
        data, target = data.to(device), target.to(device).float()
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        all_preds.extend(output.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    rmse = math.sqrt(np.mean((np.array(all_preds) - np.array(all_labels)) ** 2))
    return avg_loss, rmse


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

    # --- Task config ---
    if args.dataset not in DS_CONFIG:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from {list(DS_CONFIG)}")
    cfg = DS_CONFIG[args.dataset]
    task = args.task if args.task else cfg['task']
    output_dim = cfg['output_dim']
    is_cls = (task == 'classification')

    # --- Load data ---
    extra_kwargs = {}
    if args.dataset == 'eeg':
        extra_kwargs['sampling_freq'] = args.sampling_freq
        extra_kwargs['sample_size'] = args.eeg_subjects
    else:
        extra_kwargs['seq_len'] = args.seq_len

    result = load_ts_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        shuffle_pct=args.shuffle_pct,
        shuffle_seed=args.shuffle_seed,
        seed=args.seed,
        device=device,
        **extra_kwargs,
    )
    train_loader = result.train_loader
    val_loader = result.val_loader
    test_loader = result.test_loader

    # Probe input shape from first batch
    sample_x, _ = next(iter(train_loader))
    n_channels = sample_x.shape[1]
    n_timesteps = sample_x.shape[2]

    print(f"{'=' * 65}")
    print(f"TCN Baseline Training — {args.dataset.upper()} ({task})")
    print(f"{'=' * 65}")
    print(f"  Device          : {device}")
    print(f"  Seed            : {args.seed}")
    print(f"  Shuffle %       : {args.shuffle_pct}")
    print(f"  Shuffle seed    : {args.shuffle_seed}")
    print(f"  Input shape     : ({n_channels}, {n_timesteps})")
    print(f"  Hidden channels : {args.hidden_channels}")
    print(f"  N blocks        : {args.n_blocks}")
    print(f"  Kernel size     : {args.kernel_size}")
    print(f"  Dropout         : {args.dropout}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  LR              : {args.lr}")
    print(f"  Epochs          : {args.n_epochs}")
    print(f"  Patience        : {args.patience}")

    # --- Model ---
    model = TCN(
        n_channels=n_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        output_dim=output_dim,
    ).to(device)

    n_params = count_parameters(model)
    print(f"  Parameters      : {n_params:,}")
    print(f"{'=' * 65}\n")

    # --- Optimizer / scheduler / criterion ---
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if is_cls:
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=args.patience, mode='max')
        metric_name = 'val_acc'
    else:
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=args.patience, mode='min')
        metric_name = 'val_rmse'

    # --- Resume ---
    tag = f"TCN_{args.dataset}_shuf{args.shuffle_pct}_seed{args.seed}"
    best_path = output_dir / f"{tag}_best.pt"
    ckpt_path = ckpt_dir / f"{tag}_checkpoint.pt"
    start_epoch, history = 0, []
    best_val_metric = 0.0 if is_cls else float('inf')

    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_metric = ckpt['best_val_metric']
        history = ckpt['history']
        print(f"  Resuming from epoch {start_epoch}, best {metric_name} {best_val_metric:.4f}")

    # --- Training ---
    best_epoch = start_epoch
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()

        if is_cls:
            train_loss, train_metric = train_epoch_cls(
                model, train_loader, criterion, optimizer, device)
            val_loss, val_metric, val_auc = evaluate_cls(
                model, val_loader, criterion, device)
            scheduler.step(val_metric)
        else:
            train_loss, train_metric = train_epoch_reg(
                model, train_loader, criterion, optimizer, device)
            val_loss, val_metric = evaluate_reg(
                model, val_loader, criterion, device)
            scheduler.step(val_metric)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        if is_cls:
            print(
                f"Epoch {epoch+1:3d}/{args.n_epochs} | "
                f"Train Loss {train_loss:.4f}  Acc {train_metric:.4f} | "
                f"Val Loss {val_loss:.4f}  Acc {val_metric:.4f}  AUC {val_auc:.4f} | "
                f"LR {lr_now:.2e} | {elapsed:.1f}s"
            )
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_metric,
                'val_loss': val_loss,
                'val_acc': val_metric,
                'val_auc': val_auc,
                'lr': lr_now,
            })
            is_better = val_metric > best_val_metric
        else:
            print(
                f"Epoch {epoch+1:3d}/{args.n_epochs} | "
                f"Train Loss {train_loss:.4f}  RMSE {train_metric:.4f} | "
                f"Val Loss {val_loss:.4f}  RMSE {val_metric:.4f} | "
                f"LR {lr_now:.2e} | {elapsed:.1f}s"
            )
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_rmse': train_metric,
                'val_loss': val_loss,
                'val_rmse': val_metric,
                'lr': lr_now,
            })
            is_better = val_metric < best_val_metric

        # Best model
        if is_better:
            best_val_metric = val_metric
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metric': val_metric,
                'seed': args.seed,
            }, best_path)
            print(f"  -> New best {metric_name}: {val_metric:.4f}")

        # Checkpoint (every epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_metric': best_val_metric,
            'history': history,
        }, ckpt_path)

        if early_stopping(val_metric, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    # --- Test best model ---
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])

    print(f"\n{'=' * 65}")
    print(f"Test Results  (best epoch {best_epoch})")

    if is_cls:
        test_loss, test_acc, test_auc = evaluate_cls(
            model, test_loader, criterion, device)
        print(f"  Loss     : {test_loss:.4f}")
        print(f"  Accuracy : {test_acc:.4f}")
        print(f"  ROC-AUC  : {test_auc:.4f}")
    else:
        test_loss, test_rmse = evaluate_reg(
            model, test_loader, criterion, device)
        print(f"  Loss     : {test_loss:.4f}")
        print(f"  RMSE     : {test_rmse:.4f}")

    print(f"{'=' * 65}")

    # --- Save history & summary ---
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / f"{tag}_history.csv", index=False)

    summary = {
        'model': 'TCN',
        'dataset': args.dataset,
        'task': task,
        'shuffle_pct': args.shuffle_pct,
        'shuffle_seed': args.shuffle_seed,
        'seed': args.seed,
        'hidden_channels': args.hidden_channels,
        'n_blocks': args.n_blocks,
        'kernel_size': args.kernel_size,
        'n_params': n_params,
        'best_epoch': best_epoch,
    }
    if is_cls:
        summary.update({
            'best_val_acc': best_val_metric,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_auc,
        })
    else:
        summary.update({
            'best_val_rmse': best_val_metric,
            'test_loss': test_loss,
            'test_rmse': test_rmse,
        })

    summary_path = output_dir / f"{tag}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"\nResults saved to {output_dir}/")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="TCN Baseline — Temporal Domain Local Expert (dilated causal Conv1d)"
    )

    # Model
    p.add_argument('--hidden-channels', type=int, default=16)
    p.add_argument('--n-blocks', type=int, default=2)
    p.add_argument('--kernel-size', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)

    # Data
    p.add_argument('--dataset', type=str, default='etth1',
                   choices=['eeg', 'etth1', 'weather'])
    p.add_argument('--task', type=str, default='',
                   choices=['', 'classification', 'regression'],
                   help='Override inferred task (default: infer from dataset)')
    p.add_argument('--shuffle-pct', type=int, default=0,
                   help='Feature shuffle percentage (0-100)')
    p.add_argument('--shuffle-seed', type=int, default=42,
                   help='RNG seed for feature permutation')
    p.add_argument('--batch-size', type=int, default=32)

    # EEG-specific
    p.add_argument('--sampling-freq', type=int, default=16,
                   help='EEG resampling frequency in Hz')
    p.add_argument('--eeg-subjects', type=int, default=50,
                   help='Number of EEG subjects to load')

    # Forecasting-specific
    p.add_argument('--seq-len', type=int, default=96,
                   help='Look-back window length for forecasting')

    # Training
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--resume', action='store_true', default=False)

    # Output
    p.add_argument('--output-dir', type=str, default='./results/tcn')
    p.add_argument('--job-id', type=str, default='')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
