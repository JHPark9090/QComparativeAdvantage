#!/usr/bin/env python3
"""
LocalTransformer Baseline — Windowed Self-Attention (Attention Domain, Local Expert)
====================================================================================
Transformer with windowed (local) self-attention for time-series classification
and regression.

The locality mechanism: The attention mask restricts each token to attending only
within a window of `window_size` neighbors.  Tokens outside the window are masked
to -inf.  When features are shuffled, semantically related features move outside
the window and become unreachable — destroying the local attention inductive bias.

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
# Local attention mask
# ---------------------------------------------------------------------------
def _make_local_mask(T: int, window_size: int) -> torch.Tensor:
    """
    Create a (T, T) attention mask for windowed local attention.

    Each token i attends only to tokens in [i - half_w, i + half_w].
    Positions outside the window are set to -inf; inside to 0.0.
    """
    mask = torch.full((T, T), float('-inf'))
    half_w = window_size // 2
    for i in range(T):
        lo = max(0, i - half_w)
        hi = min(T, i + half_w + 1)
        mask[i, lo:hi] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017). No trainable params."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (batch, T, d_model)"""
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LocalTransformerLayer(nn.Module):
    """
    Single transformer layer: MHA + FFN + 2x LayerNorm + residual connections.
    Accepts an external attn_mask for local attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                         batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """x: (batch, T, d_model)"""
        # MHA + residual
        attn_out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual
        x = self.norm2(x + self.ffn(x))
        return x


class LocalTransformer(nn.Module):
    """
    Transformer with windowed self-attention — the classical "Local Expert"
    for the attention domain.

    Architecture
    ------------
    Input: (batch, n_channels, n_timesteps)
    Permute -> (batch, T, C)
    Linear(C, d_model)         — input projection
    + SinusoidalPE             — positional encoding (0 trainable params)
    LocalTransformerLayer x n_layers  (with local attention mask)
    GlobalAvgPool over T       -> (batch, d_model)
    Linear(d_model, output_dim)
    """

    def __init__(self, n_channels: int, d_model: int = 16, n_heads: int = 2,
                 n_layers: int = 1, window_size: int = 5, dropout: float = 0.1,
                 output_dim: int = 2):
        super().__init__()

        self.input_proj = nn.Linear(n_channels, d_model)
        self.pe = SinusoidalPE(d_model)

        self.layers = nn.ModuleList([
            LocalTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.fc = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim
        self.window_size = window_size

        # Mask is registered as buffer once we know T (set in forward)
        self._mask = None
        self._mask_T = -1

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Lazily create and cache the local attention mask."""
        if self._mask_T != T or self._mask is None:
            self._mask = _make_local_mask(T, self.window_size).to(device)
            self._mask_T = T
        return self._mask

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timesteps)
        Returns:
            (batch, output_dim) for classification
            (batch,) for regression (output_dim == 1)
        """
        # (batch, C, T) -> (batch, T, C)
        x = x.permute(0, 2, 1)
        T = x.size(1)

        # Input projection + positional encoding
        x = self.input_proj(x)     # (batch, T, d_model)
        x = self.pe(x)

        # Transformer layers with local mask
        mask = self._get_mask(T, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        # Global average pool over time
        x = x.mean(dim=1)          # (batch, d_model)
        x = self.fc(x)             # (batch, output_dim)

        if self.output_dim == 1:
            x = x.squeeze(-1)      # (batch,) for regression
        return x


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
    print(f"LocalTransformer Baseline Training — {args.dataset.upper()} ({task})")
    print(f"{'=' * 65}")
    print(f"  Device          : {device}")
    print(f"  Seed            : {args.seed}")
    print(f"  Shuffle %       : {args.shuffle_pct}")
    print(f"  Shuffle seed    : {args.shuffle_seed}")
    print(f"  Input shape     : ({n_channels}, {n_timesteps})")
    print(f"  d_model         : {args.d_model}")
    print(f"  n_heads         : {args.n_heads}")
    print(f"  n_attn_layers   : {args.n_attn_layers}")
    print(f"  window_size     : {args.window_size}")
    print(f"  Dropout         : {args.dropout}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  LR              : {args.lr}")
    print(f"  Epochs          : {args.n_epochs}")
    print(f"  Patience        : {args.patience}")

    # --- Model ---
    model = LocalTransformer(
        n_channels=n_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_attn_layers,
        window_size=args.window_size,
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
    tag = f"LocalTransformer_{args.dataset}_shuf{args.shuffle_pct}_seed{args.seed}"
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
        'model': 'LocalTransformer',
        'dataset': args.dataset,
        'task': task,
        'shuffle_pct': args.shuffle_pct,
        'shuffle_seed': args.shuffle_seed,
        'seed': args.seed,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_attn_layers': args.n_attn_layers,
        'window_size': args.window_size,
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
        description="LocalTransformer Baseline — Attention Domain Local Expert "
                    "(windowed self-attention)"
    )

    # Model
    p.add_argument('--d-model', type=int, default=16,
                   help='Transformer embedding dimension')
    p.add_argument('--n-heads', type=int, default=2,
                   help='Number of attention heads')
    p.add_argument('--n-attn-layers', type=int, default=1,
                   help='Number of transformer layers')
    p.add_argument('--window-size', type=int, default=5,
                   help='Local attention window size')
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
    p.add_argument('--output-dir', type=str, default='./results/local_transformer')
    p.add_argument('--job-id', type=str, default='')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
