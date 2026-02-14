#!/usr/bin/env python3
"""
QCNN — Quantum Convolutional Neural Network (Spatial Domain, Global Expert)
============================================================================
Adapted from QCNN_Comparison.py (Cong et al. 2019) for the Comparative
Advantage experiment.

Key adaptation: the classical Conv2d feature extractor is replaced with a
simple nn.Linear layer so that the model has **no locality assumption**.
All spatial/global structure comes exclusively from the quantum circuit's
entanglement, making it the proper "Global Expert" counterpart to a
locality-dependent CNN baseline.

Input: flat 1D tensor (batch, flat_dim) from Load_Image_Datasets.py
       — already feature-shuffled when shuffle_pct > 0.

Datasets: MNIST (784-dim, 10 classes), CIFAR-10 (3072-dim, 10 classes).
"""

import os
import random
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import scipy.constants  # Must import before pennylane (lazy loading fix)
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

import pennylane as qml

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
    qml.numpy.random.seed(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class QCNN(nn.Module):
    """
    Quantum CNN from Cong et al. (2019) with nearest-neighbour entanglement.

    Architecture
    ------------
    1. Linear projection: flat_dim -> n_qubits  (no locality)
    2. AngleEmbedding (RY)
    3. [Convolutional layer + Pooling layer] x n_layers
       - Convolution: nearest-neighbour U3-Ising-U3 blocks
       - Pooling: mid-circuit measurement with conditional U3
    4. ArbitraryUnitary on the 2 remaining qubits
    5. Measurement: <Z> on each remaining qubit → n_remaining values
    6. Linear: n_remaining -> num_classes

    Constraint: n_qubits >= 2^(n_layers+1) so that at least 2 qubits remain
    after all pooling layers (ArbitraryUnitary needs 4^2-1 = 15 params).
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 2,
                 input_dim: int = 784, num_classes: int = 10):
        super().__init__()

        remaining = n_qubits // (2 ** n_layers)
        assert remaining >= 2, (
            f"After {n_layers} pooling layers, only {remaining} qubit(s) "
            f"remain.  Need >= 2.  Increase n_qubits or decrease n_layers."
        )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_remaining = remaining
        self.num_classes = num_classes

        # --- Classical layers (locality-free) ---
        self.fc_in = nn.Linear(input_dim, n_qubits)

        # --- Quantum parameters ---
        self.conv_params = nn.Parameter(torch.randn(n_layers, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(n_layers, n_qubits // 2, 3))
        self.last_params = nn.Parameter(torch.randn(15))  # 4^2 - 1

        # --- Output ---
        self.fc_out = nn.Linear(n_qubits, num_classes)

        # --- PennyLane device ---
        # No fixed wire count: mid-circuit measurements add auxiliary wires
        self.dev = qml.device("default.qubit")

    # -- quantum circuit --------------------------------------------------
    def circuit(self, conv_weights, pool_weights, last_weights, features):
        wires = list(range(self.n_qubits))

        # Variational angle embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(self.n_layers):
            self._apply_convolution(conv_weights[layer], wires)
            self._apply_pooling(pool_weights[layer], wires)
            wires = wires[::2]  # halve active wires

        # Final unitary on remaining qubits
        qml.ArbitraryUnitary(last_weights, wires)

        # Measure PauliZ on all qubits
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

    def _apply_convolution(self, weights, wires):
        """Nearest-neighbour convolutional layer (original QCNN)."""
        n_wires = len(wires)
        for parity in [0, 1]:
            for idx, w in enumerate(wires):
                if idx % 2 == parity and idx < n_wires - 1:
                    qml.U3(*weights[idx, :3], wires=w)
                    qml.U3(*weights[idx + 1, 3:6], wires=wires[idx + 1])
                    qml.IsingZZ(weights[idx, 6], wires=[w, wires[idx + 1]])
                    qml.IsingYY(weights[idx, 7], wires=[w, wires[idx + 1]])
                    qml.IsingXX(weights[idx, 8], wires=[w, wires[idx + 1]])
                    qml.U3(*weights[idx, 9:12], wires=w)
                    qml.U3(*weights[idx + 1, 12:], wires=wires[idx + 1])

    def _apply_pooling(self, pool_weights, wires):
        """Pooling via mid-circuit measurement with conditional U3."""
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."
        for idx, w in enumerate(wires):
            if idx % 2 == 1 and idx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(
                    *pool_weights[idx // 2], wires=wires[idx - 1]
                )

    # -- forward -----------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: (batch, flat_dim) — flattened image features
        Returns:
            logits: (batch, num_classes)
        """
        # Linear dimension reduction -> tanh to [-1, 1] for angle embedding
        reduced_x = torch.tanh(self.fc_in(x))

        # Quantum circuit execution (PennyLane auto-broadcasts over batch)
        qnode = qml.qnode(self.dev, interface="torch")(self.circuit)
        quantum_out = qnode(
            self.conv_params, self.pool_params, self.last_params, reduced_x
        )
        # quantum_out is a tuple of (batch,) tensors — stack to (batch, n_measurement)
        quantum_out = torch.stack(quantum_out, dim=1).to(torch.float32)

        logits = self.fc_out(quantum_out)
        return logits


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
    print(f"QCNN Training — {args.dataset.upper()}")
    print(f"{'=' * 65}")
    print(f"  Device       : {device}")
    print(f"  Seed         : {args.seed}")
    print(f"  Shuffle %    : {args.shuffle_pct}")
    print(f"  Shuffle seed : {args.shuffle_seed}")
    print(f"  n_qubits     : {args.n_qubits}")
    print(f"  n_layers     : {args.n_layers}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  Epochs       : {args.n_epochs}")
    print(f"  Patience     : {args.patience}")

    # --- Model ---
    model = QCNN(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        input_dim=cfg['input_dim'],
        num_classes=cfg['num_classes'],
    ).to(device)

    n_params = count_parameters(model)
    print(f"  Parameters   : {n_params:,}")
    print(f"{'=' * 65}\n")

    # --- Optimizer / scheduler / criterion ---
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # --- Resume ---
    tag = f"QCNN_{args.dataset}_shuf{args.shuffle_pct}_seed{args.seed}"
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
        'model': 'QCNN',
        'dataset': args.dataset,
        'shuffle_pct': args.shuffle_pct,
        'shuffle_seed': args.shuffle_seed,
        'seed': args.seed,
        'n_qubits': args.n_qubits,
        'n_layers': args.n_layers,
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
        description="QCNN (Cong et al. 2019) — Spatial Domain Global Expert"
    )

    # Model
    p.add_argument('--n-qubits', type=int, default=8)
    p.add_argument('--n-layers', type=int, default=2)

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
    p.add_argument('--output-dir', type=str, default='./results/qcnn')
    p.add_argument('--job-id', type=str, default='')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
