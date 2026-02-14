"""
Time-series data loaders with Feature Shuffling Protocol support.

Datasets:
    - EEG: PhysioNet Motor Imagery (64 channels, configurable sampling freq)
    - ETTh1: Electricity Transformer Temperature (7 features, hourly)
    - Weather: 21 meteorological features, 10-min resolution

All loaders return DatasetResult (from Load_Image_Datasets) with:
    - 3D DataLoaders: (batch, channels, timesteps) for QTCN/QTSTransformer
    - 2D flat raw tensors on CPU for QE (Von Neumann entropy) computation
    - shuffle_perm for reproducibility of Feature Shuffling Protocol

Author: QComparativeAdvantage project
Date: February 2026
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import urllib.request
import mne

from Load_Image_Datasets import (
    DatasetResult,
    generate_feature_shuffle_permutation,
    apply_feature_shuffle,
)


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_ROOT = Path('/pscratch/sd/j/junghoon/QComparativeAdvantage/data')
EEG_DATA_PATH = '/pscratch/sd/j/junghoon/PhysioNet_EEG'


# =============================================================================
# DATASET METADATA (adapted from VQC-PeriodicData/data/real_world_datasets.py)
# =============================================================================

DATASET_INFO = {
    'etth1': {
        'target': 'OT',
        'n_features': 7,
        'freq': 'hourly',
        'description': 'Electricity Transformer Temperature (hourly)',
        'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
        'split': 'informer',  # 12/4/4 months
    },
    'weather': {
        'target': 'OT',
        'n_features': 21,
        'freq': '10min',
        'description': 'Weather (21 meteorological features, 10-min)',
        'url': None,  # Google Drive — manual download
        'split': 'ratio',  # 0.7/0.1/0.2
    },
}


# =============================================================================
# SEQUENCE CREATION (from VQC-PeriodicData/data/real_world_datasets.py)
# =============================================================================

def create_multivariate_sequences(data, target, seq_len, pred_len=1):
    """
    Create sliding window sequences for multivariate forecasting.

    Args:
        data: Input features, shape (timesteps, n_features)
        target: Target values, shape (timesteps,)
        seq_len: Look-back window length
        pred_len: Prediction horizon (1 = next-step)

    Returns:
        x: shape (n_sequences, n_features, seq_len) — channels-first for Conv1d
        y: shape (n_sequences,) — scalar target at pred_len steps ahead
    """
    n_total = len(data) - seq_len - pred_len + 1
    if n_total <= 0:
        raise ValueError(
            f"Not enough data: {len(data)} steps for seq_len={seq_len}, "
            f"pred_len={pred_len}. Need at least {seq_len + pred_len} steps."
        )

    x = np.zeros((n_total, data.shape[1], seq_len), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.float32)

    for i in range(n_total):
        x[i] = data[i:i + seq_len].T  # (n_features, seq_len)
        y[i] = target[i + seq_len + pred_len - 1]

    return x, y


# =============================================================================
# DATA LOADING HELPERS (from VQC-PeriodicData/data/real_world_datasets.py)
# =============================================================================

def _download_etth1(data_dir):
    """Download ETTh1 CSV from GitHub if not present."""
    data_dir = Path(data_dir)
    save_path = data_dir / 'ETTh1.csv'

    if save_path.exists():
        return str(save_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    url = DATASET_INFO['etth1']['url']
    print(f"Downloading ETTh1 from {url}...")

    try:
        urllib.request.urlretrieve(url, str(save_path))
        print(f"Saved to {save_path}")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download ETTh1: {e}\n"
            f"Please manually download from:\n"
            f"  {url}\n"
            f"And save to: {save_path}"
        )

    return str(save_path)


def _load_csv(dataset_name, data_path=None):
    """Load dataset CSV with auto-download for ETTh1."""
    if dataset_name == 'etth1':
        if data_path is None:
            data_path = _download_etth1(str(DATA_ROOT / 'etth1'))
        df = pd.read_csv(data_path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        return df

    elif dataset_name == 'weather':
        if data_path is None:
            default_path = DATA_ROOT / 'weather' / 'weather.csv'
            if default_path.exists():
                data_path = str(default_path)
            else:
                raise FileNotFoundError(
                    f"Weather dataset not found at {default_path}\n"
                    f"Please download from the Autoformer repository:\n"
                    f"  https://drive.google.com/drive/folders/1ohGYWWfm4i9LC71pE29fhz4Y_UZcQYMZ\n"
                    f"Save 'weather.csv' to: {default_path.parent}/"
                )
        df = pd.read_csv(data_path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        return df

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: etth1, weather")


def _get_split_indices(dataset_name, n_total):
    """Return (train_end, val_end) indices for the dataset split."""
    if dataset_name == 'etth1':
        # Informer standard split: 12/4/4 months of hourly data
        train_end = 12 * 30 * 24  # 8640
        val_end = train_end + 4 * 30 * 24  # 11520
        train_end = min(train_end, int(0.6 * n_total))
        val_end = min(val_end, int(0.8 * n_total))
    else:
        # Standard 0.7/0.1/0.2 split for weather
        train_end = int(0.7 * n_total)
        val_end = int(0.8 * n_total)

    return train_end, val_end


# =============================================================================
# SHUFFLE + PACKAGE HELPER
# =============================================================================

def _shuffle_and_package(
    X_train_3d, y_train,
    X_val_3d, y_val,
    X_test_3d, y_test,
    device, batch_size,
    shuffle_pct, shuffle_seed,
):
    """
    Core function shared by all 3 loaders.
    Handles flatten -> shuffle -> reshape -> package pipeline.

    Args:
        X_train_3d, X_val_3d, X_test_3d: numpy arrays of shape (n, C, T)
        y_train, y_val, y_test: numpy arrays of shape (n,)
        device: torch device for DataLoader tensors
        batch_size: batch size for DataLoaders
        shuffle_pct: percentage of features to shuffle (0-100)
        shuffle_seed: seed for deterministic shuffle permutation

    Returns:
        DatasetResult with 3D DataLoaders and 2D flat raw tensors
    """
    n_channels = X_train_3d.shape[1]
    n_timesteps = X_train_3d.shape[2]
    flat_dim = n_channels * n_timesteps

    # Flatten: (n, C, T) -> (n, C*T)
    X_train_flat = torch.from_numpy(X_train_3d.reshape(-1, flat_dim))
    X_val_flat = torch.from_numpy(X_val_3d.reshape(-1, flat_dim))
    X_test_flat = torch.from_numpy(X_test_3d.reshape(-1, flat_dim))

    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.float32))

    # Feature shuffling
    perm = generate_feature_shuffle_permutation(flat_dim, shuffle_pct, seed=shuffle_seed)
    X_train_flat = apply_feature_shuffle(X_train_flat, perm)
    X_val_flat = apply_feature_shuffle(X_val_flat, perm)
    X_test_flat = apply_feature_shuffle(X_test_flat, perm)

    # Save raw CPU copies (2D flat) for QE computation
    X_train_raw = X_train_flat.clone()
    y_train_raw = y_train_t.clone()
    X_val_raw = X_val_flat.clone()
    y_val_raw = y_val_t.clone()
    X_test_raw = X_test_flat.clone()
    y_test_raw = y_test_t.clone()

    # Reshape back to 3D: (n, C*T) -> (n, C, T)
    X_train_3d_t = X_train_flat.reshape(-1, n_channels, n_timesteps).to(device)
    X_val_3d_t = X_val_flat.reshape(-1, n_channels, n_timesteps).to(device)
    X_test_3d_t = X_test_flat.reshape(-1, n_channels, n_timesteps).to(device)

    y_train_d = y_train_t.to(device)
    y_val_d = y_val_t.to(device)
    y_test_d = y_test_t.to(device)

    # Create TensorDatasets and DataLoaders (3D tensors)
    train_dataset = TensorDataset(X_train_3d_t, y_train_d)
    val_dataset = TensorDataset(X_val_3d_t, y_val_d)
    test_dataset = TensorDataset(X_test_3d_t, y_test_d)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return DatasetResult(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=flat_dim,
        X_train=X_train_raw,
        y_train=y_train_raw,
        X_val=X_val_raw,
        y_val=y_val_raw,
        X_test=X_test_raw,
        y_test=y_test_raw,
        shuffle_perm=perm,
    )


# =============================================================================
# EEG LOADER
# =============================================================================

def load_eeg(seed, device, batch_size, sampling_freq=16, sample_size=50,
             shuffle_pct=0, shuffle_seed=None):
    """
    Load PhysioNet EEG Motor Imagery dataset with feature shuffling support.

    Uses subject-level splitting (70/15/15) to prevent data leakage.

    Args:
        seed: Random seed for reproducibility
        device: torch device
        batch_size: batch size for DataLoaders
        sampling_freq: target resampling frequency (Hz)
        sample_size: number of subjects to load (1-109)
        shuffle_pct: percentage of features to shuffle (0-100)
        shuffle_seed: seed for deterministic shuffle permutation

    Returns:
        DatasetResult with 3D DataLoaders (batch, 64, T) and 2D flat raw tensors
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Subject-level split: 70% train, 15% val, 15% test
    subject_ids = np.arange(1, sample_size + 1)
    train_subjects, temp_subjects = train_test_split(
        subject_ids, test_size=0.3, random_state=seed
    )
    val_subjects, test_subjects = train_test_split(
        temp_subjects, test_size=0.5, random_state=seed
    )

    print(f"EEG subjects — train: {len(train_subjects)}, "
          f"val: {len(val_subjects)}, test: {len(test_subjects)}")

    def _load_subjects(subject_list, sfreq):
        RUNS = [4, 8, 12]
        physionet_paths = [
            mne.datasets.eegbci.load_data(
                subjects=subj_id,
                runs=RUNS,
                path=EEG_DATA_PATH,
                update_path=False,
            ) for subj_id in subject_list
        ]
        physionet_paths = np.concatenate(physionet_paths)

        parts = []
        for path in physionet_paths:
            raw = mne.io.read_raw_edf(
                path, preload=True, stim_channel='auto', verbose='WARNING'
            )
            raw.resample(sfreq, npad="auto")
            parts.append(raw)

        raw = mne.concatenate_raws(parts)
        events, _ = mne.events_from_annotations(raw)
        eeg_channel_inds = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )
        epoched = mne.Epochs(
            raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
            proj=False, picks=eeg_channel_inds, baseline=None, preload=True
        )

        X = (epoched.get_data() * 1e3).astype(np.float32)  # millivolts
        y = (epoched.events[:, 2] - 2).astype(np.int64)  # 0=left, 1=right
        return X, y

    X_train, y_train = _load_subjects(train_subjects, sampling_freq)
    X_val, y_val = _load_subjects(val_subjects, sampling_freq)
    X_test, y_test = _load_subjects(test_subjects, sampling_freq)

    print(f"EEG shapes — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # Standardize: fit on train, transform all (per-feature on flattened data)
    n_channels, n_timesteps = X_train.shape[1], X_train.shape[2]
    flat_dim = n_channels * n_timesteps

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, flat_dim)
    scaler.fit(X_train_flat)
    X_train_flat = scaler.transform(X_train_flat).astype(np.float32)
    X_val_flat = scaler.transform(X_val.reshape(-1, flat_dim)).astype(np.float32)
    X_test_flat = scaler.transform(X_test.reshape(-1, flat_dim)).astype(np.float32)

    # Reshape back to 3D for _shuffle_and_package
    X_train_3d = X_train_flat.reshape(-1, n_channels, n_timesteps)
    X_val_3d = X_val_flat.reshape(-1, n_channels, n_timesteps)
    X_test_3d = X_test_flat.reshape(-1, n_channels, n_timesteps)

    return _shuffle_and_package(
        X_train_3d, y_train,
        X_val_3d, y_val,
        X_test_3d, y_test,
        device, batch_size,
        shuffle_pct, shuffle_seed,
    )


# =============================================================================
# FORECASTING LOADER (internal)
# =============================================================================

def _load_forecasting(dataset_name, seed, device, batch_size,
                      seq_len=96, pred_len=1, normalize='standard',
                      shuffle_pct=0, shuffle_seed=None, data_path=None):
    """
    Internal loader for ETTh1 and Weather forecasting datasets.

    Args:
        dataset_name: 'etth1' or 'weather'
        seed: random seed
        device: torch device
        batch_size: batch size for DataLoaders
        seq_len: look-back window length (default 96)
        pred_len: prediction horizon (default 1)
        normalize: 'standard' (z-score) or 'none'
        shuffle_pct: percentage of features to shuffle (0-100)
        shuffle_seed: seed for deterministic shuffle permutation
        data_path: path to CSV file (auto-download for etth1 if None)

    Returns:
        DatasetResult
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    info = DATASET_INFO[dataset_name]

    # Load CSV
    df = _load_csv(dataset_name, data_path)
    print(f"Loaded {info['description']}: {df.shape}")

    # Determine target column
    target_col = info['target']
    if target_col not in df.columns:
        target_col = df.columns[-1]

    # Sequential split BEFORE normalization
    n_total = len(df)
    train_end, val_end = _get_split_indices(dataset_name, n_total)

    # Ensure target is included as input feature (all channels as input)
    feature_cols = [c for c in df.columns if c != target_col] + [target_col]
    data_array = df[feature_cols].values.astype(np.float32)
    target_idx = feature_cols.index(target_col)

    train_data = data_array[:train_end]
    val_data = data_array[train_end:val_end]
    test_data = data_array[val_end:]

    # Normalize: fit on train only
    if normalize == 'standard':
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data).astype(np.float32)
        val_data = scaler.transform(val_data).astype(np.float32)
        test_data = scaler.transform(test_data).astype(np.float32)

    # Create sequences: (n_seq, n_features, seq_len)
    X_train, y_train = create_multivariate_sequences(
        train_data, train_data[:, target_idx], seq_len, pred_len
    )
    X_val, y_val = create_multivariate_sequences(
        val_data, val_data[:, target_idx], seq_len, pred_len
    )
    X_test, y_test = create_multivariate_sequences(
        test_data, test_data[:, target_idx], seq_len, pred_len
    )

    print(f"Sequences — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    return _shuffle_and_package(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        device, batch_size,
        shuffle_pct, shuffle_seed,
    )


# =============================================================================
# PUBLIC WRAPPERS
# =============================================================================

def load_etth1(seed, device, batch_size, seq_len=96, pred_len=1,
               normalize='standard', shuffle_pct=0, shuffle_seed=None,
               data_path=None):
    """Load ETTh1 dataset. flat_dim = 7 x 96 = 672."""
    return _load_forecasting(
        'etth1', seed, device, batch_size,
        seq_len=seq_len, pred_len=pred_len, normalize=normalize,
        shuffle_pct=shuffle_pct, shuffle_seed=shuffle_seed, data_path=data_path,
    )


def load_weather(seed, device, batch_size, seq_len=96, pred_len=1,
                 normalize='standard', shuffle_pct=0, shuffle_seed=None,
                 data_path=None):
    """Load Weather dataset. flat_dim = 21 x 96 = 2016."""
    return _load_forecasting(
        'weather', seed, device, batch_size,
        seq_len=seq_len, pred_len=pred_len, normalize=normalize,
        shuffle_pct=shuffle_pct, shuffle_seed=shuffle_seed, data_path=data_path,
    )


# =============================================================================
# DISPATCHER
# =============================================================================

def load_ts_data(dataset_name, batch_size, shuffle_pct=0, shuffle_seed=None,
                 seed=2025, **kwargs):
    """
    Dispatch to the appropriate time-series loader.

    Args:
        dataset_name: 'eeg', 'etth1', or 'weather'
        batch_size: batch size for DataLoaders
        shuffle_pct: percentage of features to shuffle (0-100)
        shuffle_seed: seed for deterministic shuffle permutation
        seed: random seed (default 2025)
        **kwargs: forwarded to the specific loader

    Returns:
        DatasetResult
    """
    device = kwargs.pop('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    name = dataset_name.lower()
    if name == 'eeg':
        return load_eeg(seed=seed, device=device, batch_size=batch_size,
                        shuffle_pct=shuffle_pct, shuffle_seed=shuffle_seed, **kwargs)
    elif name == 'etth1':
        return load_etth1(seed=seed, device=device, batch_size=batch_size,
                          shuffle_pct=shuffle_pct, shuffle_seed=shuffle_seed, **kwargs)
    elif name == 'weather':
        return load_weather(seed=seed, device=device, batch_size=batch_size,
                            shuffle_pct=shuffle_pct, shuffle_seed=shuffle_seed, **kwargs)
    else:
        raise ValueError(f"Unknown time-series dataset: {dataset_name}. "
                         f"Choose from: eeg, etth1, weather")


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Time-Series Data Loader Smoke Test")
    print("=" * 60)

    # 1. ETTh1 basics
    print("\n--- Test 1: ETTh1 basic shapes ---")
    result = load_etth1(seed=2025, device='cpu', batch_size=32,
                        shuffle_pct=50, shuffle_seed=42)
    assert result.input_dim == 672, f"Expected 672, got {result.input_dim}"
    assert result.X_train.ndim == 2, f"Raw tensors should be 2D, got {result.X_train.ndim}D"
    for x, y in result.train_loader:
        assert x.shape[1:] == (7, 96), f"Expected (7, 96), got {x.shape[1:]}"
        break
    print("PASS: input_dim=672, raw 2D, loader 3D (7, 96)")

    # 2. Backward compat (tuple unpacking)
    print("\n--- Test 2: Backward compatibility (tuple unpacking) ---")
    train_loader, val_loader, test_loader, input_dim = result
    assert input_dim == 672
    print("PASS: tuple unpacking works")

    # 3. Deterministic shuffle
    print("\n--- Test 3: Deterministic shuffle ---")
    r2 = load_etth1(seed=2025, device='cpu', batch_size=32,
                    shuffle_pct=50, shuffle_seed=42)
    assert torch.equal(result.shuffle_perm, r2.shuffle_perm)
    print("PASS: same seed -> same permutation")

    # 4. Identity at shuffle_pct=0
    print("\n--- Test 4: Identity at shuffle_pct=0 ---")
    r0 = load_etth1(seed=2025, device='cpu', batch_size=32, shuffle_pct=0)
    assert torch.equal(r0.shuffle_perm, torch.arange(672))
    print("PASS: shuffle_pct=0 -> identity permutation")

    # 5. EEG (small subset)
    print("\n--- Test 5: EEG loader ---")
    result_eeg = load_eeg(seed=2025, device='cpu', batch_size=32,
                          sampling_freq=16, sample_size=10,
                          shuffle_pct=30, shuffle_seed=7)
    assert result_eeg.X_train.ndim == 2
    for x, y in result_eeg.train_loader:
        assert x.ndim == 3
        break
    print(f"PASS: EEG input_dim={result_eeg.input_dim}, raw 2D, loader 3D")

    # 6. Dispatcher
    print("\n--- Test 6: Dispatcher ---")
    result_d = load_ts_data('etth1', batch_size=32, shuffle_pct=100, shuffle_seed=99)
    assert isinstance(result_d, DatasetResult)
    assert not torch.equal(result_d.shuffle_perm, torch.arange(672))
    print("PASS: dispatcher returns DatasetResult with non-identity perm")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)
