from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from signal_preprocessing.eegdnet_denoiser import _EEGDNet, _resolved_segment_length


def _rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)) + 1e-12)


def _lambda_for_target_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> float:
    return float(_rms(clean) / (_rms(noise) * (10.0 ** (float(snr_db) / 10.0))))


def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size <= 1 or arr.size == int(target_len):
        return arr.astype(np.float32, copy=True)
    src = np.linspace(0.0, 1.0, num=arr.size, endpoint=True)
    dst = np.linspace(0.0, 1.0, num=int(target_len), endpoint=True)
    out = np.interp(dst, src, arr)
    return out.astype(np.float32)


class SyntheticEEGDenoiseDataset(Dataset):
    def __init__(
        self,
        clean_eeg: np.ndarray,
        eog_noise: np.ndarray,
        emg_noise: np.ndarray,
        *,
        target_len: int,
        n_samples: int,
        snr_eog_low: float,
        snr_eog_high: float,
        snr_emg_low: float,
        snr_emg_high: float,
        eog_probability: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.clean_eeg = np.asarray(clean_eeg, dtype=np.float32)
        self.eog_noise = np.asarray(eog_noise, dtype=np.float32)
        self.emg_noise = np.asarray(emg_noise, dtype=np.float32)
        self.target_len = int(target_len)

        if self.clean_eeg.ndim != 2 or self.eog_noise.ndim != 2 or self.emg_noise.ndim != 2:
            raise ValueError("Input arrays must be 2D: [n_epochs, n_samples]")

        self.n_samples = int(n_samples)
        rng = np.random.default_rng(int(seed))

        self.clean_idx = rng.integers(0, self.clean_eeg.shape[0], size=self.n_samples, endpoint=False)
        self.use_eog = rng.random(self.n_samples) < float(eog_probability)
        self.eog_idx = rng.integers(0, self.eog_noise.shape[0], size=self.n_samples, endpoint=False)
        self.emg_idx = rng.integers(0, self.emg_noise.shape[0], size=self.n_samples, endpoint=False)

        snr_eog = rng.uniform(float(snr_eog_low), float(snr_eog_high), size=self.n_samples)
        snr_emg = rng.uniform(float(snr_emg_low), float(snr_emg_high), size=self.n_samples)
        self.snr_db = np.where(self.use_eog, snr_eog, snr_emg).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_eeg[int(self.clean_idx[idx])]
        if clean.shape[0] != self.target_len:
            clean = _resample_1d(clean, self.target_len)

        if bool(self.use_eog[idx]):
            noise = self.eog_noise[int(self.eog_idx[idx])]
        else:
            noise = self.emg_noise[int(self.emg_idx[idx])]

        if noise.shape[0] != self.target_len:
            noise = _resample_1d(noise, self.target_len)

        lam = _lambda_for_target_snr(clean, noise, float(self.snr_db[idx]))
        noisy = clean + (lam * noise)

        std_noisy = float(np.std(noisy) + 1e-12)
        clean_norm = (clean / std_noisy).astype(np.float32)
        noisy_norm = (noisy / std_noisy).astype(np.float32)

        return torch.from_numpy(noisy_norm), torch.from_numpy(clean_norm)


def _load_eegdenoisenet_arrays(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg_path = data_root / "EEG_all_epochs.npy"
    eog_path = data_root / "EOG_all_epochs.npy"
    emg_path = data_root / "EMG_all_epochs.npy"

    missing = [str(p) for p in (eeg_path, eog_path, emg_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required EEGdenoiseNet files: {missing}")

    clean_eeg = np.load(eeg_path)
    eog_noise = np.load(eog_path)
    emg_noise = np.load(emg_path)

    return clean_eeg, eog_noise, emg_noise


def _split_train_val(arr: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    val_n = max(1, int(round(n * float(val_fraction))))
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return arr[train_idx], arr[val_idx]


def _build_model(*, window_size: int, segment_length: int, depths: int, heads: int, ff_multiplier: float, dropout: float) -> _EEGDNet:
    resolved_segment = _resolved_segment_length(int(window_size), int(segment_length))
    if resolved_segment <= 0:
        raise ValueError("window_size and segment_length are incompatible")

    return _EEGDNet(
        signal_len=int(window_size),
        segment_len=int(resolved_segment),
        depths=int(depths),
        heads=int(heads),
        ff_multiplier=float(ff_multiplier),
        dropout=float(dropout),
    )


def _epoch_pass(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: str) -> float:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    n_batches = 0

    for noisy_batch, clean_batch in loader:
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            pred = model(noisy_batch)
            loss = criterion(pred, clean_batch)
            if train_mode:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return running_loss / max(1, n_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEGDNet using EEGdenoiseNet benchmark mixing protocol.")

    parser.add_argument(
        "--data-root",
        default=r"C:/Users/Admin/Documents/GitHub/EEGdenoiseNet/data",
        help="Path containing EEG_all_epochs.npy, EOG_all_epochs.npy, and EMG_all_epochs.npy",
    )
    parser.add_argument("--output", default="pipeline/models/eegdnet/eegdnet_best.pt", help="Checkpoint output path")
    parser.add_argument("--history-output", default="pipeline/models/eegdnet/eegdnet_train_history.json", help="Training history JSON output path")

    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--depths", type=int, default=6)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--ff-multiplier", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument("--val-samples", type=int, default=4000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eog-probability", type=float, default=0.5)

    parser.add_argument("--snr-eog-low", type=float, default=-7.0)
    parser.add_argument("--snr-eog-high", type=float, default=2.0)
    parser.add_argument("--snr-emg-low", type=float, default=-7.0)
    parser.add_argument("--snr-emg-high", type=float, default=4.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    data_root = Path(args.data_root)
    output_path = Path(args.output)
    history_path = Path(args.history_output)

    clean_eeg, eog_noise, emg_noise = _load_eegdenoisenet_arrays(data_root)

    clean_train, clean_val = _split_train_val(clean_eeg, args.val_fraction, seed=args.seed + 1)
    eog_train, eog_val = _split_train_val(eog_noise, args.val_fraction, seed=args.seed + 2)
    emg_train, emg_val = _split_train_val(emg_noise, args.val_fraction, seed=args.seed + 3)

    train_ds = SyntheticEEGDenoiseDataset(
        clean_train,
        eog_train,
        emg_train,
        target_len=args.window_size,
        n_samples=args.train_samples,
        snr_eog_low=args.snr_eog_low,
        snr_eog_high=args.snr_eog_high,
        snr_emg_low=args.snr_emg_low,
        snr_emg_high=args.snr_emg_high,
        eog_probability=args.eog_probability,
        seed=args.seed + 10,
    )

    val_ds = SyntheticEEGDenoiseDataset(
        clean_val,
        eog_val,
        emg_val,
        target_len=args.window_size,
        n_samples=args.val_samples,
        snr_eog_low=args.snr_eog_low,
        snr_eog_high=args.snr_eog_high,
        snr_emg_low=args.snr_emg_low,
        snr_emg_high=args.snr_emg_high,
        eog_probability=args.eog_probability,
        seed=args.seed + 20,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = _build_model(
        window_size=args.window_size,
        segment_length=args.segment_length,
        depths=args.depths,
        heads=args.heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
    )
    model.to(args.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.5, 0.9))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = math.inf

    print(f"Training on device: {args.device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Output checkpoint: {output_path}")

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _epoch_pass(model, train_loader, criterion, optimizer, args.device)
        val_loss = _epoch_pass(model, val_loader, criterion, None, args.device)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = float(val_loss)
            ckpt = {
                "state_dict": model.state_dict(),
                "model_config": {
                    "window_size": int(args.window_size),
                    "segment_length": int(args.segment_length),
                    "depths": int(args.depths),
                    "heads": int(args.heads),
                    "ff_multiplier": float(args.ff_multiplier),
                    "dropout": float(args.dropout),
                },
                "train_config": {
                    "data_root": str(data_root),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "train_samples": int(args.train_samples),
                    "val_samples": int(args.val_samples),
                    "snr_eog": [float(args.snr_eog_low), float(args.snr_eog_high)],
                    "snr_emg": [float(args.snr_emg_low), float(args.snr_emg_high)],
                    "eog_probability": float(args.eog_probability),
                    "seed": int(args.seed),
                },
                "best_val_loss": float(best_val),
            }
            torch.save(ckpt, output_path)
            print(f"Saved best checkpoint (val_loss={best_val:.6f}): {output_path}")

    summary = {
        "best_val_loss": float(best_val),
        "history": history,
        "checkpoint": str(output_path),
    }
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved training history: {history_path}")


if __name__ == "__main__":
    main()
