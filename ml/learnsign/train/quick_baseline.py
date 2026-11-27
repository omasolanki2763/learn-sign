"""Quick baseline 1D-CNN model architecture, resampling, and training pipeline."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from learnsign.features.schema import FEATURES_PER_FRAME
from learnsign.train.monitor import EpochMonitor

DEFAULT_TARGET_FRAMES: int = 64
MAX_MODEL_PARAMS: int = 500_000
DEFAULT_NUM_CLASSES: int = 250


def resample_sequence(seq: np.ndarray, target_frames: int = DEFAULT_TARGET_FRAMES) -> np.ndarray:
    """Resample variable-length sequence (T, FEATURES_PER_FRAME) to fixed target_frames.

    Applies 1D linear interpolation across the time axis for each feature dimension.

    Args:
        seq: Array of shape (T, FEATURES_PER_FRAME).
        target_frames: Fixed frame count for neural network input (default 64).

    Returns:
        float32 array of shape (target_frames, FEATURES_PER_FRAME).
    """
    t, num_features = seq.shape
    if t == target_frames:
        return seq.astype(np.float32)

    if t == 0:
        return np.zeros((target_frames, num_features), dtype=np.float32)

    if t == 1:
        return np.repeat(seq.astype(np.float32), target_frames, axis=0)

    orig_times = np.linspace(0.0, 1.0, t)
    target_times = np.linspace(0.0, 1.0, target_frames)

    resampled = np.empty((target_frames, num_features), dtype=np.float32)
    for c in range(num_features):
        col = seq[:, c]
        # Linear interpolation preserving NaNs when entire column is NaN
        if np.all(np.isnan(col)):
            resampled[:, c] = np.nan
        else:
            # Handle sporadic NaNs by forward/backward fill before interpolation
            valid_idx = ~np.isnan(col)
            if np.sum(valid_idx) == 1:
                resampled[:, c] = col[valid_idx][0]
            else:
                resampled[:, c] = np.interp(target_times, orig_times[valid_idx], col[valid_idx])

    return resampled


class QuickBaseline1DCNN(nn.Module):
    """Compact 1D-CNN for sign classification (strictly <= 500k parameters).

    Architecture:
    - 4 Conv1d blocks with BatchNorm1d and ReLU activations.
    - Adaptive average pooling across time.
    - Linear classification head.
    """

    def __init__(
        self,
        in_features: int = FEATURES_PER_FRAME,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        """Initialize 1D-CNN layers.

        Args:
            in_features: Feature dimension per frame (defaults to FEATURES_PER_FRAME).
            num_classes: Number of ASL gloss classes (250 for GISLR).
        """
        super().__init__()
        self.in_features = in_features

        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv1d(in_features, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv1d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Missing landmarks (undetected hand/pose or dropped frames) arrive as NaN from
        `normalize_frame` and `resample_sequence`. Because `Conv1d`/`BatchNorm1d` cannot
        propagate NaN, remaining NaN values are filled with 0.0 here at model input.

        Rationale vs explicit presence mask channels: in torso-normalized coordinates,
        (0.0, 0.0) coincides with the shoulder midpoint, so zero-fill alone cannot
        distinguish an undetected hand from one resting at torso center. We keep the
        frozen v1 feature contract (`FEATURES_PER_FRAME`, without per-part presence
        mask channels) for Step 0 parity so existing parquets and MediaPipe Tasks
        compare on identical inputs; explicit per-part presence channels are deferred
        to feature contract v2, and Slice 1 browser inference matches this convention.

        Args:
            x: Tensor of shape (Batch, in_features, T) or (Batch, T, in_features).

        Returns:
            Logits of shape (Batch, num_classes).
        """
        # If input is (Batch, Frames, Channels), transpose to (Batch, Channels, Frames)
        if x.shape[1] != self.in_features and x.shape[2] == self.in_features:
            x = x.transpose(1, 2)

        # Replace any missing landmark NaNs with 0.0 at the v1 model boundary
        x = torch.nan_to_num(x, nan=0.0)

        feats = self.backbone(x)  # (Batch, 192, 1)
        feats = feats.flatten(1)  # (Batch, 192)
        logits = self.head(feats)  # (Batch, num_classes)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LandmarkSequenceDataset(Dataset):
    """PyTorch Dataset for pre-processed landmark sequences."""

    def __init__(self, sequences: list[np.ndarray], labels: list[int]) -> None:
        """Initialize dataset.

        Args:
            sequences: List of (T, 144) landmark arrays.
            labels: List of integer class targets.
        """
        self.sequences = [resample_sequence(seq) for seq in sequences]
        self.labels = labels

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a single resampled sequence and label."""
        seq = torch.from_numpy(self.sequences[idx]).float()
        label = self.labels[idx]
        return seq, label


def overfit_one_batch(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    threshold: float = 0.1,
    max_steps: int = 150,
) -> float:
    """Overfit a single mini-batch as a sanity check before full training.

    Ensures the model architecture, gradients, and loss functions can memorize
    a small batch. Aborts if loss does not drop below threshold.

    Args:
        model: PyTorch model.
        batch_x: Input tensor of shape (B, C, T) or (B, T, C).
        batch_y: Label tensor of shape (B,).
        criterion: Loss function.
        optimizer: PyTorch optimizer.
        threshold: Required loss target to pass the check (default 0.1).
        max_steps: Maximum training steps attempted.

    Returns:
        Final loss value achieved.

    Raises:
        RuntimeError: If the model fails to reach loss < threshold within max_steps.
    """
    model.train()
    final_loss = float("inf")

    for _step in range(max_steps):
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if final_loss < threshold:
            return final_loss

    raise RuntimeError(
        f"Overfit-one-batch check failed: final loss {final_loss:.4f} >= threshold {threshold}"
    )


def compute_topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> list[float]:
    """Compute Top-K accuracies for predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        if batch_size == 0:
            return [0.0] * len(topk)

        _, pred = logits.topk(min(maxk, logits.size(1)), 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k_clamped = min(k, logits.size(1))
            correct_k = correct[:k_clamped].reshape(-1).float().sum(0, keepdim=True)
            res.append(float((correct_k / batch_size).item()))
        return res


def train_quick_baseline(
    model: QuickBaseline1DCNN,
    train_loader: Any,
    val_loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 15,
    log_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train quick baseline 1D-CNN with TensorBoard logging and epoch monitoring.

    Args:
        model: 1D-CNN model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: PyTorch optimizer.
        epochs: Maximum number of epochs (<= 15).
        log_dir: TensorBoard log directory.
        device: 'cpu' or 'cuda'.

    Returns:
        Dictionary with training history and final evaluation metrics.
    """
    device_obj = torch.device(device)
    model.to(device_obj)

    writer = SummaryWriter(log_dir=str(log_dir)) if log_dir else None
    monitor = EpochMonitor(patience=5)

    history: list[dict[str, Any]] = []
    global_step = 0

    # Overfit one batch check runs first on the first training batch
    first_x, first_y = next(iter(train_loader))
    first_x, first_y = first_x.to(device_obj), first_y.to(device_obj)
    overfit_one_batch(model, first_x, first_y, criterion, optimizer, threshold=0.1)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_steps = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device_obj), batch_y.to(device_obj)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Compute gradient norm for logging
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item()
            )
            optimizer.step()

            step_loss = loss.item()
            running_train_loss += step_loss
            train_steps += 1
            global_step += 1

            if writer:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", step_loss, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)

        avg_train_loss = running_train_loss / max(1, train_steps)

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device_obj), batch_y.to(device_obj)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_total += loss.item()
                val_steps += 1
                all_logits.append(logits)
                all_targets.append(batch_y)

        avg_val_loss = val_loss_total / max(1, val_steps)
        cat_logits = (
            torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, DEFAULT_NUM_CLASSES))
        )
        cat_targets = (
            torch.cat(all_targets, dim=0) if all_targets else torch.empty((0,), dtype=torch.long)
        )
        top1, top5 = compute_topk_accuracy(cat_logits, cat_targets, topk=(1, 5))

        if writer:
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            writer.add_scalar("val/top1", top1, epoch)
            writer.add_scalar("val/top5", top5, epoch)

        diagnostic_status = monitor.update(epoch, avg_train_loss, avg_val_loss, top1)
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_top1": top1,
            "val_top5": top5,
            "status": diagnostic_status,
        }
        history.append(epoch_record)

        if monitor.should_stop():
            break

    if writer:
        writer.close()

    return {
        "epochs_trained": len(history),
        "history": history,
        "final_top1": history[-1]["val_top1"] if history else 0.0,
        "final_top5": history[-1]["val_top5"] if history else 0.0,
    }
