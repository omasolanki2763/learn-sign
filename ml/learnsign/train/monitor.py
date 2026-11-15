"""Training monitor for diagnostic flags and early stopping."""


# Default underfitting loss threshold: for num_classes=250, initial random guess loss
# is ln(250) ≈ 5.52. A loss remaining flat above 2.0 indicates severe underfitting.
DEFAULT_UNDERFITTING_LOSS_THRESHOLD: float = 2.0


class EpochMonitor:
    """Tracks epoch-level metrics and identifies training diagnostics.

    Detects:
    - Overfitting: train loss falling consistently while validation loss rises.
    - Stagnation: validation top-1 accuracy failing to improve across patience epochs.
    - Underfitting: training loss failing to decrease meaningfully.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        """Initialize monitor with patience threshold.

        Args:
            patience: Number of consecutive epochs before raising diagnostic stop flags.
            min_delta: Minimum metric improvement threshold to qualify as progress.
        """
        self.patience = patience
        self.min_delta = min_delta

        self.history: list[dict[str, float]] = []
        self.best_val_top1 = -1.0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.overfitting_streak = 0
        self.stagnation_streak = 0
        self.stop_requested = False

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_top1: float,
    ) -> str:
        """Record epoch metrics and determine diagnostic status.

        Args:
            epoch: Current training epoch index.
            train_loss: Training loss for current epoch.
            val_loss: Validation loss for current epoch.
            val_top1: Validation Top-1 accuracy for current epoch.

        Returns:
            Status string: 'ok', 'overfitting', 'stagnation', or 'underfitting'.
        """
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_top1": val_top1,
        }
        self.history.append(record)

        # Check for improvement
        if val_top1 > self.best_val_top1 + self.min_delta:
            self.best_val_top1 = val_top1
            self.epochs_without_improvement = 0
            self.stagnation_streak = 0
        else:
            self.epochs_without_improvement += 1
            self.stagnation_streak += 1

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss

        # Overfitting check: train loss falls while val loss rises compared to previous epoch
        if len(self.history) >= 2:
            prev = self.history[-2]
            train_loss_fell = train_loss < prev["train_loss"]
            val_loss_rose = val_loss > prev["val_loss"]
            if train_loss_fell and val_loss_rose:
                self.overfitting_streak += 1
            else:
                self.overfitting_streak = 0
        else:
            self.overfitting_streak = 0

        # Diagnostics evaluation
        if self.overfitting_streak >= self.patience:
            self.stop_requested = True
            return "overfitting"

        if self.stagnation_streak >= self.patience:
            self.stop_requested = True
            return "stagnation"

        if len(self.history) >= self.patience:
            recent_train_losses = [h["train_loss"] for h in self.history[-self.patience :]]
            train_loss_diff = recent_train_losses[0] - recent_train_losses[-1]
            if (
                train_loss_diff < self.min_delta
                and recent_train_losses[-1] > DEFAULT_UNDERFITTING_LOSS_THRESHOLD
            ):
                self.stop_requested = True
                return "underfitting"

        return "ok"

    def should_stop(self) -> bool:
        """Check if training should be halted early due to patience exhaustion."""
        return self.stop_requested
