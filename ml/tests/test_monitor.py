"""Unit tests for training progress monitoring, overfitting, and stagnation detection."""

from learnsign.train.monitor import EpochMonitor


def test_monitor_overfitting() -> None:
    """Step 1 test: Assert monitor returns 'overfitting'
    when train loss drops while val loss rises.
    """
    monitor = EpochMonitor(patience=5)

    # Epochs 1-4: normal drop
    status = monitor.update(epoch=1, train_loss=1.0, val_loss=1.0, val_top1=0.60)
    assert status == "ok"
    assert not monitor.should_stop()

    # 5 consecutive epochs of train loss falling and val loss rising
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
    val_losses = [1.1, 1.2, 1.3, 1.4, 1.5]
    val_top1s = [0.60, 0.59, 0.58, 0.57, 0.56]

    final_status = "ok"
    for ep, (tl, vl, vt) in enumerate(
        zip(train_losses, val_losses, val_top1s, strict=True), start=2
    ):
        final_status = monitor.update(epoch=ep, train_loss=tl, val_loss=vl, val_top1=vt)

    assert final_status == "overfitting"
    assert monitor.should_stop()


def test_monitor_stagnation() -> None:
    """Step 1 test: Assert monitor returns 'stagnation'
    when validation top-1 is flat across patience epochs.
    """
    monitor = EpochMonitor(patience=5)

    # Initial epoch
    monitor.update(epoch=1, train_loss=1.0, val_loss=1.0, val_top1=0.50)

    # 5 epochs where val_top1 does not improve (flat 0.50)
    statuses = []
    for ep in range(2, 7):
        status = monitor.update(epoch=ep, train_loss=0.9 - (ep * 0.05), val_loss=1.0, val_top1=0.50)
        statuses.append(status)

    assert "stagnation" in statuses
    assert monitor.should_stop()


def test_monitor_healthy_training() -> None:
    """Assert monitor returns 'ok' when train and val loss decrease and accuracy improves."""
    monitor = EpochMonitor(patience=5)

    for ep in range(1, 6):
        status = monitor.update(
            epoch=ep,
            train_loss=1.0 - (ep * 0.1),
            val_loss=1.0 - (ep * 0.08),
            val_top1=0.50 + (ep * 0.05),
        )
        assert status == "ok"
        assert not monitor.should_stop()
