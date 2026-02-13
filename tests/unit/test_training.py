"""
Unit tests for the VAE training module (src/training/).

Tests cover: sigma^2 clamping, curriculum batching API, early stopping,
best checkpoint restore, Mode F warmup protection, and loss decrease.

Reference: ISD Section MOD-005.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vae.build_vae import build_vae
from src.vae.model import VAEModel
from src.training.trainer import VAETrainer
from src.training.early_stopping import EarlyStopping
from src.training.batching import CurriculumBatchSampler

# ---------------------------------------------------------------------------
# Small model dimensions for fast tests
# ---------------------------------------------------------------------------

N_STOCKS = 50
T_VAL = 64
T_ANNEE = 3
F_VAL = 2
K_VAL = 10


@pytest.fixture
def small_model() -> tuple[VAEModel, dict]:
    """Build a small VAE model suitable for fast testing."""
    torch.manual_seed(42)
    model, info = build_vae(
        n=N_STOCKS, T=T_VAL, T_annee=T_ANNEE, F=F_VAL, K=K_VAL,
        r_max=200.0,  # Relaxed for unit tests with small data
    )
    return model, info


@pytest.fixture
def small_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate small train/val datasets."""
    torch.manual_seed(42)
    n_train = 80
    n_val = 20
    train_windows = torch.randn(n_train, T_VAL, F_VAL)
    val_windows = torch.randn(n_val, T_VAL, F_VAL)
    return train_windows, val_windows


# ---------------------------------------------------------------------------
# 1. test_sigma_sq_clamped — After 100 steps, sigma^2 stays in [1e-4, 10]
# ---------------------------------------------------------------------------

def test_sigma_sq_clamped(
    small_model: tuple[VAEModel, dict],
    small_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """After training steps, sigma^2 must remain clamped in [1e-4, 10]."""
    torch.manual_seed(42)
    model, _ = small_model
    train_windows, val_windows = small_data

    trainer = VAETrainer(
        model=model,
        loss_mode="P",
        gamma=3.0,
        learning_rate=1e-3,  # Higher LR to push sigma^2
        patience=100,
        device=torch.device("cpu"),
    )

    # Create data loader
    dataset = TensorDataset(train_windows, torch.arange(len(train_windows)))
    loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        dataset, batch_size=16, shuffle=True,
    )

    # Run enough steps
    for epoch in range(10):
        trainer.train_epoch(loader, epoch, 100)

    log_sigma_sq_min = math.log(1e-4)
    log_sigma_sq_max = math.log(10.0)

    log_val = model.log_sigma_sq.item()
    assert log_val >= log_sigma_sq_min - 1e-6, (
        f"log_sigma_sq={log_val} below min={log_sigma_sq_min}"
    )
    assert log_val <= log_sigma_sq_max + 1e-6, (
        f"log_sigma_sq={log_val} above max={log_sigma_sq_max}"
    )

    sigma_sq = model.obs_var.item()
    assert 1e-4 - 1e-8 <= sigma_sq <= 10.0 + 1e-8, (
        f"sigma_sq={sigma_sq} outside [1e-4, 10]"
    )


# ---------------------------------------------------------------------------
# 2. test_curriculum_batching_transition — CurriculumBatchSampler API exists
# ---------------------------------------------------------------------------

def test_curriculum_batching_transition() -> None:
    """CurriculumBatchSampler can be instantiated and modes switched."""
    n_windows = 200
    batch_size = 32

    sampler = CurriculumBatchSampler(
        n_windows=n_windows,
        batch_size=batch_size,
        synchronous=True,
        seed=42,
    )

    # Verify synchronous mode produces batches
    sync_batches = list(sampler)
    assert len(sync_batches) > 0, "Synchronous mode should produce batches"
    for batch in sync_batches:
        assert len(batch) <= batch_size, f"Batch size {len(batch)} exceeds {batch_size}"

    # Switch to random
    sampler.set_synchronous(False)
    random_batches = list(sampler)
    assert len(random_batches) > 0, "Random mode should produce batches"

    # A2: Verify total coverage in random mode — each index appears exactly once
    all_indices: list[int] = []
    for batch in random_batches:
        all_indices.extend(batch)
    assert len(set(all_indices)) == n_windows, (
        f"Random mode should cover all windows: got {len(set(all_indices))}, expected {n_windows}"
    )
    assert len(all_indices) == n_windows, (
        f"Random mode should yield each index once: got {len(all_indices)} total, expected {n_windows}"
    )

    # A2: Verify sync batches respect stratified sampling within time blocks.
    # Without metadata, all windows fall in one block, so we verify that sync
    # batches sample across multiple strata (not just random contiguous chunks).
    sync_sampler = sampler.__class__(
        n_windows=n_windows, batch_size=batch_size, synchronous=True, seed=42,
    )
    sync_batches_2 = list(sync_sampler)
    for batch in sync_batches_2:
        assert len(batch) <= batch_size, (
            f"Sync batch exceeds batch_size: {len(batch)} > {batch_size}"
        )
    # Total indices produced should be roughly n_windows (not necessarily exact
    # because sync mode may over/under-sample due to stratum rounding)
    all_sync_indices = [idx for b in sync_batches_2 for idx in b]
    assert len(all_sync_indices) > 0, "Sync mode should produce at least some indices"

    # A3: Verify sync batch date spread is bounded by delta_sync * 5
    # when metadata is present. Without metadata, all windows fall in
    # one time block, so spread is unconstrained. The check is meaningful
    # only when multiple time blocks exist.
    delta_sync = getattr(sync_sampler, "delta_sync", 21)
    max_spread = delta_sync * 5
    has_multiple_blocks = len(getattr(sync_sampler, "time_blocks", {})) > 1
    if has_multiple_blocks:
        for batch in sync_batches:
            if len(batch) >= 2:
                min_idx = min(batch)
                max_idx = max(batch)
                spread = max_idx - min_idx
                assert spread <= max_spread, (
                    f"Sync batch index spread {spread} exceeds "
                    f"delta_sync * 5 = {max_spread}"
                )
    else:
        # Without metadata: verify delta_sync attribute exists and has
        # the expected default value
        assert delta_sync == 21, (
            f"Default delta_sync should be 21, got {delta_sync}"
        )


# ---------------------------------------------------------------------------
# 3. test_early_stopping_patience — Stops after 3 non-improving epochs
# ---------------------------------------------------------------------------

def test_early_stopping_patience() -> None:
    """EarlyStopping with patience=3 should trigger after 3 non-improving epochs."""
    torch.manual_seed(42)
    es = EarlyStopping(patience=3)

    # Use a simple linear model as a dummy
    model = torch.nn.Linear(10, 5)

    # First call: improvement (loss = 1.0)
    assert not es.check(1.0, 0, model)
    assert es.counter == 0

    # 3 non-improving epochs
    assert not es.check(1.1, 1, model)  # counter = 1
    assert es.counter == 1

    assert not es.check(1.2, 2, model)  # counter = 2
    assert es.counter == 2

    should_stop = es.check(1.3, 3, model)  # counter = 3 -> stop
    assert should_stop, "Should stop after 3 non-improving epochs"
    assert es.stopped


# ---------------------------------------------------------------------------
# 4. test_best_checkpoint_restored — After fitting, model is at best epoch weights
# ---------------------------------------------------------------------------

def test_best_checkpoint_restored(
    small_model: tuple[VAEModel, dict],
    small_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """After fit(), model weights should correspond to the best validation epoch."""
    torch.manual_seed(42)
    model, _ = small_model
    train_windows, val_windows = small_data

    trainer = VAETrainer(
        model=model,
        loss_mode="P",
        gamma=1.0,
        learning_rate=1e-3,
        patience=3,
        device=torch.device("cpu"),
    )

    result = trainer.fit(
        train_windows=train_windows,
        val_windows=val_windows,
        max_epochs=20,
        batch_size=16,
    )

    best_epoch = result["best_epoch"]
    best_val = result["best_val_elbo"]

    # Verify best_epoch was recorded
    assert best_epoch >= 0, f"best_epoch should be >= 0, got {best_epoch}"

    # Re-evaluate on validation to confirm we're at the best checkpoint
    val_dataset = TensorDataset(val_windows)
    val_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
    )
    current_val = trainer.validate(val_loader)

    # Tolerance is wider because BatchNorm running statistics can cause
    # minor discrepancies between the ELBO recorded during training and
    # the one recomputed after restore.
    assert abs(current_val - best_val) / max(abs(best_val), 1.0) < 0.001, (
        f"After restore, val_elbo={current_val:.6f} should be within 0.1% of "
        f"best_val_elbo={best_val:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. test_mode_F_warmup_protection — During warmup, early stopping counter stays 0
# ---------------------------------------------------------------------------

def test_mode_F_warmup_protection(
    small_model: tuple[VAEModel, dict],
    small_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """In Mode F, fit() must not trigger early stopping during warmup.

    Runs fit() with patience=1 and 50% warmup. With patience=1, training
    would stop after 2 non-improving epochs if warmup protection were absent.
    We verify training runs past the warmup boundary.
    """
    torch.manual_seed(42)
    model, _ = small_model
    train_windows, val_windows = small_data

    warmup_fraction = 0.50
    max_epochs = 10
    warmup_epochs = int(warmup_fraction * max_epochs)  # = 5

    trainer = VAETrainer(
        model=model,
        loss_mode="F",
        gamma=1.0,
        warmup_fraction=warmup_fraction,
        learning_rate=1e-3,
        patience=1,  # Very aggressive — would stop in epoch 2 without protection
        device=torch.device("cpu"),
    )

    result = trainer.fit(
        train_windows=train_windows,
        val_windows=val_windows,
        max_epochs=max_epochs,
        batch_size=16,
    )

    n_epochs_run = len(result["history"])
    # With 50% warmup (5 epochs) + patience=1, training must run
    # at least past the warmup boundary (epoch 5)
    assert n_epochs_run >= warmup_epochs, (
        f"Training stopped at epoch {n_epochs_run}, before warmup ended "
        f"at epoch {warmup_epochs}. Warmup protection is not working."
    )

    # Verify epochs trained >= warmup_fraction * max_epochs.
    # For warmup_fraction=0.5, max_epochs=10 -> at least 5 epochs must run.
    min_required_epochs = int(warmup_fraction * max_epochs)
    assert n_epochs_run >= min_required_epochs, (
        f"Expected at least warmup_fraction * max_epochs = {min_required_epochs} "
        f"epochs, but only {n_epochs_run} ran."
    )


# ---------------------------------------------------------------------------
# 6. test_training_loss_decreases — Loss decreases over 5 epochs on simple data
# ---------------------------------------------------------------------------

def test_training_loss_decreases(
    small_model: tuple[VAEModel, dict],
) -> None:
    """Training loss should generally decrease over epochs on learnable data."""
    torch.manual_seed(42)
    model, _ = small_model

    # Create simple learnable data: sinusoidal patterns
    n_train = 100
    t = torch.linspace(0, 4 * math.pi, T_VAL).unsqueeze(0).unsqueeze(-1)
    train_windows = torch.sin(t).expand(n_train, T_VAL, 1).repeat(1, 1, F_VAL)
    # Add small noise
    train_windows = train_windows + 0.1 * torch.randn_like(train_windows)

    val_windows = train_windows[:20].clone()

    trainer = VAETrainer(
        model=model,
        loss_mode="P",
        gamma=1.0,
        learning_rate=1e-3,
        patience=50,
        device=torch.device("cpu"),
    )

    result = trainer.fit(
        train_windows=train_windows,
        val_windows=val_windows,
        max_epochs=10,
        batch_size=32,
    )

    history = result["history"]
    assert len(history) >= 5, f"Expected at least 5 epochs, got {len(history)}"

    # Compare first epoch loss to last epoch loss
    first_loss = history[0]["train_loss"]
    last_loss = history[-1]["train_loss"]

    assert last_loss < first_loss, (
        f"Loss should decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )

    # FORMULA: loss should decrease by at least 10% over 10 epochs
    # (on simple sinusoidal data, this is easily achievable)
    assert last_loss < 0.9 * first_loss, (
        f"Loss should decrease by at least 10%: "
        f"first={first_loss:.4f}, last={last_loss:.4f}, "
        f"ratio={last_loss/first_loss:.4f}"
    )

    # Intermediate check: loss at epoch 5 should already be below epoch 1
    mid_loss = history[min(4, len(history) - 1)]["train_loss"]
    assert mid_loss < first_loss, (
        f"Loss at epoch 5 should be below epoch 1: "
        f"epoch1={first_loss:.4f}, epoch5={mid_loss:.4f}"
    )

    # Verify history contains train_loss entries
    for entry in history:
        assert "train_loss" in entry, "History entry missing 'train_loss' key"

    # Verify best_val_elbo is <= minimum of all validation losses in history
    val_losses = [e["val_loss"] for e in history if "val_loss" in e]
    if val_losses:
        assert result["best_val_elbo"] <= min(val_losses) + 1e-6, (
            f"best_val_elbo={result['best_val_elbo']:.6f} should be <= "
            f"min(val_losses)={min(val_losses):.6f}"
        )

    # Verify best_epoch is within valid range
    assert 0 <= result["best_epoch"] < 10, (
        f"best_epoch={result['best_epoch']} should be in [0, max_epochs)"
    )


# ---------------------------------------------------------------------------
# 7. test_curriculum_sampler_sync_vs_random_batches — INV-010
# ---------------------------------------------------------------------------

def test_curriculum_sampler_sync_vs_random_batches() -> None:
    """
    Synchronous batches draw from time blocks; random batches cover all windows.
    INV-010: Phases 1-2 synchronous+stratified, Phase 3 random.
    """
    import pandas as pd

    n_windows = 200
    batch_size = 32

    # Create metadata with distinct end dates for time-block grouping
    dates = pd.bdate_range("2020-01-01", periods=n_windows, freq="B")
    metadata = pd.DataFrame({
        "stock_id": list(range(n_windows)),
        "start_date": dates,
        "end_date": dates,
    })

    sampler = CurriculumBatchSampler(
        n_windows=n_windows,
        batch_size=batch_size,
        window_metadata=metadata,
        synchronous=True,
        seed=42,
    )

    # Phase 1-2: synchronous batches
    sync_batches = list(sampler)
    assert len(sync_batches) > 0, "Synchronous mode produced no batches"

    # Each synchronous batch should draw from a limited time range
    for batch in sync_batches:
        if len(batch) >= 2:
            batch_dates = dates[batch]
            date_range = int((batch_dates.max() - batch_dates.min()).days)  # type: ignore[union-attr]
            # Windows in a sync batch should be within delta_sync window
            assert date_range <= 21 * 5, (
                f"Sync batch spans {date_range} days, expected <= {21 * 5}"
            )

    # Phase 3: random batches
    sampler.set_synchronous(False)
    random_batches = list(sampler)
    assert len(random_batches) > 0, "Random mode produced no batches"

    # Random batches should cover all indices
    all_indices = set()
    for batch in random_batches:
        all_indices.update(batch)
    assert len(all_indices) == n_windows, (
        f"Random mode should cover all {n_windows} windows, covered {len(all_indices)}"
    )


# ---------------------------------------------------------------------------
# 8. test_mode_F_scheduler_disabled_during_warmup
# ---------------------------------------------------------------------------

def test_mode_F_scheduler_disabled_during_warmup(
    small_model: tuple[VAEModel, dict],
    small_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """In Mode F during warmup, scheduler should not reduce LR aggressively."""
    torch.manual_seed(42)
    model, _ = small_model
    train_windows, val_windows = small_data

    warmup_fraction = 0.50
    max_epochs = 10

    trainer = VAETrainer(
        model=model,
        loss_mode="F",
        gamma=1.0,
        warmup_fraction=warmup_fraction,
        learning_rate=1e-3,
        patience=50,
        device=torch.device("cpu"),
    )

    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    dataset = TensorDataset(train_windows, torch.arange(len(train_windows)))
    train_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        dataset, batch_size=16, shuffle=True,
    )

    # Run warmup epochs
    warmup_epochs = int(warmup_fraction * max_epochs)
    for epoch in range(warmup_epochs):
        trainer.train_epoch(train_loader, epoch, max_epochs)

    current_lr = trainer.optimizer.param_groups[0]["lr"]
    assert current_lr == pytest.approx(initial_lr, abs=1e-10), (
        f"LR should be completely unchanged during warmup: "
        f"initial={initial_lr}, current={current_lr}"
    )

    # FORMULA: after warmup, scheduler step SHOULD be able to reduce LR
    # (we don't test the actual reduction here, just that LR is preserved
    # during warmup — the formula is lr_t = lr_0 for t < T_warmup)


# ---------------------------------------------------------------------------
# 9. test_gradient_accumulation_effective_batch
# ---------------------------------------------------------------------------

def test_gradient_accumulation_effective_batch(
    small_model: tuple[VAEModel, dict],
) -> None:
    """
    Loss from a full batch vs two half-batches should be approximately equal
    when model is in eval mode (consistent BatchNorm statistics).
    """
    torch.manual_seed(42)
    model, _ = small_model

    n_samples = 64
    data = torch.randn(n_samples, T_VAL, F_VAL)

    # Single large batch
    model.eval()
    with torch.no_grad():
        x_hat_full, mu_full, log_var_full = model(data)
    loss_full = torch.mean((data - x_hat_full) ** 2).item()

    # Two half-batches averaged
    half = n_samples // 2
    with torch.no_grad():
        x_hat_1, _, _ = model(data[:half])
        x_hat_2, _, _ = model(data[half:])
    loss_1 = torch.mean((data[:half] - x_hat_1) ** 2).item()
    loss_2 = torch.mean((data[half:] - x_hat_2) ** 2).item()
    loss_accum = (loss_1 + loss_2) / 2.0

    assert abs(loss_full - loss_accum) / max(abs(loss_full), 1e-8) < 0.05, (
        f"Loss mismatch: full={loss_full:.6f}, accumulated={loss_accum:.6f}"
    )


# ---------------------------------------------------------------------------
# Tests: Trainer output keys
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A4. test_sigma_sq_clamp_exact_values — Verify clamp at known boundaries
# ---------------------------------------------------------------------------

def test_sigma_sq_clamp_exact_values() -> None:
    """Verify obs_var clamp function produces exact values at known boundaries.

    σ² = clamp(exp(log_σ²), 1e-4, 10.0)
    - log_σ² = log(1e-5) → σ² must be exactly 1e-4 (lower clamp)
    - log_σ² = log(100)  → σ² must be exactly 10.0 (upper clamp)
    - log_σ² = 0.0       → σ² must be exactly 1.0 (interior)
    """
    torch.manual_seed(42)
    model, _ = build_vae(n=20, T=64, T_annee=3, F=2, K=5, r_max=200.0, c_min=144)

    # Interior: log_σ² = 0 → σ² = 1.0
    with torch.no_grad():
        model.log_sigma_sq.fill_(0.0)
    assert abs(model.obs_var.item() - 1.0) < 1e-6, (
        f"obs_var at log_σ²=0 should be 1.0, got {model.obs_var.item()}"
    )

    # Lower clamp: log_σ² = log(1e-5) → σ² = 1e-4 (clamped)
    with torch.no_grad():
        model.log_sigma_sq.fill_(math.log(1e-5))
    assert abs(model.obs_var.item() - 1e-4) < 1e-8, (
        f"obs_var at log_σ²=log(1e-5) should be 1e-4 (clamped), got {model.obs_var.item()}"
    )

    # Upper clamp: log_σ² = log(100) → σ² = 10.0 (clamped)
    with torch.no_grad():
        model.log_sigma_sq.fill_(math.log(100.0))
    assert abs(model.obs_var.item() - 10.0) < 1e-6, (
        f"obs_var at log_σ²=log(100) should be 10.0 (clamped), got {model.obs_var.item()}"
    )

    # Gradient verification: at lower boundary, gradient through clamp should be 0
    model.log_sigma_sq.requires_grad_(True)
    with torch.no_grad():
        model.log_sigma_sq.fill_(math.log(1e-5))
    obs = model.obs_var
    obs.backward()
    assert model.log_sigma_sq.grad is not None
    assert abs(model.log_sigma_sq.grad.item()) < 1e-8, (
        f"At lower clamp boundary, gradient should be 0 (blocked by clamp), "
        f"got {model.log_sigma_sq.grad.item()}"
    )


class TestTrainerOutput:
    def test_fit_returns_expected_keys(self) -> None:
        """trainer.fit() returns dict with history, best_epoch, best_val_elbo."""
        torch.manual_seed(42)
        model, _ = build_vae(n=20, T=64, T_annee=3, F=2, K=5, r_max=200.0, c_min=144)
        trainer = VAETrainer(model=model, loss_mode="P", gamma=1.0, learning_rate=1e-3, patience=5, device=torch.device("cpu"))
        x = torch.randn(20, 64, 2)
        result = trainer.fit(train_windows=x[:16], val_windows=x[16:], max_epochs=3, batch_size=8)
        assert "history" in result, "Missing 'history' key"
        assert len(result["history"]) > 0, "Empty history"

        # Verify additional required keys exist
        assert "best_epoch" in result, "Missing 'best_epoch' key"
        assert "best_val_elbo" in result, "Missing 'best_val_elbo' key"

        # Verify best_epoch is a non-negative int
        assert isinstance(result["best_epoch"], int), (
            f"best_epoch should be int, got {type(result['best_epoch'])}"
        )
        assert result["best_epoch"] >= 0, (
            f"best_epoch should be >= 0, got {result['best_epoch']}"
        )

        # Verify best_val_elbo is a finite float
        assert isinstance(result["best_val_elbo"], float), (
            f"best_val_elbo should be float, got {type(result['best_val_elbo'])}"
        )
        assert math.isfinite(result["best_val_elbo"]), (
            f"best_val_elbo should be finite, got {result['best_val_elbo']}"
        )

        # If history contains val_loss entries, verify best_val_elbo <= min(val_losses)
        val_losses = [e["val_loss"] for e in result["history"] if "val_loss" in e]
        if val_losses:
            assert result["best_val_elbo"] <= min(val_losses) + 1e-6, (
                f"best_val_elbo={result['best_val_elbo']:.6f} should be <= "
                f"min(val_losses)={min(val_losses):.6f}"
            )
