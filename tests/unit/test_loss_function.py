"""
Unit tests for the VAE loss computation module (src/vae/loss.py).

Tests cover: crisis weighting, 3 loss modes (P/F/A), co-movement loss,
curriculum scheduling, validation ELBO, and numerical stability.

Reference: ISD Section MOD-004.
"""

import math

import pytest
import torch

from src.vae.loss import (
    compute_co_movement_loss,
    compute_loss,
    compute_reconstruction_loss,
    compute_validation_elbo,
    get_beta_t,
    get_lambda_co,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 4
T = 64
F = 2
K = 10
TOTAL_EPOCHS = 100


@pytest.fixture
def tensors() -> dict[str, torch.Tensor]:
    """Create small deterministic tensors for testing."""
    torch.manual_seed(42)
    x = torch.randn(B, T, F)
    x_hat = torch.randn(B, T, F)
    mu = torch.randn(B, K)
    log_var = torch.randn(B, K) * 0.5
    log_sigma_sq = torch.tensor(0.0)  # sigma^2 = 1.0
    crisis_fractions = torch.tensor([0.0, 0.3, 0.7, 1.0])
    return {
        "x": x,
        "x_hat": x_hat,
        "mu": mu,
        "log_var": log_var,
        "log_sigma_sq": log_sigma_sq,
        "crisis_fractions": crisis_fractions,
    }


# ---------------------------------------------------------------------------
# 1. test_D_factor_present — Mode P reconstruction coefficient includes T*F/(2*sigma^2)
# ---------------------------------------------------------------------------

def test_D_factor_present(tensors: dict[str, torch.Tensor]) -> None:
    """Mode P recon term should include the D/(2*sigma^2) factor."""
    torch.manual_seed(42)
    log_sigma_sq = torch.tensor(0.0, requires_grad=True)

    loss, components = compute_loss(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=log_sigma_sq,
        crisis_fractions=tensors["crisis_fractions"],
        epoch=50,
        total_epochs=TOTAL_EPOCHS,
        mode="P",
        gamma=3.0,
    )

    D = T * F
    sigma_sq = torch.clamp(torch.exp(log_sigma_sq), min=1e-4, max=10.0)
    expected_coeff = D / (2.0 * sigma_sq.item())

    # recon_term = D/(2*sigma^2) * L_recon
    # So recon_term / L_recon should equal D/(2*sigma^2)
    ratio = components["recon_term"] / max(components["recon"], 1e-12)
    assert abs(ratio - expected_coeff) < 1e-3, (
        f"D/(2*sigma^2) factor not present: ratio={ratio:.4f}, expected={expected_coeff:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. test_mode_P_gradients — sigma^2 receives gradient, beta effectively 1
# ---------------------------------------------------------------------------

def test_mode_P_gradients(tensors: dict[str, torch.Tensor]) -> None:
    """In Mode P, log_sigma_sq should receive gradient (learned), and beta=1."""
    torch.manual_seed(42)
    log_sigma_sq = torch.tensor(0.0, requires_grad=True)

    loss, components = compute_loss(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=log_sigma_sq,
        crisis_fractions=tensors["crisis_fractions"],
        epoch=50,
        total_epochs=TOTAL_EPOCHS,
        mode="P",
        gamma=3.0,
    )

    loss.backward()
    assert log_sigma_sq.grad is not None, "Mode P: log_sigma_sq should receive gradient"
    assert log_sigma_sq.grad.abs().item() > 0, "Mode P: gradient should be non-zero"


# ---------------------------------------------------------------------------
# 3. test_mode_F_sigma_frozen — In Mode F, sigma^2 has no gradient
# ---------------------------------------------------------------------------

def test_mode_F_sigma_frozen(tensors: dict[str, torch.Tensor]) -> None:
    """In Mode F, log_sigma_sq should NOT receive gradient (frozen sigma^2=1)."""
    torch.manual_seed(42)
    log_sigma_sq = torch.tensor(0.0, requires_grad=True)

    # mu needs requires_grad so that the computation graph is non-trivial
    # and backward() can propagate through the KL term
    mu = tensors["mu"].clone().requires_grad_(True)

    loss, components = compute_loss(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=mu,
        log_var=tensors["log_var"],
        log_sigma_sq=log_sigma_sq,
        crisis_fractions=tensors["crisis_fractions"],
        epoch=50,
        total_epochs=TOTAL_EPOCHS,
        mode="F",
        gamma=3.0,
        beta_fixed=1.0,
    )

    loss.backward()
    # Mode F does not include sigma_sq in the loss formula, so
    # the gradient should either be None or zero
    if log_sigma_sq.grad is not None:
        assert log_sigma_sq.grad.abs().item() < 1e-12, (
            f"Mode F: sigma^2 should have no gradient, got {log_sigma_sq.grad.item()}"
        )


# ---------------------------------------------------------------------------
# 4. test_mode_F_beta_annealing — get_beta_t correctly anneals from 0 to 1
# ---------------------------------------------------------------------------

def test_mode_F_beta_annealing() -> None:
    """get_beta_t should anneal linearly from 0 to 1 over warmup epochs."""
    warmup_fraction = 0.20
    total_epochs = 100
    T_warmup = int(warmup_fraction * total_epochs)  # = 20

    # At epoch 0: beta_t = beta_min = 0.01 (floor prevents KL collapse)
    assert get_beta_t(0, total_epochs, warmup_fraction) == 0.01

    # At epoch 10 (midway): beta_t = 10/20 = 0.5
    assert abs(get_beta_t(10, total_epochs, warmup_fraction) - 0.5) < 1e-6

    # At epoch 20 (end of warmup): beta_t = 20/20 = 1.0
    assert get_beta_t(T_warmup, total_epochs, warmup_fraction) == 1.0

    # After warmup: beta_t stays at 1.0
    assert get_beta_t(50, total_epochs, warmup_fraction) == 1.0
    assert get_beta_t(99, total_epochs, warmup_fraction) == 1.0


# ---------------------------------------------------------------------------
# 5. test_mode_A_beta_applied — KL term multiplied by beta_fixed > 1
# ---------------------------------------------------------------------------

def test_mode_A_beta_applied(tensors: dict[str, torch.Tensor]) -> None:
    """Mode A with beta_fixed > 1 should scale the KL term."""
    torch.manual_seed(42)

    # Loss with beta=1 (baseline)
    _, comp_b1 = compute_loss(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=tensors["log_sigma_sq"],
        crisis_fractions=tensors["crisis_fractions"],
        epoch=50,
        total_epochs=TOTAL_EPOCHS,
        mode="A",
        gamma=3.0,
        beta_fixed=1.0,
    )

    # Loss with beta=4
    _, comp_b4 = compute_loss(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=tensors["log_sigma_sq"],
        crisis_fractions=tensors["crisis_fractions"],
        epoch=50,
        total_epochs=TOTAL_EPOCHS,
        mode="A",
        gamma=3.0,
        beta_fixed=4.0,
    )

    # The KL term itself (component) should be the same
    assert abs(comp_b1["kl"] - comp_b4["kl"]) < 1e-6, "KL component should be identical"

    # But total should differ by 3 * KL (beta=4 vs beta=1)
    expected_diff = 3.0 * comp_b1["kl"]
    actual_diff = comp_b4["total"] - comp_b1["total"]
    assert abs(actual_diff - expected_diff) < 1e-2, (
        f"Mode A beta scaling: expected diff={expected_diff:.4f}, got={actual_diff:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. test_modes_exclusive — Invalid mode raises error
# ---------------------------------------------------------------------------

def test_modes_exclusive(tensors: dict[str, torch.Tensor]) -> None:
    """Invalid mode string should raise AssertionError."""
    torch.manual_seed(42)
    with pytest.raises(AssertionError, match="INV-006"):
        compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=tensors["log_sigma_sq"],
            crisis_fractions=tensors["crisis_fractions"],
            epoch=0,
            total_epochs=TOTAL_EPOCHS,
            mode="INVALID",
        )


# ---------------------------------------------------------------------------
# 7. test_crisis_weight_gamma_1 — gamma=1 gives same loss for all windows
# ---------------------------------------------------------------------------

def test_crisis_weight_gamma_1() -> None:
    """With gamma=1, crisis_fractions should have no effect on the loss."""
    torch.manual_seed(42)
    x = torch.randn(B, T, F)
    x_hat = torch.randn(B, T, F)

    cf_zero = torch.zeros(B)
    cf_one = torch.ones(B)

    loss_zero = compute_reconstruction_loss(x, x_hat, cf_zero, gamma=1.0)
    loss_one = compute_reconstruction_loss(x, x_hat, cf_one, gamma=1.0)

    assert torch.allclose(loss_zero, loss_one, atol=1e-6), (
        f"gamma=1: loss should be identical regardless of crisis_fractions. "
        f"loss_zero={loss_zero.item():.6f}, loss_one={loss_one.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 8. test_crisis_weight_gamma_3 — Higher crisis fraction gives higher weight
# ---------------------------------------------------------------------------

def test_crisis_weight_gamma_3() -> None:
    """With gamma=3, f_c=1 should yield higher loss; f_c=0 no extra weight."""
    torch.manual_seed(42)
    x = torch.randn(B, T, F)
    x_hat = torch.randn(B, T, F)

    cf_zero = torch.zeros(B)
    cf_crisis = torch.ones(B)

    loss_no_crisis = compute_reconstruction_loss(x, x_hat, cf_zero, gamma=3.0)
    loss_full_crisis = compute_reconstruction_loss(x, x_hat, cf_crisis, gamma=3.0)

    # f_c=0 -> gamma_eff=1, f_c=1 -> gamma_eff=3
    # So full crisis loss should be 3x the no-crisis loss
    ratio = loss_full_crisis.item() / max(loss_no_crisis.item(), 1e-12)
    assert abs(ratio - 3.0) < 1e-4, (
        f"Expected 3x ratio with gamma=3 and f_c=1, got {ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# 9. test_curriculum_phases — get_lambda_co correct at each phase boundary
# ---------------------------------------------------------------------------

def test_curriculum_phases() -> None:
    """lambda_co curriculum scheduling: max at phase1, linear decay, zero at phase3."""
    lambda_co_max = 0.5
    total = 100

    # Phase 1: epochs 0..29 -> lambda_co_max
    assert get_lambda_co(0, total, lambda_co_max) == lambda_co_max
    assert get_lambda_co(15, total, lambda_co_max) == lambda_co_max
    assert get_lambda_co(29, total, lambda_co_max) == lambda_co_max

    # Phase 1 boundary: epoch 30 is the start of phase 2 (30% of 100 = 30)
    phase1_end = int(0.30 * total)  # = 30
    phase2_end = int(0.60 * total)  # = 60

    # At phase2 start: progress = 0/(60-30) = 0 -> lambda_co_max
    val_at_30 = get_lambda_co(phase1_end, total, lambda_co_max)
    assert abs(val_at_30 - lambda_co_max) < 1e-6

    # At phase2 midpoint: progress = 15/30 = 0.5 -> 0.5 * lambda_co_max
    val_mid = get_lambda_co(45, total, lambda_co_max)
    assert abs(val_mid - 0.5 * lambda_co_max) < 1e-6

    # Phase 3: epoch >= 60 -> 0
    assert get_lambda_co(60, total, lambda_co_max) == 0.0
    assert get_lambda_co(99, total, lambda_co_max) == 0.0


# ---------------------------------------------------------------------------
# 10. test_validation_elbo_excludes_gamma — same result for gamma=1 and gamma=3
# ---------------------------------------------------------------------------

def test_validation_elbo_excludes_gamma(tensors: dict[str, torch.Tensor]) -> None:
    """Validation ELBO (INV-011) excludes gamma, so gamma value should not matter."""
    torch.manual_seed(42)

    elbo = compute_validation_elbo(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=tensors["log_sigma_sq"],
    )

    # compute_validation_elbo does not take a gamma parameter.
    # It always uses unweighted MSE (gamma=1 equivalent).
    # Call it a second time to confirm determinism.
    elbo2 = compute_validation_elbo(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=tensors["log_sigma_sq"],
    )

    assert torch.allclose(elbo, elbo2, atol=1e-6), (
        "Validation ELBO should be deterministic and gamma-independent"
    )

    # Also verify validation ELBO differs from training loss with gamma=3
    _, train_comp = compute_loss(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=tensors["log_sigma_sq"],
        crisis_fractions=tensors["crisis_fractions"],
        epoch=50,
        total_epochs=TOTAL_EPOCHS,
        mode="P",
        gamma=3.0,
    )
    # Training loss with gamma=3 and non-zero crisis should differ from ELBO
    assert abs(elbo.item() - train_comp["total"]) > 1e-3, (
        "Validation ELBO should differ from training loss with gamma=3 and crisis weights"
    )


# ---------------------------------------------------------------------------
# 11. test_co_movement_symmetric — Co-movement loss symmetric
# ---------------------------------------------------------------------------

def test_co_movement_symmetric() -> None:
    """Co-movement loss should be symmetric: L_co(A, B) == L_co(B, A) when pairs are the same."""
    torch.manual_seed(42)
    mu = torch.randn(B, K)
    raw_returns = torch.randn(B, T)

    L_co_1 = compute_co_movement_loss(mu, raw_returns)

    # Reverse the batch order: pairs (i,j) become different
    # but overall loss should be the same since all pairs are still considered
    mu_rev = mu.flip(0)
    raw_returns_rev = raw_returns.flip(0)
    L_co_2 = compute_co_movement_loss(mu_rev, raw_returns_rev)

    assert torch.allclose(L_co_1, L_co_2, atol=1e-5), (
        f"Co-movement loss should be symmetric: "
        f"L_co_1={L_co_1.item():.6f}, L_co_2={L_co_2.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 12. test_loss_finite — No NaN or Inf for valid inputs
# ---------------------------------------------------------------------------

def test_loss_finite(tensors: dict[str, torch.Tensor]) -> None:
    """Loss should be finite (no NaN or Inf) for all modes with valid inputs."""
    torch.manual_seed(42)

    for mode, beta in [("P", 1.0), ("F", 1.0), ("A", 4.0)]:
        loss, components = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=tensors["log_sigma_sq"],
            crisis_fractions=tensors["crisis_fractions"],
            epoch=50,
            total_epochs=TOTAL_EPOCHS,
            mode=mode,
            gamma=3.0,
            beta_fixed=beta,
        )

        assert torch.isfinite(loss), f"Mode {mode}: loss is not finite: {loss.item()}"
        for key, val in components.items():
            assert math.isfinite(val), f"Mode {mode}: component '{key}' is not finite: {val}"
