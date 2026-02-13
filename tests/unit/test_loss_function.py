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
    assert abs(ratio - expected_coeff) < 1e-6, (
        f"D/(2*sigma^2) factor not present: ratio={ratio:.6f}, expected={expected_coeff:.6f}"
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

    # A1: Verify gradient VALUE matches the analytical formula.
    # Mode P loss = D/(2σ²)·L_recon + (D/2)·ln(σ²) + L_KL
    # ∂loss/∂log_σ² = σ² · ∂loss/∂σ²  (chain rule via σ²=exp(log_σ²))
    # ∂loss/∂σ² = -D/(2σ⁴)·L_recon + D/(2σ²)
    # → ∂loss/∂log_σ² = (D/2)·(1 - L_recon/σ²)
    D = T * F
    sigma_sq = torch.clamp(torch.exp(log_sigma_sq), min=1e-4, max=10.0)
    L_recon = components["recon"]
    expected_grad = (D / 2.0) * (1.0 - L_recon / sigma_sq.item())
    assert abs(log_sigma_sq.grad.item() - expected_grad) < 1e-4, (
        f"Mode P: gradient value mismatch. "
        f"actual={log_sigma_sq.grad.item():.6f}, "
        f"expected=(D/2)·(1 - L_recon/σ²)={expected_grad:.6f}"
    )


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
# 4b. test_beta_annealing_exact_linear_interpolation — formula verification
# ---------------------------------------------------------------------------

def test_beta_annealing_exact_linear_interpolation() -> None:
    """Verify exact linear beta_t formula: beta_t = max(beta_min, min(1, epoch / T_warmup)).

    Tests at 0%, 15%, 30%, 50%, and 100% of total epochs with two different
    warmup fractions to ensure the formula is correct, not just the boundaries.
    Uses the default beta_min=0.01.

    ISD/DVT coverage gap: formula-level verification of linear interpolation.
    """
    beta_min = 0.01

    # --- Configuration 1: warmup_fraction=0.20, total_epochs=100 ---
    total_epochs = 100
    warmup_fraction = 0.20
    T_warmup = max(1, int(warmup_fraction * total_epochs))  # = 20

    test_points = [
        # (epoch, expected_beta, description)
        (0, beta_min, "0% of epochs: epoch/T_warmup=0, clamped to beta_min"),
        (15, 15.0 / T_warmup, "15% of epochs: mid-warmup linear region"),
        (30, 1.0, "30% of epochs: past warmup, clamped to 1.0"),
        (50, 1.0, "50% of epochs: well past warmup"),
        (99, 1.0, "100% of epochs: final epoch"),
    ]

    for epoch, expected, desc in test_points:
        actual = get_beta_t(epoch, total_epochs, warmup_fraction)
        # Apply the same formula manually to get the exact expected value
        manual = max(beta_min, min(1.0, epoch / T_warmup))
        assert actual == manual, (
            f"Beta formula mismatch at epoch={epoch} ({desc}): "
            f"get_beta_t={actual}, manual formula={manual}"
        )
        assert abs(actual - expected) < 1e-12, (
            f"Beta value wrong at epoch={epoch} ({desc}): "
            f"actual={actual}, expected={expected}"
        )

    # --- Configuration 2: warmup_fraction=0.30, total_epochs=200 ---
    total_epochs_2 = 200
    warmup_fraction_2 = 0.30
    T_warmup_2 = max(1, int(warmup_fraction_2 * total_epochs_2))  # = 60

    # Fine-grained check: verify linearity at multiple interior points
    for epoch in range(0, T_warmup_2 + 10):
        actual = get_beta_t(epoch, total_epochs_2, warmup_fraction_2)
        manual = max(beta_min, min(1.0, epoch / T_warmup_2))
        assert abs(actual - manual) < 1e-15, (
            f"Linearity violated at epoch={epoch}: actual={actual}, manual={manual}"
        )

    # Verify strict monotonicity during warmup (after beta_min region)
    betas = [get_beta_t(e, total_epochs_2, warmup_fraction_2) for e in range(T_warmup_2 + 1)]
    for i in range(1, len(betas)):
        assert betas[i] >= betas[i - 1], (
            f"Beta not monotonically increasing: beta[{i-1}]={betas[i-1]}, beta[{i}]={betas[i]}"
        )

    # Verify beta_min floor is active at epoch 0 (not exactly 0.0)
    assert get_beta_t(0, total_epochs_2, warmup_fraction_2) == beta_min, (
        f"Beta at epoch 0 should be beta_min={beta_min}, not 0.0"
    )


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
    assert abs(actual_diff - expected_diff) < 1e-4, (
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

    # Per-window verification: manually compute gamma_eff and verify weighted MSE
    mse_per_window = torch.mean((x - x_hat) ** 2, dim=(1, 2))
    gamma_eff_crisis = 1.0 + cf_crisis * (3.0 - 1.0)  # [3.0, 3.0, 3.0, 3.0]
    expected_full_crisis = torch.mean(gamma_eff_crisis * mse_per_window)
    assert torch.allclose(loss_full_crisis, expected_full_crisis, atol=1e-6), (
        f"Full crisis loss does not match manual gamma_eff computation: "
        f"got {loss_full_crisis.item():.6f}, expected {expected_full_crisis.item():.6f}"
    )

    # Crisis loss must be strictly greater than no-crisis loss
    assert loss_full_crisis.item() > loss_no_crisis.item(), (
        f"Crisis loss ({loss_full_crisis.item():.6f}) should be strictly greater "
        f"than no-crisis loss ({loss_no_crisis.item():.6f})"
    )


# ---------------------------------------------------------------------------
# 8b. test_recon_loss_normalization_formula — MSE uses mean over T*F
# ---------------------------------------------------------------------------

def test_recon_loss_normalization_formula() -> None:
    """Verify reconstruction loss uses per-element mean (1/(T*F)) not sum.

    Formula: L_recon = (1/B) * sum_w [ gamma_eff(w) * (1/(T*F)) * sum_t sum_f (x - x_hat)^2 ]

    This test creates known x, x_hat tensors, manually computes the MSE using
    the explicit (1/(T*F)) normalization, and compares with:
    1. compute_reconstruction_loss output (batch-level, with gamma=1)
    2. compute_loss "recon" component (should match)

    ISD/DVT coverage gap: formula-level verification that MSE is per-element
    mean, not sum reduction (a common implementation error that silently
    scales gradients by T*F).
    """
    torch.manual_seed(123)

    # Use non-trivial dimensions to catch normalization errors
    B_test = 3
    T_test = 32
    F_test = 2
    K_test = 4
    D_test = T_test * F_test  # = 64

    x = torch.randn(B_test, T_test, F_test)
    x_hat = torch.randn(B_test, T_test, F_test)

    # --- Manual computation using explicit formula ---
    # Per-window MSE: (1/(T*F)) * sum_t sum_f (x_w - x_hat_w)^2
    squared_errors = (x - x_hat) ** 2  # (B, T, F)
    # Sum over T and F, then divide by T*F
    mse_per_window_manual = squared_errors.sum(dim=(1, 2)) / (T_test * F_test)  # (B,)

    # With gamma=1, gamma_eff=1 for all windows. L_recon = mean over batch.
    L_recon_manual = mse_per_window_manual.mean().item()

    # --- Via compute_reconstruction_loss (gamma=1, no crisis) ---
    crisis_fractions = torch.zeros(B_test)
    L_recon_func = compute_reconstruction_loss(x, x_hat, crisis_fractions, gamma=1.0)

    assert abs(L_recon_func.item() - L_recon_manual) < 1e-7, (
        f"compute_reconstruction_loss does not match manual (1/(T*F)) formula: "
        f"func={L_recon_func.item():.10f}, manual={L_recon_manual:.10f}"
    )

    # --- Via compute_loss "recon" component ---
    mu = torch.randn(B_test, K_test)
    log_var = torch.randn(B_test, K_test) * 0.5
    log_sigma_sq = torch.tensor(0.0)

    _, components = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis_fractions,
        epoch=50, total_epochs=100, mode="P", gamma=1.0,
        lambda_co_max=0.0,
    )

    assert abs(components["recon"].item() - L_recon_manual) < 1e-7, (
        f"compute_loss 'recon' component does not match manual formula: "
        f"func={components['recon'].item():.10f}, manual={L_recon_manual:.10f}"
    )

    # --- Verify it's NOT sum reduction (which would be T*F times larger) ---
    sum_reduction = squared_errors.sum(dim=(1, 2)).mean().item()  # without /TF
    assert abs(L_recon_manual - sum_reduction) > 1.0, (
        f"Manual MSE ({L_recon_manual:.6f}) is suspiciously close to sum reduction "
        f"({sum_reduction:.6f}). The function should use mean, not sum."
    )
    expected_ratio = D_test
    actual_ratio = sum_reduction / max(L_recon_manual, 1e-12)
    assert abs(actual_ratio - expected_ratio) < 0.01, (
        f"Sum/Mean ratio should be exactly D={expected_ratio}, got {actual_ratio:.4f}"
    )

    # --- Verify per-window MSE matches torch.mean over (T,F) dims ---
    mse_torch = torch.mean(squared_errors, dim=(1, 2))  # (B,)
    assert torch.allclose(mse_per_window_manual, mse_torch, atol=1e-10), (
        f"Manual sum/(T*F) should equal torch.mean over (T,F): "
        f"max diff={torch.max(torch.abs(mse_per_window_manual - mse_torch)).item()}"
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
    """Validation ELBO (INV-011) must use gamma-free formula.

    Verification strategy:
    1. Manually compute D/(2*sigma_sq)*MSE_unweighted + (D/2)*ln(sigma_sq) + KL
       (this formula has no gamma) and confirm it matches the function output.
    2. Confirm training losses change with different gammas while ELBO stays the same.
    """
    torch.manual_seed(42)

    # 1. Compute validation ELBO via function
    elbo = compute_validation_elbo(
        x=tensors["x"],
        x_hat=tensors["x_hat"],
        mu=tensors["mu"],
        log_var=tensors["log_var"],
        log_sigma_sq=tensors["log_sigma_sq"],
    )

    # 2. Independently compute the gamma-free formula
    D = T * F
    sigma_sq = torch.clamp(torch.exp(tensors["log_sigma_sq"]), 1e-4, 10.0)
    mse_unweighted = torch.mean((tensors["x"] - tensors["x_hat"]) ** 2).item()
    kl_manual = (0.5 * torch.sum(
        tensors["mu"] ** 2 + torch.exp(tensors["log_var"]) - tensors["log_var"] - 1,
        dim=1,
    )).mean().item()
    manual_elbo = (
        (D / (2 * sigma_sq.item())) * mse_unweighted
        + (D / 2) * math.log(sigma_sq.item())
        + kl_manual
    )
    assert abs(elbo.item() - manual_elbo) < 1e-4, (
        f"Validation ELBO does not match gamma-free formula: "
        f"func={elbo.item():.6f}, manual={manual_elbo:.6f}"
    )

    # 3. Training losses with different gammas must differ (gamma matters for training)
    # but ELBO is always the same (gamma-free)
    train_losses = []
    for gamma_val in [1.0, 3.0, 5.0]:
        _, comp = compute_loss(
            x=tensors["x"],
            x_hat=tensors["x_hat"],
            mu=tensors["mu"],
            log_var=tensors["log_var"],
            log_sigma_sq=tensors["log_sigma_sq"],
            crisis_fractions=tensors["crisis_fractions"],
            epoch=50,
            total_epochs=TOTAL_EPOCHS,
            mode="P",
            gamma=gamma_val,
        )
        train_losses.append(comp["total"].item())

    # With non-zero crisis_fractions, training loss must change with gamma
    assert abs(train_losses[0] - train_losses[1]) > 1e-3, (
        "Training loss should differ between gamma=1 and gamma=3"
    )
    assert abs(train_losses[1] - train_losses[2]) > 1e-3, (
        "Training loss should differ between gamma=3 and gamma=5"
    )


# ---------------------------------------------------------------------------
# 10b. test_D_factor_coefficient_exact_value — exact D/(2*sigma^2)
# ---------------------------------------------------------------------------

def test_D_factor_coefficient_exact_value(tensors: dict[str, torch.Tensor]) -> None:
    """Mode P: recon_coeff must be exactly D/(2*sigma^2) for known sigma^2."""
    torch.manual_seed(42)

    # Use known log_sigma_sq so sigma^2 is predictable
    for log_val in [-2.0, 0.0, 1.0]:
        log_sigma_sq = torch.tensor(log_val, requires_grad=True)
        sigma_sq = torch.clamp(torch.exp(log_sigma_sq), min=1e-4, max=10.0).item()

        _, comp = compute_loss(
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
        expected_coeff = D / (2.0 * sigma_sq)
        actual_ratio = comp["recon_term"] / max(comp["recon"], 1e-12)

        assert abs(actual_ratio - expected_coeff) < 1e-6, (
            f"log_sigma_sq={log_val}: ratio={actual_ratio:.6f}, "
            f"expected D/(2*sigma^2)={expected_coeff:.6f}"
        )


# ---------------------------------------------------------------------------
# 14. test_co_movement_loss_with_known_correlations
# ---------------------------------------------------------------------------

def test_co_movement_loss_with_known_correlations() -> None:
    """Co-movement loss correctness for known correlation scenarios.

    Formula: L_co = (1/|P|) Σ (d(z_i, z_j) - g(ρ_ij))²
    where d = cosine distance, g(ρ) = 1 - ρ (target distance).

    Case 1: Identical returns (ρ=1) + identical mu (d=0) → target=0, L_co ≈ 0
    Case 2: Identical returns (ρ=1) + different mu (d>0) → target=0, L_co > 0
    Case 3: Reversed returns (ρ≈-1) + identical mu (d=0) → target≈2, L_co ≈ 4
    """
    torch.manual_seed(42)

    B_test = 4
    K_test = 10
    T_test = 64

    # Case 1: Identical returns + identical mu → L_co ≈ 0
    base_returns = torch.randn(1, T_test)
    raw_returns_same = base_returns.expand(B_test, T_test).contiguous()
    mu_same = torch.ones(B_test, K_test)
    L_co_case1 = compute_co_movement_loss(mu_same, raw_returns_same)
    assert L_co_case1.item() < 1e-6, (
        f"Case 1 (ρ=1, d=0): L_co should be ≈0, got {L_co_case1.item()}"
    )

    # Case 2: Identical returns + different mu → L_co > 0
    mu_diff = torch.randn(B_test, K_test)
    L_co_case2 = compute_co_movement_loss(mu_diff, raw_returns_same)
    assert L_co_case2.item() > 0.0, (
        f"Case 2 (ρ=1, d>0): L_co should be > 0, got {L_co_case2.item()}"
    )
    assert torch.isfinite(L_co_case2), f"L_co is not finite: {L_co_case2.item()}"

    # Case 3: Anti-correlated returns (ρ≈-1) + identical mu (d=0)
    # g(ρ) = 1 - (-1) = 2, d(z_i, z_j) = 0
    # L_co = (0 - 2)² = 4 for each pair
    raw_pos = torch.linspace(0, 1, T_test).unsqueeze(0).expand(2, -1)
    raw_neg = -raw_pos
    raw_anti = torch.cat([raw_pos[:1], raw_neg[:1]], dim=0)  # 2 samples
    mu_ident = torch.ones(2, K_test)
    L_co_case3 = compute_co_movement_loss(mu_ident, raw_anti)
    # With ρ ≈ -1, target = 1 - (-1) = 2.0, cosine_dist = 0, loss per pair = (0 - 2)² = 4.0
    assert abs(L_co_case3.item() - 4.0) < 0.5, (
        f"Case 3 (ρ≈-1, d=0): L_co should be ≈4.0, got {L_co_case3.item()}"
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


# ---------------------------------------------------------------------------
# Tests: Mode properties, validation ELBO formula, co-movement with raw returns
# ---------------------------------------------------------------------------


class TestModeProperties:
    def test_mode_P_does_not_anneal_beta(self) -> None:
        """In Mode P, the KL contribution is constant across epochs (no beta annealing)."""
        T, F_val, K = 64, 2, 10
        B = 4
        torch.manual_seed(42)
        x = torch.randn(B, T, F_val)
        x_hat = torch.randn(B, T, F_val)
        mu = torch.randn(B, K)
        log_var = torch.randn(B, K)
        log_sigma_sq = torch.tensor(0.0)
        crisis = torch.zeros(B)
        losses_at_epochs = []
        for epoch in [0, 25, 50, 75, 99]:
            _, comp = compute_loss(
                x, x_hat, mu, log_var, log_sigma_sq, crisis,
                epoch=epoch, total_epochs=100, mode="P",
                gamma=1.0, lambda_co_max=0.0,
            )
            losses_at_epochs.append(comp["kl"].item())
        # KL contributions should be identical (no annealing)
        for i in range(1, len(losses_at_epochs)):
            assert abs(losses_at_epochs[i] - losses_at_epochs[0]) < 1e-8, (
                f"KL differs between epoch 0 and epoch {[0,25,50,75,99][i]}"
            )

    def test_mode_F_has_beta_annealing(self) -> None:
        """Mode F has beta that increases from beta_min to 1.0 during warmup."""
        # With warmup_fraction=0.20, total_epochs=100: T_warmup=20
        beta_0 = get_beta_t(0, 100, 0.20)
        beta_10 = get_beta_t(10, 100, 0.20)
        beta_20 = get_beta_t(20, 100, 0.20)
        beta_50 = get_beta_t(50, 100, 0.20)
        assert beta_0 == 0.01, f"beta at epoch 0 should be beta_min=0.01, got {beta_0}"
        assert beta_0 < beta_10 < beta_20, "beta should increase during warmup"
        assert beta_20 == 1.0, f"beta at T_warmup should be 1.0, got {beta_20}"
        assert beta_50 == 1.0, f"beta past warmup should be 1.0, got {beta_50}"

    def test_mode_F_beta_used_in_compute_loss(self) -> None:
        """compute_loss in Mode F must use get_beta_t: KL contribution differs between epochs."""
        T_val, F_val, K_val = 64, 2, 10
        B_val = 4
        torch.manual_seed(42)
        x = torch.randn(B_val, T_val, F_val)
        x_hat = torch.randn(B_val, T_val, F_val)
        mu = torch.randn(B_val, K_val)
        log_var = torch.randn(B_val, K_val)
        log_sigma_sq = torch.tensor(0.0)
        crisis = torch.zeros(B_val)

        # Epoch 0: beta_t = 0.01 (beta_min)
        _, comp_0 = compute_loss(
            x, x_hat, mu, log_var, log_sigma_sq, crisis,
            epoch=0, total_epochs=100, mode="F",
            gamma=1.0, beta_fixed=1.0, lambda_co_max=0.0,
        )

        # Epoch 20 (T_warmup with warmup_fraction=0.20): beta_t = 1.0
        _, comp_20 = compute_loss(
            x, x_hat, mu, log_var, log_sigma_sq, crisis,
            epoch=20, total_epochs=100, mode="F",
            gamma=1.0, beta_fixed=1.0, lambda_co_max=0.0,
        )

        # Recon should be the same (no dependency on epoch in Mode F recon)
        assert abs(comp_0["recon"].item() - comp_20["recon"].item()) < 1e-6, (
            "Reconstruction loss should not depend on epoch"
        )

        # Total loss should differ because beta_t differs (0.01 vs 1.0)
        # total_0 = D/2 * L_recon + 0.01 * L_kl
        # total_20 = D/2 * L_recon + 1.0 * L_kl
        # diff = 0.99 * L_kl
        expected_diff = 0.99 * comp_0["kl"].item()
        actual_diff = comp_20["total"].item() - comp_0["total"].item()
        assert abs(actual_diff - expected_diff) < 1e-4, (
            f"Mode F total loss difference should be 0.99*KL={expected_diff:.6f}, "
            f"got {actual_diff:.6f}"
        )

    def test_validation_elbo_formula_exact(self) -> None:
        """Validation ELBO matches manual formula: D/(2*sigma_sq)*MSE + (D/2)*log(sigma_sq) + KL."""
        T, F_val, K = 64, 2, 10
        D = T * F_val
        B = 4
        torch.manual_seed(42)
        x = torch.randn(B, T, F_val)
        x_hat = torch.randn(B, T, F_val)
        mu = torch.randn(B, K)
        log_var = torch.randn(B, K)
        log_sigma_sq = torch.tensor(0.5)
        sigma_sq = torch.clamp(torch.exp(log_sigma_sq), 1e-4, 10.0)
        # Manual computation
        L_recon = torch.mean((x - x_hat) ** 2).item()
        L_kl = (0.5 * torch.sum(mu**2 + torch.exp(log_var) - log_var - 1, dim=1)).mean().item()
        manual = (D / (2 * sigma_sq.item())) * L_recon + (D / 2) * torch.log(sigma_sq).item() + L_kl
        # Function
        L_val = compute_validation_elbo(x, x_hat, mu, log_var, log_sigma_sq).item()
        assert abs(L_val - manual) < 1e-4, f"ELBO mismatch: func={L_val}, manual={manual}"

    def test_co_movement_uses_raw_returns(self) -> None:
        """Co-movement loss correctness: verify exact Spearman rho and loss on known inputs.

        ISD MOD-004: 'DO NOT compute Spearman correlation on z-scored data — use raw returns.'
        Verifies both the Spearman computation and the loss formula
        L_co = mean((cosine_dist - (1 - rho))^2) with known expected values.
        """
        # Use T>=50 for reliable Spearman correlation (T=5 is degenerate
        # due to ties and exact rank patterns).
        T_co = 64
        n_stocks = 8
        K_co = 4

        # Case 1: Anti-correlated returns, identical latents
        # Build two groups of stocks: first half ascending, second half descending
        # so cross-sectional Spearman rho ~ -1 for opposite-group pairs.
        torch.manual_seed(42)
        t_axis = torch.linspace(0, 1, T_co)
        raw_anti = torch.zeros(n_stocks, T_co)
        for i in range(n_stocks // 2):
            raw_anti[i] = t_axis + torch.randn(T_co) * 0.05
        for i in range(n_stocks // 2, n_stocks):
            raw_anti[i] = -t_axis + torch.randn(T_co) * 0.05

        mu_identical = torch.ones(n_stocks, K_co)
        L_co_anti = compute_co_movement_loss(mu_identical, raw_anti)
        # With identical mu (cosine_dist=0) and mixed correlations, L_co > 0
        assert L_co_anti.item() > 0.1, (
            f"Anti-correlated groups + identical mu: "
            f"expected L_co > 0.1, got {L_co_anti.item():.6f}"
        )

        # Case 2: Perfectly correlated returns, identical latents
        # All stocks have the same trend -> pairwise rho ~ 1
        # Identical mu -> cosine_dist = 0, target = 1 - 1 = 0
        # L_co ~ 0
        raw_corr = torch.zeros(n_stocks, T_co)
        for i in range(n_stocks):
            raw_corr[i] = t_axis + torch.randn(T_co) * 0.01
        L_co_corr = compute_co_movement_loss(mu_identical, raw_corr)
        assert L_co_corr.item() < 0.1, (
            f"Correlated returns (rho~1) + identical mu (d=0): "
            f"expected L_co < 0.1, got {L_co_corr.item():.6f}"
        )

        # Case 3: Correlated returns, orthogonal latents
        # rho ~ 1 -> target ~ 0, but cosine_dist > 0 (orthogonal mu)
        # So L_co > 0 (distance mismatch)
        mu_orthogonal = torch.zeros(n_stocks, K_co)
        for i in range(n_stocks):
            mu_orthogonal[i, i % K_co] = 1.0
        L_co_orth = compute_co_movement_loss(mu_orthogonal, raw_corr)
        assert L_co_orth.item() > L_co_corr.item(), (
            f"Correlated returns + orthogonal mu should have higher L_co "
            f"than correlated returns + identical mu: "
            f"orthogonal={L_co_orth.item():.6f}, identical={L_co_corr.item():.6f}"
        )


# ---------------------------------------------------------------------------
# M1: D factor at production dimensions (T=504, F=2, D=1008)
# ---------------------------------------------------------------------------

def test_D_factor_production_dimensions() -> None:
    """
    M1: Verify D factor is correct at production size T=504, F=2 (D=1008).
    A bug that hardcodes D=128 would pass small tests but fail here.
    """
    T_prod = 504
    F_prod = 2
    D_prod = T_prod * F_prod  # = 1008
    B_prod = 2
    K_prod = 10

    torch.manual_seed(42)
    x = torch.randn(B_prod, T_prod, F_prod)
    x_hat = torch.randn(B_prod, T_prod, F_prod)
    mu = torch.randn(B_prod, K_prod)
    log_var = torch.randn(B_prod, K_prod) * 0.5
    log_sigma_sq = torch.tensor(0.5, requires_grad=True)
    crisis = torch.zeros(B_prod)

    sigma_sq = torch.clamp(torch.exp(log_sigma_sq), min=1e-4, max=10.0).item()
    expected_coeff = D_prod / (2.0 * sigma_sq)

    _, comp = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=50, total_epochs=100, mode="P", gamma=1.0,
    )

    actual_ratio = comp["recon_term"].item() / max(comp["recon"].item(), 1e-12)
    # At production dimensions (D=1008), the coefficient D/(2*sigma_sq) ~ 306.
    # Float32 has ~7 decimal digits of precision, so absolute error on ~306
    # can be up to ~3e-5. Use 5e-5 to remain tight while respecting float32 limits.
    assert abs(actual_ratio - expected_coeff) < 5e-5, (
        f"M1: D factor wrong at production dimensions. "
        f"Expected D/(2*sigma_sq) = {expected_coeff:.6f} (D={D_prod}), "
        f"got ratio = {actual_ratio:.6f}"
    )


# ---------------------------------------------------------------------------
# M3: Co-movement on z-scored vs raw returns must differ
# ---------------------------------------------------------------------------

def test_co_movement_z_scored_vs_raw_differs() -> None:
    """
    M3: ISD MOD-004 — 'DO NOT compute Spearman on z-scored data'.
    Per-row z-scoring preserves within-row ranks, so Spearman is invariant.
    However, cross-sectional z-scoring (across stocks at each time step)
    changes temporal ranks, producing different co-movement loss values.
    """
    torch.manual_seed(42)
    B_test = 8
    K_test = 8
    T_test = 64

    # Create returns with stock-specific trends so that cross-sectional
    # z-scoring changes within-stock temporal ranks.
    raw_returns = torch.randn(B_test, T_test)
    trends = torch.linspace(-2.0, 2.0, B_test).unsqueeze(1)  # (B, 1)
    t_axis = torch.linspace(0, 1, T_test).unsqueeze(0)  # (1, T)
    raw_returns = raw_returns + trends * t_axis

    # Cross-sectional z-scoring: normalize across stocks at each time step
    mean_cross = raw_returns.mean(dim=0, keepdim=True)  # (1, T)
    std_cross = raw_returns.std(dim=0, keepdim=True).clamp(min=1e-8)
    z_scored_cross = (raw_returns - mean_cross) / std_cross

    mu = torch.randn(B_test, K_test)

    L_co_raw = compute_co_movement_loss(mu, raw_returns)
    L_co_zscored = compute_co_movement_loss(mu, z_scored_cross)

    assert L_co_raw.item() != pytest.approx(L_co_zscored.item(), abs=1e-6), (
        f"M3: Co-movement loss should differ between raw and cross-sectionally "
        f"z-scored returns. raw={L_co_raw.item():.6f}, "
        f"z-scored={L_co_zscored.item():.6f}. "
        f"Spearman must be computed on raw returns (ISD MOD-004)."
    )


# ---------------------------------------------------------------------------
# m4: Mixed crisis fractions in a batch
# ---------------------------------------------------------------------------

def test_crisis_weight_mixed_fractions() -> None:
    """
    m4: Verify crisis weighting with mixed f_c values in a single batch.
    Existing tests only check f_c=0 (all) and f_c=1 (all). This tests
    a realistic mixed batch like [0.0, 0.3, 0.7, 1.0].
    """
    torch.manual_seed(42)
    x = torch.randn(B, T, F)
    x_hat = torch.randn(B, T, F)

    cf_mixed = torch.tensor([0.0, 0.3, 0.7, 1.0])
    gamma = 3.0

    loss = compute_reconstruction_loss(x, x_hat, cf_mixed, gamma=gamma)

    # Manually compute expected loss
    mse_per_window = torch.mean((x - x_hat) ** 2, dim=(1, 2))
    gamma_eff = 1.0 + cf_mixed * (gamma - 1.0)
    # gamma_eff = [1.0, 1.6, 2.4, 3.0]
    expected = torch.mean(gamma_eff * mse_per_window)

    assert torch.allclose(loss, expected, atol=1e-6), (
        f"Mixed crisis fractions: loss={loss.item():.6f}, "
        f"expected={expected.item():.6f}"
    )

    # Verify gamma_eff values match formula
    expected_gamma_eff = torch.tensor([1.0, 1.6, 2.4, 3.0])
    assert torch.allclose(gamma_eff, expected_gamma_eff, atol=1e-6), (
        f"gamma_eff mismatch: {gamma_eff.tolist()} vs {expected_gamma_eff.tolist()}"
    )

    # The loss should be between the all-f_c=0 and all-f_c=1 losses
    loss_no_crisis = compute_reconstruction_loss(
        x, x_hat, torch.zeros(B), gamma=gamma,
    )
    loss_full_crisis = compute_reconstruction_loss(
        x, x_hat, torch.ones(B), gamma=gamma,
    )
    assert loss_no_crisis.item() <= loss.item() <= loss_full_crisis.item(), (
        f"Mixed loss should be between no-crisis ({loss_no_crisis.item():.6f}) "
        f"and full-crisis ({loss_full_crisis.item():.6f}), got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# TIER 3a: Mode F beta_t annealing verification
# ---------------------------------------------------------------------------

def test_mode_F_beta_t_values_in_loss() -> None:
    """
    Mode F must produce beta_t = max(beta_min, min(1, epoch/T_warmup))
    at specific epochs, and the KL term must be scaled accordingly.

    DVT Section 4.4: Mode F uses linear beta annealing.
    """
    torch.manual_seed(42)
    x = torch.randn(B, T, F)
    x_hat = torch.randn(B, T, F)
    mu = torch.randn(B, 10)
    log_var = torch.randn(B, 10) * 0.5
    log_sigma_sq = torch.tensor(0.0)
    crisis = torch.zeros(B)

    total_epochs = 100
    warmup_fraction = 0.20
    T_warmup = int(warmup_fraction * total_epochs)  # = 20

    # Epoch 0: beta_t = max(0.01, min(1, 0/20)) = 0.01 (beta_min floor)
    _, comp_0 = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=0, total_epochs=total_epochs, mode="F",
        warmup_fraction=warmup_fraction,
    )
    assert abs(comp_0["beta_t"] - 0.01) < 1e-8, (
        f"At epoch=0, beta_t should be 0.01 (beta_min), got {comp_0['beta_t']}"
    )

    # Epoch 10 (mid-warmup): beta_t = max(0.01, min(1, 10/20)) = 0.5
    _, comp_10 = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=10, total_epochs=total_epochs, mode="F",
        warmup_fraction=warmup_fraction,
    )
    assert abs(comp_10["beta_t"] - 0.5) < 1e-8, (
        f"At epoch=10 (mid-warmup), beta_t should be 0.5, got {comp_10['beta_t']}"
    )

    # Epoch 20 (end of warmup): beta_t = max(0.01, min(1, 20/20)) = 1.0
    _, comp_20 = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=20, total_epochs=total_epochs, mode="F",
        warmup_fraction=warmup_fraction,
    )
    assert abs(comp_20["beta_t"] - 1.0) < 1e-8, (
        f"At epoch=20 (warmup end), beta_t should be 1.0, got {comp_20['beta_t']}"
    )

    # Epoch 50 (post-warmup): beta_t = 1.0 (clamped)
    _, comp_50 = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=50, total_epochs=total_epochs, mode="F",
        warmup_fraction=warmup_fraction,
    )
    assert abs(comp_50["beta_t"] - 1.0) < 1e-8, (
        f"At epoch=50 (post-warmup), beta_t should be 1.0, got {comp_50['beta_t']}"
    )


# ---------------------------------------------------------------------------
# TIER 3b: Co-movement lambda_co inclusion/exclusion in total loss
# ---------------------------------------------------------------------------

def test_co_movement_included_in_phase_1_excluded_in_phase_3() -> None:
    """
    Verify total loss includes lambda_co * L_co during phase 1 (epochs 0-30%)
    and excludes it during phase 3 (epochs 60-100%).

    ISD INV-010: Co-movement curriculum phases.
    """
    torch.manual_seed(42)
    x = torch.randn(B, T, F)
    x_hat = torch.randn(B, T, F)
    mu = torch.randn(B, 10)
    log_var = torch.randn(B, 10) * 0.5
    log_sigma_sq = torch.tensor(0.0)
    crisis = torch.zeros(B)

    total_epochs = 100
    lambda_co_max = 0.5

    # Create a non-zero co-movement loss
    L_co = torch.tensor(2.0)

    # Phase 1 (epoch 10): lambda_co should be lambda_co_max
    loss_with_co, comp_p1 = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=10, total_epochs=total_epochs, mode="P",
        lambda_co_max=lambda_co_max, co_movement_loss=L_co,
    )
    assert abs(comp_p1["lambda_co"] - lambda_co_max) < 1e-8, (
        f"Phase 1 lambda_co should be {lambda_co_max}, got {comp_p1['lambda_co']}"
    )

    # Compute loss without co-movement for comparison
    loss_no_co, comp_no = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=10, total_epochs=total_epochs, mode="P",
        lambda_co_max=lambda_co_max, co_movement_loss=torch.tensor(0.0),
    )

    # Difference should be lambda_co_max * L_co = 0.5 * 2.0 = 1.0
    expected_diff = lambda_co_max * L_co.item()
    actual_diff = loss_with_co.item() - loss_no_co.item()
    assert abs(actual_diff - expected_diff) < 1e-4, (
        f"Phase 1: total_loss difference should be lambda_co*L_co={expected_diff}, "
        f"got {actual_diff:.6f}"
    )

    # Phase 3 (epoch 80): lambda_co should be 0.0
    loss_p3, comp_p3 = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=80, total_epochs=total_epochs, mode="P",
        lambda_co_max=lambda_co_max, co_movement_loss=L_co,
    )
    assert abs(comp_p3["lambda_co"]) < 1e-8, (
        f"Phase 3 lambda_co should be 0.0, got {comp_p3['lambda_co']}"
    )

    # With lambda_co=0, the co-movement loss should not affect total
    loss_p3_no_co, _ = compute_loss(
        x=x, x_hat=x_hat, mu=mu, log_var=log_var,
        log_sigma_sq=log_sigma_sq, crisis_fractions=crisis,
        epoch=80, total_epochs=total_epochs, mode="P",
        lambda_co_max=lambda_co_max, co_movement_loss=torch.tensor(0.0),
    )
    assert abs(loss_p3.item() - loss_p3_no_co.item()) < 1e-6, (
        f"Phase 3: co-movement should not affect total loss. "
        f"With L_co={loss_p3.item():.6f}, without={loss_p3_no_co.item():.6f}"
    )


# ---------------------------------------------------------------------------
# End-to-end ELBO with fully known values (all terms computed manually)
# ---------------------------------------------------------------------------


class TestEndToEndELBOKnownValues:
    """Verify total loss equals manually computed ELBO for each mode (P, F, A)."""

    def test_mode_P_total_loss_known_values(self) -> None:
        """Mode P: total = D/(2σ²)·L_recon + (D/2)·ln(σ²) + KL + λ_co·L_co.

        With fully deterministic inputs, compute each term manually.
        """
        torch.manual_seed(0)
        # Use B=1 for simplicity
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])         # (1, 2, 2)
        x_hat = torch.tensor([[[1.1, 1.9], [2.8, 4.2]]])      # (1, 2, 2)
        mu = torch.tensor([[0.5, -0.3]])                       # (1, 2)
        log_var = torch.tensor([[0.0, -1.0]])                   # (1, 2)
        log_sigma_sq = torch.tensor(math.log(0.5))              # σ²=0.5
        crisis_fractions = torch.tensor([0.0])                  # no crisis

        D = 2 * 2  # T * F = 4
        sigma_sq = 0.5

        # L_recon = mean over batch of mean-per-element MSE
        mse = ((x - x_hat) ** 2).mean()  # mean over all elements
        L_recon_expected = mse.item()

        # recon_term = D/(2σ²) * L_recon
        recon_term_expected = (D / (2 * sigma_sq)) * L_recon_expected

        # log_term = (D/2) * ln(σ²)
        log_term_expected = (D / 2) * math.log(sigma_sq)

        # KL per dim: 0.5 * (exp(log_var) + mu^2 - 1 - log_var)
        # dim 0: 0.5 * (exp(0) + 0.25 - 1 - 0) = 0.5 * (1 + 0.25 - 1 - 0) = 0.125
        # dim 1: 0.5 * (exp(-1) + 0.09 - 1 - (-1)) = 0.5 * (0.3679 + 0.09 - 1 + 1) = 0.2289
        kl_per_dim = 0.5 * (torch.exp(log_var) + mu ** 2 - 1 - log_var)
        kl_expected = kl_per_dim.sum(dim=1).mean().item()  # sum over K, mean over batch

        # Total (no crisis, λ_co=0 at epoch 50 in phase 2)
        _, components = compute_loss(
            x=x, x_hat=x_hat, mu=mu, log_var=log_var,
            log_sigma_sq=log_sigma_sq, crisis_fractions=crisis_fractions,
            epoch=70, total_epochs=100, mode="P", gamma=1.0,
            lambda_co_max=0.0,
        )

        # Verify each component
        assert abs(components["recon"].item() - L_recon_expected) < 1e-6, (
            f"L_recon: got {components['recon'].item():.8f}, "
            f"expected {L_recon_expected:.8f}"
        )
        assert abs(components["recon_term"].item() - recon_term_expected) < 1e-4, (
            f"recon_term: got {components['recon_term'].item():.6f}, "
            f"expected {recon_term_expected:.6f}"
        )
        assert abs(components["kl"].item() - kl_expected) < 1e-6, (
            f"KL: got {components['kl'].item():.8f}, expected {kl_expected:.8f}"
        )

        # Total = recon_term + log_term + KL (gamma_eff=1 for f_c=0)
        total_expected = recon_term_expected + log_term_expected + kl_expected
        assert abs(components["total"].item() - total_expected) < 1e-4, (
            f"Total Mode P: got {components['total'].item():.6f}, "
            f"expected {total_expected:.6f}"
        )

    def test_mode_F_total_loss_known_values(self) -> None:
        """Mode F: total = (D/2)·L_recon + β_t·KL (no log σ² term, no σ² in recon)."""
        torch.manual_seed(0)
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        x_hat = torch.tensor([[[1.1, 1.9], [2.8, 4.2]]])
        mu = torch.tensor([[0.5, -0.3]])
        log_var = torch.tensor([[0.0, -1.0]])
        log_sigma_sq = torch.tensor(0.0)  # Not used in Mode F
        crisis_fractions = torch.tensor([0.0])

        D = 2 * 2  # T * F = 4

        # L_recon = MSE
        L_recon = ((x - x_hat) ** 2).mean().item()

        # recon_term = (D/2) * L_recon (no sigma_sq division)
        recon_term_expected = (D / 2) * L_recon

        # beta_t at epoch=100 (post-warmup) with warmup_fraction=0.5
        # epoch=100 is beyond warmup → beta_t = 1.0
        beta_t = 1.0

        kl_per_dim = 0.5 * (torch.exp(log_var) + mu ** 2 - 1 - log_var)
        kl_expected = kl_per_dim.sum(dim=1).mean().item()

        _, components = compute_loss(
            x=x, x_hat=x_hat, mu=mu, log_var=log_var,
            log_sigma_sq=log_sigma_sq, crisis_fractions=crisis_fractions,
            epoch=99, total_epochs=100, mode="F", gamma=1.0,
            warmup_fraction=0.5,
        )

        # Mode F: no log_sigma_sq term
        total_expected = recon_term_expected + beta_t * kl_expected

        assert abs(components["recon_term"].item() - recon_term_expected) < 1e-4, (
            f"Mode F recon_term: got {components['recon_term'].item():.6f}, "
            f"expected {recon_term_expected:.6f}"
        )
        assert abs(components["total"].item() - total_expected) < 1e-4, (
            f"Mode F total: got {components['total'].item():.6f}, "
            f"expected {total_expected:.6f}"
        )

    def test_validation_elbo_known_values(self) -> None:
        """Validation ELBO = D/(2σ²)·L_recon + (D/2)ln(σ²) + KL (no crisis, no co-movement)."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        x_hat = torch.tensor([[[1.1, 1.9], [2.8, 4.2]]])
        mu = torch.tensor([[0.5, -0.3]])
        log_var = torch.tensor([[0.0, -1.0]])
        log_sigma_sq = torch.tensor(math.log(0.5))

        D = 4
        sigma_sq = 0.5

        L_recon = ((x - x_hat) ** 2).mean().item()
        recon_term = (D / (2 * sigma_sq)) * L_recon
        log_term = (D / 2) * math.log(sigma_sq)
        kl_per_dim = 0.5 * (torch.exp(log_var) + mu ** 2 - 1 - log_var)
        kl = kl_per_dim.sum(dim=1).mean().item()

        expected_elbo = recon_term + log_term + kl

        elbo = compute_validation_elbo(x, x_hat, mu, log_var, log_sigma_sq)

        assert abs(elbo.item() - expected_elbo) < 1e-4, (
            f"Validation ELBO: got {elbo.item():.6f}, expected {expected_elbo:.6f}"
        )
