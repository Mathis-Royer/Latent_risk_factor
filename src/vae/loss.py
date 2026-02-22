"""
VAE Loss computation: three modes (P/F/A), crisis weighting, co-movement loss.

Implements:
- Crisis-weighted reconstruction loss (MSE per-element mean × γ_eff)
- KL divergence (summed over K, averaged over batch)
- Three assembly modes: P (primary), F (fallback), A (advanced)
- Co-movement loss (Spearman × cosine distance)
- Curriculum scheduling for λ_co
- Validation ELBO (excludes γ and λ_co, INV-011)

Reference: ISD Section MOD-004.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F_torch

from src.validation import (
    assert_finite_tensor,
    assert_kl_non_negative,
    assert_reconstruction_bounded,
)


# ---------------------------------------------------------------------------
# Sub-task 1: Crisis-weighted reconstruction loss
# ---------------------------------------------------------------------------

def compute_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    crisis_fractions: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Crisis-weighted per-element MSE, averaged over batch.

    MSE(w) = (1/(T×F)) · Σ_{t,f} (x - x̂)²   [per-element mean per window]
    γ_eff(w) = 1 + f_c(w) · (γ - 1)            [continuous crisis weight]
    L_recon = (1/B) · Σ_w γ_eff(w) · MSE(w)    [batch mean, weighted]

    WARNING: MSE is per-element mean. The factor D = T×F is applied
    separately as a multiplicative coefficient in the assembly functions.

    :param x (torch.Tensor): Input windows (B, T, F)
    :param x_hat (torch.Tensor): Reconstruction (B, T, F)
    :param crisis_fractions (torch.Tensor): f_c per window (B,)
    :param gamma (float): Crisis overweighting factor

    :return L_recon (torch.Tensor): Scalar, weighted reconstruction loss
    """
    # Ensure float32 for numerically stable loss (inputs may be float16 from AMP)
    x = x.float()
    x_hat = x_hat.float()

    # Per-element squared error, mean over T and F dims → (B,)
    mse_per_window = torch.mean((x - x_hat) ** 2, dim=(1, 2))

    # Crisis weighting: γ_eff = 1 + f_c · (γ - 1)
    gamma_eff = 1.0 + crisis_fractions * (gamma - 1.0)

    # Weighted batch mean
    return torch.mean(gamma_eff * mse_per_window)


# ---------------------------------------------------------------------------
# Sub-task 1b: Per-feature reconstruction loss
# ---------------------------------------------------------------------------

def compute_reconstruction_loss_per_feature(
    x: torch.Tensor,
    x_hat: torch.Tensor,
) -> list[float]:
    """
    Compute per-feature MSE (averaged over batch and time).

    Used for diagnostics to identify if the VAE reconstructs some features
    (e.g., returns) better than others (e.g., volatility).

    :param x (torch.Tensor): Input windows (B, T, F)
    :param x_hat (torch.Tensor): Reconstruction (B, T, F)

    :return per_feature_mse (list[float]): MSE for each feature [mse_0, mse_1, ...]
    """
    # Ensure float32 for numerical stability
    x = x.float()
    x_hat = x_hat.float()

    F = x.shape[2]
    per_feature_mse: list[float] = []

    for f in range(F):
        # MSE for feature f, averaged over batch and time
        mse_f = torch.mean((x[:, :, f] - x_hat[:, :, f]) ** 2).item()
        per_feature_mse.append(mse_f)

    return per_feature_mse


# ---------------------------------------------------------------------------
# Sub-task 2: KL divergence
# ---------------------------------------------------------------------------

def compute_kl_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence: averaged over batch, summed over K, with 1/2 outside.

    L_KL = (1/B) · Σ_i (1/2) · Σ_k (μ²_ik + exp(log_var_ik) - log_var_ik - 1)

    :param mu (torch.Tensor): Latent mean (B, K)
    :param log_var (torch.Tensor): Latent log-variance (B, K)

    :return L_KL (torch.Tensor): Scalar KL loss
    """
    # Ensure float32 (exp(log_var) can overflow float16 when log_var > 11)
    mu = mu.float()
    log_var = log_var.float()

    # Sum over K dimensions, then average over batch
    kl_per_sample = 0.5 * torch.sum(
        mu ** 2 + torch.exp(log_var) - log_var - 1.0,
        dim=1,
    )
    return torch.mean(kl_per_sample)


# ---------------------------------------------------------------------------
# Sub-task 3-5: Loss assembly (Modes P/F/A)
# ---------------------------------------------------------------------------

def compute_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    crisis_fractions: torch.Tensor,
    epoch: int,
    total_epochs: int,
    mode: str,
    gamma: float = 3.0,
    lambda_co_max: float = 0.5,
    beta_fixed: float = 1.0,
    warmup_fraction: float = 0.20,
    co_movement_loss: torch.Tensor | None = None,
    sigma_sq_min: float = 1e-4,
    sigma_sq_max: float = 10.0,
    curriculum_phase1_frac: float = 0.30,
    curriculum_phase2_frac: float = 0.30,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Complete loss computation for a single batch.

    :param x (torch.Tensor): Input windows (B, T, F)
    :param x_hat (torch.Tensor): Reconstruction (B, T, F)
    :param mu (torch.Tensor): Latent mean (B, K)
    :param log_var (torch.Tensor): Latent log-variance (B, K)
    :param log_sigma_sq (torch.Tensor): Scalar log σ² (INV-002)
    :param crisis_fractions (torch.Tensor): f_c per window (B,)
    :param epoch (int): Current epoch (0-indexed)
    :param total_epochs (int): Total number of epochs
    :param mode (str): Loss mode — "P", "F", or "A"
    :param gamma (float): Crisis overweighting factor
    :param lambda_co_max (float): Maximum co-movement loss weight
    :param beta_fixed (float): Fixed beta for Mode A (must be 1.0 for P)
    :param warmup_fraction (float): Fraction of epochs for Mode F β warmup
    :param co_movement_loss (torch.Tensor | None): Pre-computed L_co scalar
    :param sigma_sq_min (float): Lower clamp for observation variance σ²
    :param sigma_sq_max (float): Upper clamp for observation variance σ²
    :param curriculum_phase1_frac (float): Fraction of epochs for full co-movement
    :param curriculum_phase2_frac (float): Fraction of epochs for linear decay

    :return total_loss (torch.Tensor): Scalar loss for backprop
    :return components (dict): Loss components for monitoring
    """
    # INV-002: σ² must be scalar
    assert log_sigma_sq.ndim == 0, f"INV-002: log_sigma_sq must be scalar, got ndim={log_sigma_sq.ndim}"

    # INV-006: modes are mutually exclusive
    assert mode in ("P", "F", "A"), f"INV-006: mode must be P, F, or A, got '{mode}'"
    if mode == "P":
        assert beta_fixed == 1.0, "INV-006: Mode P requires beta=1.0"

    T = x.shape[1]
    F = x.shape[2]
    D = T * F  # INV-001: D = T × F

    # Reconstruction loss (per-element mean, crisis-weighted)
    L_recon = compute_reconstruction_loss(x, x_hat, crisis_fractions, gamma)

    # KL loss
    L_kl = compute_kl_loss(mu, log_var)

    # Validate reconstruction loss
    assert not torch.isnan(L_recon), "L_recon is NaN after compute_reconstruction_loss"
    assert_reconstruction_bounded(L_recon, 1e6, "L_recon")

    # Validate KL divergence is non-negative (theory guarantee)
    assert_kl_non_negative(L_kl, "L_kl")

    # σ² = clamp(exp(log_sigma_sq), sigma_sq_min, sigma_sq_max)
    sigma_sq = torch.clamp(torch.exp(log_sigma_sq), min=sigma_sq_min, max=sigma_sq_max)
    assert sigma_sq_min <= float(sigma_sq.item()) <= sigma_sq_max, (
        f"sigma_sq={sigma_sq.item():.6g} outside [{sigma_sq_min}, {sigma_sq_max}]"
    )

    # Curriculum λ_co
    lambda_co = get_lambda_co(
        epoch, total_epochs, lambda_co_max,
        phase1_frac=curriculum_phase1_frac,
        phase2_frac=curriculum_phase2_frac,
    )

    # Co-movement contribution
    L_co = co_movement_loss if co_movement_loss is not None else torch.tensor(0.0, device=x.device)

    # Scale λ_co by D/2 so it is commensurate with the D-scaled reconstruction
    # term. Without this, λ_co·L_co is ~1000× smaller than D/(2σ²)·L_recon
    # and the co-movement gradient signal has no effect on the encoder.
    co_term = lambda_co * (D / 2.0) * L_co

    # Assembly by mode
    if mode == "P":
        # Mode P: D/(2σ²)·L_recon + (D/2)·ln(σ²) + L_KL + (D/2)·λ_co·L_co
        recon_term = (D / (2.0 * sigma_sq)) * L_recon
        log_norm_term = (D / 2.0) * torch.log(sigma_sq)
        total_loss = recon_term + log_norm_term + L_kl + co_term

    elif mode == "F":
        # Mode F: D/2·L_recon + β_t·L_KL + (D/2)·λ_co·L_co
        # σ²=1 frozen (no gradient on log_sigma_sq)
        # β_t = max(β_min, min(1, epoch / T_warmup))
        beta_t = get_beta_t(epoch, total_epochs, warmup_fraction)

        recon_term = (D / 2.0) * L_recon
        log_norm_term = torch.tensor(0.0, device=x.device)
        total_loss = recon_term + beta_t * L_kl + co_term

    else:  # mode == "A"
        # Mode A: D/(2σ²)·L_recon + (D/2)·ln(σ²) + β·L_KL + (D/2)·λ_co·L_co
        recon_term = (D / (2.0 * sigma_sq)) * L_recon
        log_norm_term = (D / 2.0) * torch.log(sigma_sq)
        total_loss = recon_term + log_norm_term + beta_fixed * L_kl + co_term

    # Per-feature reconstruction loss (for diagnostics)
    per_feature_mse = compute_reconstruction_loss_per_feature(x, x_hat)

    # Monitoring components — detached tensors to avoid per-batch GPU sync
    components: dict[str, Any] = {
        "recon": L_recon.detach(),
        "kl": L_kl.detach(),
        "co_mov": L_co.detach(),
        "sigma_sq": sigma_sq.detach(),
        "lambda_co": lambda_co,
        "recon_term": recon_term.detach(),
        "log_norm_term": log_norm_term.detach(),
        "total": total_loss.detach(),
        "recon_per_feature": per_feature_mse,  # [mse_return, mse_vol] for F=2
    }

    if mode == "F":
        components["beta_t"] = get_beta_t(epoch, total_epochs, warmup_fraction)

    assert_finite_tensor(total_loss, "total_loss")

    return total_loss, components


# ---------------------------------------------------------------------------
# Sub-task 6: Co-movement loss
# ---------------------------------------------------------------------------

def compute_co_movement_loss(
    mu: torch.Tensor,
    raw_returns: torch.Tensor,
    max_pairs: int = 2048,
) -> torch.Tensor:
    """
    Co-movement loss: penalizes latent distances that disagree with
    Spearman rank correlations computed on raw returns.

    For each eligible pair (i, j):
      1. ρ_ij = Spearman rank correlation on raw returns
      2. d(z_i, z_j) = cosine distance = 1 - cos_sim(μ_i, μ_j)
      3. g(ρ_ij) = 1 - ρ_ij  (target distance)
      4. L_co = (1/|P|) · Σ (d(z_i, z_j) - g(ρ_ij))²

    Simplified version: when called with synchronized windows from the same
    time block, all pairs are eligible. The caller is responsible for
    ensuring temporal synchronization.

    :param mu (torch.Tensor): Latent means (B, K)
    :param raw_returns (torch.Tensor): Raw returns for Spearman (B, T)
    :param max_pairs (int): Maximum number of pairs to sample

    :return L_co (torch.Tensor): Scalar co-movement loss
    """
    B = mu.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=mu.device, requires_grad=True)

    # Generate all pairs (or subsample)
    n_pairs = B * (B - 1) // 2
    if n_pairs == 0:
        return torch.tensor(0.0, device=mu.device, requires_grad=True)

    # Build pair indices
    idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=mu.device)

    # Subsample if too many pairs
    if len(idx_i) > max_pairs:
        perm = torch.randperm(len(idx_i), device=mu.device)[:max_pairs]
        idx_i = idx_i[perm]
        idx_j = idx_j[perm]

    # Spearman rank correlation on raw returns (no gradient needed)
    # CRITICAL: Check for NaN in raw_returns before Spearman (diagnostic fix)
    assert not torch.isnan(raw_returns).any(), (
        "NaN in raw_returns before Spearman correlation"
    )
    with torch.no_grad():
        spearman_corr = _batch_spearman(
            raw_returns[idx_i], raw_returns[idx_j],
        )
        # Target distance: g(ρ) = 1 - ρ
        target_dist = 1.0 - spearman_corr

    # Cosine distance in latent space (needs gradient)
    # Cast to float32: cosine_similarity norm can overflow float16
    mu_f32 = mu.float()
    cos_sim = F_torch.cosine_similarity(mu_f32[idx_i], mu_f32[idx_j], dim=1)
    cosine_dist = 1.0 - cos_sim

    # L_co = mean squared error between cosine distance and target
    L_co = torch.mean((cosine_dist - target_dist) ** 2)

    return L_co


def _batch_spearman(
    returns_i: torch.Tensor,
    returns_j: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Spearman rank correlation for pairs of return series.

    :param returns_i (torch.Tensor): Returns for stock i (n_pairs, T)
    :param returns_j (torch.Tensor): Returns for stock j (n_pairs, T)

    :return rho (torch.Tensor): Spearman correlations (n_pairs,)
    """
    # Convert to ranks along time dimension
    ranks_i = _to_ranks(returns_i)
    ranks_j = _to_ranks(returns_j)

    # Pearson correlation on ranks = Spearman
    # Centered ranks
    ranks_i_c = ranks_i - ranks_i.mean(dim=1, keepdim=True)
    ranks_j_c = ranks_j - ranks_j.mean(dim=1, keepdim=True)

    # Correlation
    num = torch.sum(ranks_i_c * ranks_j_c, dim=1)
    denom = torch.sqrt(
        torch.sum(ranks_i_c ** 2, dim=1) * torch.sum(ranks_j_c ** 2, dim=1)
    )

    # Clamp denominator to avoid division by zero
    rho = num / torch.clamp(denom, min=1e-8)
    rho = torch.clamp(rho, -1.0, 1.0)
    assert rho.min() >= -1.0 and rho.max() <= 1.0, (
        f"Spearman rho out of [-1, 1]: min={rho.min().item()}, max={rho.max().item()}"
    )
    return rho


def _to_ranks(x: torch.Tensor) -> torch.Tensor:
    """
    Convert each row to ranks (1-based) via single-pass scatter.

    Memory optimization: Uses torch.scatter_ instead of double argsort,
    eliminating one intermediate tensor (~4MB/batch savings).

    :param x (torch.Tensor): Values (n_pairs, T)

    :return ranks (torch.Tensor): Ranks (n_pairs, T), float
    """
    B, T_dim = x.shape
    ranks = torch.empty_like(x)
    sorted_indices = x.argsort(dim=1)
    # Create rank values [1, 2, ..., T] expanded to batch dimension
    rank_values = torch.arange(
        1, T_dim + 1, device=x.device, dtype=x.dtype,
    ).expand(B, -1)
    # Scatter rank values to their original positions
    ranks.scatter_(1, sorted_indices, rank_values)
    return ranks


# ---------------------------------------------------------------------------
# Sub-task 7: Curriculum scheduling
# ---------------------------------------------------------------------------

def get_lambda_co(
    epoch: int,
    total_epochs: int,
    lambda_co_max: float = 0.5,
    phase1_frac: float = 0.30,
    phase2_frac: float = 0.30,
) -> float:
    """
    Co-movement curriculum scheduling (INV-010).

    Phase 1 (0 → phase1_frac):                λ_co = λ_co_max
    Phase 2 (phase1_frac → phase1+phase2):     λ_co decays linearly to 0
    Phase 3 (phase1+phase2 → 100%):            λ_co = 0

    :param epoch (int): Current epoch (0-indexed)
    :param total_epochs (int): Total number of epochs
    :param lambda_co_max (float): Maximum co-movement loss weight
    :param phase1_frac (float): Fraction of epochs for full co-movement
    :param phase2_frac (float): Fraction of epochs for linear decay

    :return lambda_co (float): λ_co for this epoch
    """
    phase1_end = int(phase1_frac * total_epochs)
    phase2_end = int((phase1_frac + phase2_frac) * total_epochs)

    if epoch < phase1_end:
        return lambda_co_max
    elif epoch < phase2_end:
        progress = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
        return lambda_co_max * (1.0 - progress)
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Sub-task 8: Validation ELBO
# ---------------------------------------------------------------------------

def compute_validation_elbo(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    sigma_sq_min: float = 1e-4,
    sigma_sq_max: float = 10.0,
) -> torch.Tensor:
    """
    Validation ELBO — EXCLUDES γ and λ_co (INV-011), INCLUDES σ².

    L_val = D/(2σ²) · L_recon(γ=1) + (D/2)·ln(σ²) + L_KL

    Where L_recon(γ=1) is unweighted mean MSE.

    :param x (torch.Tensor): Input windows (B, T, F)
    :param x_hat (torch.Tensor): Reconstruction (B, T, F)
    :param mu (torch.Tensor): Latent mean (B, K)
    :param log_var (torch.Tensor): Latent log-variance (B, K)
    :param log_sigma_sq (torch.Tensor): Scalar log σ²
    :param sigma_sq_min (float): Lower clamp for observation variance σ²
    :param sigma_sq_max (float): Upper clamp for observation variance σ²

    :return L_val (torch.Tensor): Scalar validation ELBO
    """
    # Ensure float32 for numerical stability (inputs may be float16 from AMP)
    x = x.float()
    x_hat = x_hat.float()
    mu = mu.float()
    log_var = log_var.float()

    T = x.shape[1]
    F = x.shape[2]
    D = T * F

    # Unweighted MSE (γ=1 → γ_eff=1 for all)
    L_recon = torch.mean((x - x_hat) ** 2)

    # KL
    L_kl = compute_kl_loss(mu, log_var)

    # σ² (log_sigma_sq is a Parameter, always float32)
    sigma_sq = torch.clamp(torch.exp(log_sigma_sq), min=sigma_sq_min, max=sigma_sq_max)

    # Assembly: includes σ² terms
    L_val = (D / (2.0 * sigma_sq)) * L_recon + (D / 2.0) * torch.log(sigma_sq) + L_kl

    return L_val


# ---------------------------------------------------------------------------
# Utility: compute beta for Mode F
# ---------------------------------------------------------------------------

def get_beta_t(
    epoch: int,
    total_epochs: int,
    warmup_fraction: float = 0.20,
    beta_min: float = 0.01,
) -> float:
    """
    Linear β annealing for Mode F with floor.

    β_t = max(β_min, min(1, epoch / T_warmup))

    β_min > 0 ensures KL regularization is never fully disabled,
    preventing the encoder from pushing μ to extreme values during
    early warmup epochs.

    :param epoch (int): Current epoch (0-indexed)
    :param total_epochs (int): Total epochs
    :param warmup_fraction (float): Fraction for warmup
    :param beta_min (float): Minimum β to prevent KL collapse

    :return beta_t (float): Current β value
    """
    T_warmup = max(1, int(warmup_fraction * total_epochs))
    return max(beta_min, min(1.0, epoch / T_warmup))
