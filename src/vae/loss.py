"""
VAE Loss computation: three modes (P/F/A), crisis weighting, co-movement loss.

Implements:
- Crisis-weighted reconstruction loss (MSE per-element mean × γ_eff)
- Per-feature weighted reconstruction loss (feature_weights rebalancing)
- KL divergence (summed over K, averaged over batch)
- Three assembly modes: P (primary), F (fallback), A (advanced)
- Co-movement loss (Spearman × cosine distance)
- Cross-sectional R² loss (factor model quality pressure on encoder)
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

# Guard flag: expensive tensor-scanning assertions are skipped in optimized mode.
# Set to False explicitly or run with `python -O` to disable.
_VALIDATE = __debug__


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

    # Vectorized: single (F,) tensor, one GPU→CPU sync
    mse_per_f = torch.mean((x - x_hat) ** 2, dim=(0, 1))  # (F,)
    return mse_per_f.tolist()


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
# Sub-task 2b: Per-feature weighted reconstruction loss
# ---------------------------------------------------------------------------

def compute_weighted_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    crisis_fractions: torch.Tensor,
    gamma: float,
    feature_weights: list[float] | None = None,
    feature_weights_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Crisis-weighted per-element MSE with optional per-feature reweighting.

    When feature_weights is provided, the MSE for each feature is weighted
    separately before averaging. This addresses reconstruction imbalance
    where the VAE over-fits to easy-to-reconstruct features (e.g., volatility)
    at the expense of harder features (e.g., returns).

    :param x (torch.Tensor): Input windows (B, T, F)
    :param x_hat (torch.Tensor): Reconstruction (B, T, F)
    :param crisis_fractions (torch.Tensor): f_c per window (B,)
    :param gamma (float): Crisis overweighting factor
    :param feature_weights (list[float] | None): Per-feature weights [w_0, w_1, ...].
        None or empty = equal weights (standard MSE). [2.0, 0.5] emphasizes
        returns (f=0) over volatility (f=1).
    :param feature_weights_tensor (torch.Tensor | None): Pre-built normalized (F,)
        tensor on device. Avoids per-batch torch.tensor() + normalization.

    :return L_recon (torch.Tensor): Scalar, weighted reconstruction loss
    """
    x = x.float()
    x_hat = x_hat.float()

    if feature_weights is None or len(feature_weights) == 0:
        return compute_reconstruction_loss(x, x_hat, crisis_fractions, gamma)

    F = x.shape[2]
    assert len(feature_weights) == F, (
        f"feature_weights length {len(feature_weights)} != F={F}"
    )

    # Per-feature MSE with weights: weighted mean over features
    if feature_weights_tensor is not None:
        fw = feature_weights_tensor
    else:
        fw = torch.tensor(feature_weights, device=x.device, dtype=torch.float32)
        fw = fw / fw.sum()  # normalize to sum=1

    # Squared error per element: (B, T, F)
    sq_err = (x - x_hat) ** 2

    # Weighted mean over T and F: for each window, sum over T of weighted-F
    # sq_err: (B, T, F), fw: (F,)
    # weighted_sq: (B, T, F) * (F,) -> (B, T, F), then mean over T, weighted sum over F
    mse_per_feature = torch.mean(sq_err, dim=1)  # (B, F)
    mse_per_window = torch.sum(mse_per_feature * fw.unsqueeze(0), dim=1)  # (B,)

    # Crisis weighting
    gamma_eff = 1.0 + crisis_fractions * (gamma - 1.0)
    return torch.mean(gamma_eff * mse_per_window)


# ---------------------------------------------------------------------------
# Sub-task 2c: Cross-sectional R² loss
# ---------------------------------------------------------------------------

def compute_cross_sectional_loss(
    mu: torch.Tensor,
    raw_returns: torch.Tensor,
    n_sample_dates: int = 20,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """
    Cross-sectional factor model R² loss.

    Forces the encoder to produce latent means (mu) that serve as useful
    factor exposures for explaining cross-sectional return variation.
    This bridges the gap between per-stock temporal training and
    cross-sectional factor model usage.

    At each sampled time offset t within the batch:
    1. Treat mu as factor exposures: B = mu  (B_valid, K)
    2. Get cross-sectional returns: r_t = raw_returns[:, t]  (B_valid,)
    3. OLS with ridge: z_hat = (B^T B + ridge·I)^{-1} B^T r_t
    4. Predict: r_hat = B @ z_hat
    5. R² = 1 - ||r - r_hat||² / ||r - mean(r)||²

    Loss = mean(1 - R²) over sampled dates.

    Differentiable through mu via torch.linalg.solve.

    :param mu (torch.Tensor): Latent means (B, K) — treated as factor exposures
    :param raw_returns (torch.Tensor): Raw returns per window (B, T)
    :param n_sample_dates (int): Number of time offsets to sample
    :param ridge (float): Ridge regularization for OLS stability

    :return L_cs (torch.Tensor): Scalar cross-sectional loss (1 - R²), in [0, 2]
    """
    B, K = mu.shape
    T = raw_returns.shape[1]

    if B < K + 1:
        # Not enough stocks for meaningful cross-sectional regression
        return torch.tensor(0.0, device=mu.device, requires_grad=True)

    # Disable AMP autocast: linalg.solve requires consistent float32 dtype,
    # but autocast downcasts matmuls to BFloat16/Float16 causing dtype mismatch.
    with torch.amp.autocast(device_type=mu.device.type, enabled=False):
        mu_f32 = mu.float()
        ret_f32 = raw_returns.float()

        # Sample time offsets (deterministic spacing for reproducibility)
        n_dates = min(n_sample_dates, T)
        step = max(1, T // n_dates)
        offsets = list(range(0, T, step))[:n_dates]

        # Gather all return cross-sections at once: (B, n_dates)
        R = ret_f32[:, offsets]

        # Filter dates with zero variance (e.g., holidays)
        r_var = R.var(dim=0)  # (n_dates,)
        valid = r_var > 1e-12
        R = R[:, valid]  # (B, n_valid)

        n_valid = R.shape[1]
        if n_valid == 0:
            return torch.tensor(0.0, device=mu.device, requires_grad=True)

        # Pre-compute B^T B + ridge * I (shared across all dates)
        BtB = mu_f32.T @ mu_f32 + ridge * torch.eye(K, device=mu.device)

        # Batched OLS: solve (B^T B + ridge*I) Z_hat = B^T R for all dates at once
        Bt_R = mu_f32.T @ R  # (K, n_valid)
        Z_hat = torch.linalg.solve(BtB, Bt_R)  # (K, n_valid) — single batched solve

        # Predict and compute R² vectorized
        R_hat = mu_f32 @ Z_hat  # (B, n_valid)
        R_mean = R.mean(dim=0, keepdim=True)  # (1, n_valid)
        ss_res = ((R - R_hat) ** 2).sum(dim=0)  # (n_valid,)
        ss_tot = ((R - R_mean) ** 2).sum(dim=0)  # (n_valid,)

        r_sq = 1.0 - ss_res / ss_tot.clamp(min=1e-12)

    return (1.0 - r_sq).mean()


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
    cross_sectional_loss: torch.Tensor | None = None,
    lambda_cs: float = 0.0,
    feature_weights: list[float] | None = None,
    feature_weights_tensor: torch.Tensor | None = None,
    compute_diagnostics: bool = True,
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
    :param cross_sectional_loss (torch.Tensor | None): Pre-computed L_cs scalar
    :param lambda_cs (float): Cross-sectional R² loss weight
    :param feature_weights (list[float] | None): Per-feature reconstruction weights
    :param feature_weights_tensor (torch.Tensor | None): Pre-built normalized (F,) tensor
        on the correct device. When provided, avoids per-batch torch.tensor() allocation.
    :param compute_diagnostics (bool): If True, compute per-feature MSE (slow).
        Set False for most training batches, True at logging intervals.

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

    # Reconstruction loss (per-element mean, crisis-weighted, optionally feature-weighted)
    L_recon = compute_weighted_reconstruction_loss(
        x, x_hat, crisis_fractions, gamma, feature_weights,
        feature_weights_tensor=feature_weights_tensor,
    )

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

    # Cross-sectional R² loss: bridges temporal training with cross-sectional usage
    L_cs = cross_sectional_loss if cross_sectional_loss is not None else torch.tensor(0.0, device=x.device)
    # Scale λ_cs by D/2 to make it commensurate with other loss terms
    cs_term = lambda_cs * (D / 2.0) * L_cs

    # Assembly by mode
    if mode == "P":
        # Mode P: D/(2σ²)·L_recon + (D/2)·ln(σ²) + L_KL + (D/2)·λ_co·L_co + (D/2)·λ_cs·L_cs
        recon_term = (D / (2.0 * sigma_sq)) * L_recon
        log_norm_term = (D / 2.0) * torch.log(sigma_sq)
        total_loss = recon_term + log_norm_term + L_kl + co_term + cs_term

    elif mode == "F":
        # Mode F: D/2·L_recon + β_t·L_KL + (D/2)·λ_co·L_co + (D/2)·λ_cs·L_cs
        # σ²=1 frozen (no gradient on log_sigma_sq)
        # β_t = max(β_min, min(1, epoch / T_warmup))
        beta_t_val = get_beta_t(epoch, total_epochs, warmup_fraction)

        recon_term = (D / 2.0) * L_recon
        log_norm_term = torch.tensor(0.0, device=x.device)
        total_loss = recon_term + beta_t_val * L_kl + co_term + cs_term

    else:  # mode == "A"
        # Mode A: D/(2σ²)·L_recon + (D/2)·ln(σ²) + β·L_KL + (D/2)·λ_co·L_co + (D/2)·λ_cs·L_cs
        recon_term = (D / (2.0 * sigma_sq)) * L_recon
        log_norm_term = (D / 2.0) * torch.log(sigma_sq)
        total_loss = recon_term + log_norm_term + beta_fixed * L_kl + co_term + cs_term

    # Per-feature reconstruction loss (expensive — only when diagnostics requested)
    per_feature_mse: list[float] = []
    if compute_diagnostics:
        per_feature_mse = compute_reconstruction_loss_per_feature(x, x_hat)

    # Monitoring components — detached tensors to avoid per-batch GPU sync
    components: dict[str, Any] = {
        "recon": L_recon.detach(),
        "kl": L_kl.detach(),
        "co_mov": L_co.detach(),
        "cross_sec": L_cs.detach(),
        "sigma_sq": sigma_sq.detach(),
        "lambda_co": lambda_co,
        "lambda_cs": lambda_cs,
        "recon_term": recon_term.detach(),
        "log_norm_term": log_norm_term.detach(),
        "total": total_loss.detach(),
        "recon_per_feature": per_feature_mse,
    }

    if mode == "F":
        components["beta_t"] = beta_t_val  # reuse already-computed value

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
    if _VALIDATE:
        assert not torch.isnan(raw_returns).any(), (
            "NaN in raw_returns before Spearman correlation"
        )
    with torch.no_grad():
        # Rank all B series ONCE, then index pairs (rank B vs 2×pairs series)
        all_ranks = _to_ranks(raw_returns)  # (B, T)
        spearman_corr = _batch_spearman_from_ranks(
            all_ranks[idx_i], all_ranks[idx_j],
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
    ranks_i = _to_ranks(returns_i)
    ranks_j = _to_ranks(returns_j)
    return _pearson_from_ranks(ranks_i, ranks_j)


def _batch_spearman_from_ranks(
    ranks_i: torch.Tensor,
    ranks_j: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Spearman correlation from pre-computed ranks (avoids redundant ranking).

    :param ranks_i (torch.Tensor): Pre-computed ranks for stock i (n_pairs, T)
    :param ranks_j (torch.Tensor): Pre-computed ranks for stock j (n_pairs, T)

    :return rho (torch.Tensor): Spearman correlations (n_pairs,)
    """
    return _pearson_from_ranks(ranks_i, ranks_j)


def _pearson_from_ranks(
    ranks_i: torch.Tensor,
    ranks_j: torch.Tensor,
) -> torch.Tensor:
    """
    Pearson correlation on rank tensors (= Spearman on original data).

    :param ranks_i (torch.Tensor): Ranks (n_pairs, T)
    :param ranks_j (torch.Tensor): Ranks (n_pairs, T)

    :return rho (torch.Tensor): Correlations (n_pairs,)
    """
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
    if _VALIDATE:
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

    # KL — inputs already float32, use internal computation directly to avoid double cast
    kl_per_sample = 0.5 * torch.sum(
        mu ** 2 + torch.exp(log_var) - log_var - 1.0,
        dim=1,
    )
    L_kl = torch.mean(kl_per_sample)

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
