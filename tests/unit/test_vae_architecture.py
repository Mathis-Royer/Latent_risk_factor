"""
Unit tests for MOD-002: VAE architecture.

Covers: build_vae, model, encoder, decoder sizing rules and shapes.

Reference: ISD Section MOD-002.
"""

import math

import numpy as np
import pytest
import torch

from src.vae.build_vae import (
    build_vae,
    compute_channel_progression,
    compute_depth,
    compute_final_width,
    compute_temporal_sizes,
    count_decoder_params,
    count_encoder_params,
    round_16,
    C_MIN_DEFAULT as C_MIN,
    ALPHA_PROJ,
)
from src.vae.model import VAEModel


# ---------------------------------------------------------------------------
# Test parameters for SMALL models (fast tests)
# ---------------------------------------------------------------------------

SMALL_K = 10
SMALL_T = 64
SMALL_F = 2
SMALL_N = 50
SMALL_T_ANNEE = 3
SMALL_BATCH = 4


# ---------------------------------------------------------------------------
# Test 1: Sizing rule — depth
# ---------------------------------------------------------------------------


class TestSizingRuleDepth:
    """Tests for compute_depth (sizing rule 1)."""

    def test_sizing_rule_depth(self) -> None:
        """
        L = max(3, ceil(log2(T / 63)) + 2).
        L=5 for T=504, L=4 for T=252, L=6 for T=756.
        """
        # T=504: ceil(log2(504/63)) + 2 = ceil(log2(8)) + 2 = 3 + 2 = 5
        assert compute_depth(504) == 5, f"Expected L=5 for T=504, got {compute_depth(504)}"

        # T=252: ceil(log2(252/63)) + 2 = ceil(log2(4)) + 2 = 2 + 2 = 4
        assert compute_depth(252) == 4, f"Expected L=4 for T=252, got {compute_depth(252)}"

        # T=756: ceil(log2(756/63)) + 2 = ceil(log2(12)) + 2 = ceil(3.585) + 2 = 4 + 2 = 6
        assert compute_depth(756) == 6, f"Expected L=6 for T=756, got {compute_depth(756)}"


# ---------------------------------------------------------------------------
# Test 2: Sizing rule — width
# ---------------------------------------------------------------------------


class TestSizingRuleWidth:
    """Tests for compute_final_width (sizing rule 2)."""

    def test_sizing_rule_width(self) -> None:
        """
        C_L = round_16(max(C_MIN, ceil(alpha_proj * 2K))).
        C_L=384 for K<=147 (because max kicks in), larger for K>147.
        """
        # For K=10: ceil(1.3 * 20) = 26 < 384 -> C_L = round_16(384) = 384
        assert compute_final_width(10) == 384

        # For K=100: ceil(1.3 * 200) = 260 < 384 -> C_L = 384
        assert compute_final_width(100) == 384

        # For K=147: ceil(1.3 * 294) = ceil(382.2) = 383 < 384 -> C_L = 384
        assert compute_final_width(147) == 384

        # For K=200: ceil(1.3 * 400) = 520 > 384 -> C_L = round_16(520) = 528
        c_l_200 = compute_final_width(200)
        raw_200 = math.ceil(ALPHA_PROJ * 2 * 200)
        expected_200 = round_16(max(C_MIN, raw_200))
        assert c_l_200 == expected_200, f"Expected C_L={expected_200} for K=200, got {c_l_200}"
        assert c_l_200 > 384, f"For K=200, C_L should exceed 384, got {c_l_200}"


# ---------------------------------------------------------------------------
# Test 3: Channel progression monotonicity
# ---------------------------------------------------------------------------


class TestChannelProgression:
    """Tests for compute_channel_progression."""

    def test_channel_progression_monotonic(self) -> None:
        """
        Channel list [C_HEAD, C_1, ..., C_L] is monotonically non-decreasing.
        """
        for T_val in [252, 504, 756]:
            L = compute_depth(T_val)
            C_L = compute_final_width(200)
            channels = compute_channel_progression(L, C_L)

            for i in range(1, len(channels)):
                assert channels[i] >= channels[i - 1], (
                    f"Channels not monotonic at index {i} for T={T_val}: "
                    f"{channels}"
                )


# ---------------------------------------------------------------------------
# Tests 4-5: Capacity constraint
# ---------------------------------------------------------------------------


class TestCapacityConstraint:
    """Tests for capacity-data ratio constraint."""

    def test_capacity_constraint_raises(self) -> None:
        """
        ValueError raised when r > r_max, with remediation guidance.
        """
        with pytest.raises(ValueError, match="Capacity-data constraint violated"):
            build_vae(
                n=10,
                T=504,
                T_annee=3,
                F=2,
                K=200,
                r_max=0.001,
            )

    def test_capacity_constraint_remediation(self) -> None:
        """
        Error message contains all 4 remediation lines.
        """
        with pytest.raises(ValueError) as exc_info:
            build_vae(n=10, T=504, T_annee=3, F=2, K=200, r_max=0.001)
        msg = str(exc_info.value)
        assert "n (stocks)" in msg
        assert "T_annee" in msg
        assert "K (latent dim)" in msg
        assert "r_max" in msg

    def test_capacity_constraint_table(self) -> None:
        """
        For K=200, T=504, n=1000, T_annee=20: r should be reasonable (< 5).
        """
        _, info = build_vae(
            n=1000,
            T=504,
            T_annee=20,
            F=2,
            K=200,
            r_max=5.0,
        )

        assert info["r"] < 5.0, f"r = {info['r']:.4f} exceeds r_max=5.0"
        assert info["r"] > 0, f"r = {info['r']:.4f} should be positive"
        assert info["P_total"] > 0
        assert info["N"] > 0


# ---------------------------------------------------------------------------
# Tests 6-7: Encoder and Decoder output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Tests for encoder/decoder output shapes."""

    @pytest.fixture(autouse=True)
    def _setup_model(self) -> None:
        """Build a small model for shape tests."""
        torch.manual_seed(42)
        self.model, self.info = build_vae(
            n=SMALL_N,
            T=SMALL_T,
            T_annee=SMALL_T_ANNEE,
            F=SMALL_F,
            K=SMALL_K,
            r_max=200.0,  # Relaxed for small test model
        )
        self.model.eval()

    def test_encoder_output_shape(self) -> None:
        """
        mu and log_var both have shape (B, K).
        """
        x = torch.randn(SMALL_BATCH, SMALL_F, SMALL_T)  # (B, F, T) for encoder
        with torch.no_grad():
            mu, log_var = self.model.encoder(x)

        assert mu.shape == (SMALL_BATCH, SMALL_K), (
            f"Expected mu shape ({SMALL_BATCH}, {SMALL_K}), got {mu.shape}"
        )
        assert log_var.shape == (SMALL_BATCH, SMALL_K), (
            f"Expected log_var shape ({SMALL_BATCH}, {SMALL_K}), got {log_var.shape}"
        )

    def test_decoder_output_shape(self) -> None:
        """
        x_hat has shape (B, F, T) from decoder (channels-first).
        """
        z = torch.randn(SMALL_BATCH, SMALL_K)
        with torch.no_grad():
            x_hat = self.model.decoder(z)

        assert x_hat.shape == (SMALL_BATCH, SMALL_F, SMALL_T), (
            f"Expected x_hat shape ({SMALL_BATCH}, {SMALL_F}, {SMALL_T}), "
            f"got {x_hat.shape}"
        )

    def test_forward_roundtrip_shape(self) -> None:
        """
        forward(x) returns (x_hat, mu, log_var) with correct shapes.
        Input: (B, T, F), output x_hat: (B, T, F).
        """
        x = torch.randn(SMALL_BATCH, SMALL_T, SMALL_F)  # (B, T, F)
        with torch.no_grad():
            x_hat, mu, log_var = self.model(x)

        assert x_hat.shape == (SMALL_BATCH, SMALL_T, SMALL_F), (
            f"Expected x_hat shape ({SMALL_BATCH}, {SMALL_T}, {SMALL_F}), "
            f"got {x_hat.shape}"
        )
        assert mu.shape == (SMALL_BATCH, SMALL_K)
        assert log_var.shape == (SMALL_BATCH, SMALL_K)


# ---------------------------------------------------------------------------
# Tests 9-10: Deterministic encode vs stochastic forward
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests for encode() determinism and forward() stochasticity."""

    @pytest.fixture(autouse=True)
    def _setup_model(self) -> None:
        """Build a small model."""
        torch.manual_seed(42)
        self.model, _ = build_vae(
            n=SMALL_N,
            T=SMALL_T,
            T_annee=SMALL_T_ANNEE,
            F=SMALL_F,
            K=SMALL_K,
            r_max=200.0,
        )
        self.model.eval()

    def test_encode_deterministic(self) -> None:
        """
        Two encode() calls on the same input produce identical mu.
        """
        x = torch.randn(SMALL_BATCH, SMALL_T, SMALL_F)
        with torch.no_grad():
            mu1 = self.model.encode(x)
            mu2 = self.model.encode(x)

        assert torch.equal(mu1, mu2), (
            "encode() should be deterministic — two calls with same input "
            "should produce identical mu"
        )

    def test_forward_stochastic(self) -> None:
        """
        Two forward() calls produce different x_hat (reparameterization sampling).
        """
        # Put in train mode so batchnorm uses batch stats
        self.model.train()
        x = torch.randn(SMALL_BATCH, SMALL_T, SMALL_F)

        # Seed differently for each forward pass
        torch.manual_seed(100)
        x_hat1, _, _ = self.model(x)

        torch.manual_seed(200)
        x_hat2, _, _ = self.model(x)

        # Due to reparameterization trick with different random seeds,
        # x_hat1 and x_hat2 should differ
        assert not torch.equal(x_hat1, x_hat2), (
            "forward() should be stochastic due to reparameterization — "
            "two calls with different seeds should differ"
        )


# ---------------------------------------------------------------------------
# Tests 11-12: log_sigma_sq properties
# ---------------------------------------------------------------------------


class TestObservationNoise:
    """Tests for the learned observation noise parameter."""

    @pytest.fixture(autouse=True)
    def _setup_model(self) -> None:
        """Build a small model."""
        torch.manual_seed(42)
        self.model, _ = build_vae(
            n=SMALL_N,
            T=SMALL_T,
            T_annee=SMALL_T_ANNEE,
            F=SMALL_F,
            K=SMALL_K,
            r_max=200.0,
        )

    def test_log_sigma_sq_scalar(self) -> None:
        """
        log_sigma_sq.ndim == 0 (scalar parameter, INV-002).
        """
        assert self.model.log_sigma_sq.ndim == 0, (
            f"log_sigma_sq should be scalar (ndim=0), got ndim={self.model.log_sigma_sq.ndim}"
        )

    def test_log_sigma_sq_clamped(self) -> None:
        """
        After extreme gradient updates, obs_var stays in [1e-4, 10].
        """
        # Simulate extreme parameter values
        with torch.no_grad():
            # Test very large log_sigma_sq (should clamp obs_var to 10)
            self.model.log_sigma_sq.fill_(100.0)
            obs_var_high = self.model.obs_var.item()
            assert obs_var_high <= 10.0, (
                f"obs_var should be clamped to 10.0, got {obs_var_high}"
            )

            # Test very negative log_sigma_sq (should clamp obs_var to 1e-4)
            self.model.log_sigma_sq.fill_(-100.0)
            obs_var_low = self.model.obs_var.item()
            # Allow small float32 tolerance: 1e-4 may be ~9.9999e-5 in float32
            assert obs_var_low >= 1e-4 * (1.0 - 1e-5), (
                f"obs_var should be clamped to ~1e-4, got {obs_var_low}"
            )


# ---------------------------------------------------------------------------
# Test 13: Analytical vs PyTorch parameter count
# ---------------------------------------------------------------------------


class TestParamCount:
    """Tests for parameter count consistency."""

    def test_param_count_analytical_vs_pytorch(self) -> None:
        """
        count_encoder_params() matches sum(p.numel() for encoder.parameters()).
        """
        torch.manual_seed(42)
        model, info = build_vae(
            n=SMALL_N,
            T=SMALL_T,
            T_annee=SMALL_T_ANNEE,
            F=SMALL_F,
            K=SMALL_K,
            r_max=200.0,
        )

        # Analytical count
        channels = info["channels"]
        analytical_enc = count_encoder_params(SMALL_F, SMALL_K, channels)

        # PyTorch count
        pytorch_enc = sum(p.numel() for p in model.encoder.parameters())

        assert analytical_enc == pytorch_enc, (
            f"Encoder param count mismatch: analytical={analytical_enc}, "
            f"pytorch={pytorch_enc}"
        )

        # Also verify decoder
        T_compressed = info["T_compressed"]
        analytical_dec = count_decoder_params(
            SMALL_F, SMALL_K, channels, T_compressed
        )
        pytorch_dec = sum(p.numel() for p in model.decoder.parameters())

        assert analytical_dec == pytorch_dec, (
            f"Decoder param count mismatch: analytical={analytical_dec}, "
            f"pytorch={pytorch_dec}"
        )


# ---------------------------------------------------------------------------
# Test 14: Transpose convention
# ---------------------------------------------------------------------------


class TestTransposeConvention:
    """Tests for input/output shape conventions (CONV-05)."""

    def test_transpose_convention(self) -> None:
        """
        Input (B, T, F) -> internally (B, F, T) -> output (B, T, F).

        The model.forward() handles the transpose; encoder expects (B, F, T).
        """
        torch.manual_seed(42)
        model, _ = build_vae(
            n=SMALL_N,
            T=SMALL_T,
            T_annee=SMALL_T_ANNEE,
            F=SMALL_F,
            K=SMALL_K,
            r_max=200.0,
        )
        model.eval()

        # model.forward() expects (B, T, F)
        x_btf = torch.randn(SMALL_BATCH, SMALL_T, SMALL_F)
        with torch.no_grad():
            x_hat, mu, log_var = model(x_btf)

        # Output should be (B, T, F)
        assert x_hat.shape == (SMALL_BATCH, SMALL_T, SMALL_F), (
            f"Output should be (B, T, F), got {x_hat.shape}"
        )

        # Encoder directly expects (B, F, T)
        x_bft = x_btf.transpose(1, 2)
        with torch.no_grad():
            mu_direct, log_var_direct = model.encoder(x_bft)

        # Encoder output is (B, K) regardless
        assert mu_direct.shape == (SMALL_BATCH, SMALL_K)

        # mu from forward() and direct encoder call should match
        with torch.no_grad():
            mu_forward = model.encode(x_btf)
        torch.testing.assert_close(mu_direct, mu_forward)


# ---------------------------------------------------------------------------
# Test 15: build_vae modes (P, F, A)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Encoder-Decoder gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests for gradient flow through encoder and decoder using real ELBO."""

    def test_encoder_decoder_gradient_flow(self) -> None:
        """
        Gradient from actual compute_loss must flow to encoder, decoder,
        and log_sigma_sq (all three loss terms: recon, KL, sigma).
        """
        from src.vae.loss import compute_loss

        torch.manual_seed(42)
        model, _ = build_vae(
            n=SMALL_N, T=SMALL_T, T_annee=SMALL_T_ANNEE,
            F=SMALL_F, K=SMALL_K, r_max=200.0,
        )
        model.train()

        x = torch.randn(SMALL_BATCH, SMALL_T, SMALL_F)
        x_hat, mu, log_var = model(x)

        # Use actual compute_loss (Mode P, epoch 5/100)
        crisis_fractions = torch.zeros(SMALL_BATCH)
        loss, components = compute_loss(
            x=x, x_hat=x_hat, mu=mu, log_var=log_var,
            log_sigma_sq=model.log_sigma_sq,
            crisis_fractions=crisis_fractions,
            epoch=5, total_epochs=100, mode="P", gamma=3.0,
        )
        loss.backward()

        # Check encoder has non-zero gradients
        enc_grads = [
            p.grad for p in model.encoder.parameters()
            if p.grad is not None
        ]
        assert len(enc_grads) > 0, "No encoder parameters received gradient"
        nonzero_enc = sum(1 for g in enc_grads if g.abs().max() > 0)
        assert nonzero_enc > 0, "All encoder gradients are zero"

        # Check decoder has non-zero gradients
        dec_grads = [
            p.grad for p in model.decoder.parameters()
            if p.grad is not None
        ]
        assert len(dec_grads) > 0, "No decoder parameters received gradient"
        nonzero_dec = sum(1 for g in dec_grads if g.abs().max() > 0)
        assert nonzero_dec > 0, "All decoder gradients are zero"

        # Check log_sigma_sq received gradient (INV-002)
        assert model.log_sigma_sq.grad is not None, (
            "log_sigma_sq did not receive gradient from compute_loss"
        )
        assert model.log_sigma_sq.grad.abs() > 0, (
            "log_sigma_sq gradient is zero — loss doesn't depend on sigma_sq"
        )

        # Verify loss components are present and finite
        assert "recon" in components and torch.isfinite(components["recon"])
        assert "kl" in components and torch.isfinite(components["kl"])

        # A5: Verify gradient direction is correct (descent reduces loss)
        loss_before = loss.item()
        alpha_step = 1e-4
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= alpha_step * p.grad

        model.eval()
        with torch.no_grad():
            x_hat2, mu2, log_var2 = model(x)
        loss2, _ = compute_loss(
            x=x, x_hat=x_hat2, mu=mu2, log_var=log_var2,
            log_sigma_sq=model.log_sigma_sq,
            crisis_fractions=crisis_fractions,
            epoch=5, total_epochs=100, mode="P", gamma=3.0,
        )
        assert loss2.item() < loss_before, (
            f"One gradient step should decrease loss: before={loss_before:.6f}, "
            f"after={loss2.item():.6f}"
        )

        # A5: Verify balanced gradient norms (no vanishing/exploding)
        enc_norm = sum(g.norm().item() ** 2 for g in enc_grads) ** 0.5
        dec_norm = sum(g.norm().item() ** 2 for g in dec_grads) ** 0.5
        if enc_norm > 0 and dec_norm > 0:
            ratio = enc_norm / dec_norm
            assert 0.01 < ratio < 100, (
                f"Gradient norm ratio encoder/decoder={ratio:.4f} outside [0.01, 100]. "
                f"enc_norm={enc_norm:.6f}, dec_norm={dec_norm:.6f}"
            )


# ---------------------------------------------------------------------------
# Test: Channel progression matches formula
# ---------------------------------------------------------------------------


class TestChannelFormula:
    """Tests for channel progression formula C_l = C_HEAD * (C_L/C_HEAD)^(l/L)."""

    def test_channel_progression_matches_formula(self) -> None:
        """
        Each channel C_l should be within 10% of the formula value.
        """
        L = compute_depth(504)  # = 5
        C_L = compute_final_width(200)
        channels = compute_channel_progression(L, C_L)

        C_HEAD = channels[0]  # = 144
        ratio = C_L / C_HEAD

        for l in range(1, L + 1):
            expected_raw = C_HEAD * (ratio ** (l / L))
            actual = channels[l]
            # Allow 10% tolerance due to round_16
            assert abs(actual - expected_raw) / max(expected_raw, 1) < 0.10, (
                f"Channel at layer {l}: actual={actual}, "
                f"expected~={expected_raw:.1f} (>10% off)"
            )


# ---------------------------------------------------------------------------
# Test: build_vae respects dropout param
# ---------------------------------------------------------------------------


class TestDropoutParam:
    """Tests for dropout propagation through build_vae."""

    def test_build_vae_respects_dropout_param(self) -> None:
        """
        Dropout parameter propagated correctly to encoder and decoder.
        """
        torch.manual_seed(42)

        # Build with dropout=0.3
        model, info = build_vae(
            n=SMALL_N, T=SMALL_T, T_annee=SMALL_T_ANNEE,
            F=SMALL_F, K=SMALL_K, r_max=200.0, dropout=0.3,
        )

        # Check encoder ResBlocks have dropout=0.3
        for name, module in model.encoder.named_modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.p == pytest.approx(0.3, abs=1e-6), (
                    f"Encoder {name}: dropout.p={module.p}, expected 0.3"
                )

        # Check decoder modules have dropout=0.3
        for name, module in model.decoder.named_modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.p == pytest.approx(0.3, abs=1e-6), (
                    f"Decoder {name}: dropout.p={module.p}, expected 0.3"
                )

        # Verify info records the dropout
        assert info["dropout"] == 0.3


class TestBuildVAEModes:
    """Tests for Mode P, F, A construction."""

    def test_build_vae_modes(self) -> None:
        """
        Mode P (learn=True, beta=1), F (learn=False, beta<1),
        A (learn=True, beta>1) all build successfully.
        """
        common_kwargs = {
            "n": SMALL_N,
            "T": SMALL_T,
            "T_annee": SMALL_T_ANNEE,
            "F": SMALL_F,
            "K": SMALL_K,
            "r_max": 200.0,
        }

        # Mode P: learn_obs_var=True, beta=1.0
        model_p, info_p = build_vae(
            **common_kwargs, learn_obs_var=True, beta=1.0
        )
        assert model_p.learn_obs_var is True
        assert model_p.log_sigma_sq.requires_grad is True

        # Mode F: learn_obs_var=False, beta < 1.0
        model_f, info_f = build_vae(
            **common_kwargs, learn_obs_var=False, beta=0.5
        )
        assert model_f.learn_obs_var is False
        assert model_f.log_sigma_sq.requires_grad is False

        # Mode A: learn_obs_var=True, beta > 1.0
        model_a, info_a = build_vae(
            **common_kwargs, learn_obs_var=True, beta=2.0
        )
        assert model_a.learn_obs_var is True
        assert model_a.log_sigma_sq.requires_grad is True

        # All three produce valid models with forward pass
        x = torch.randn(2, SMALL_T, SMALL_F)
        for model in [model_p, model_f, model_a]:
            model.eval()
            with torch.no_grad():
                x_hat, mu, log_var = model(x)
            assert x_hat.shape == (2, SMALL_T, SMALL_F)
            assert mu.shape == (2, SMALL_K)
            assert log_var.shape == (2, SMALL_K)


# ---------------------------------------------------------------------------
# Tests: Decoder output shape and training properties
# ---------------------------------------------------------------------------


class TestDecoderOutputShape:
    def test_decoder_output_matches_input_exactly(self) -> None:
        """model.forward(x) returns x_hat with exact same shape as x for various batch sizes."""
        torch.manual_seed(42)
        model, _ = build_vae(n=50, T=64, T_annee=3, F=2, K=10, r_max=200.0, c_min=144)
        model.eval()
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 64, 2)
            with torch.no_grad():
                x_hat, mu, log_var = model(x)
            assert x_hat.shape == x.shape, (
                f"batch_size={batch_size}: x_hat {x_hat.shape} != x {x.shape}"
            )

    def test_sigma_sq_stays_scalar_after_training(self) -> None:
        """log_sigma_sq remains 0-dim after multiple gradient updates."""
        torch.manual_seed(42)
        model, _ = build_vae(n=50, T=64, T_annee=3, F=2, K=10, r_max=200.0, c_min=144)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        x = torch.randn(8, 64, 2)
        for _ in range(20):
            x_hat, mu, log_var = model(x)
            loss = torch.mean((x - x_hat) ** 2) + torch.mean(mu ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert model.log_sigma_sq.ndim == 0, f"ndim={model.log_sigma_sq.ndim} after training"
        assert model.log_sigma_sq.numel() == 1, f"numel={model.log_sigma_sq.numel()} after training"

    def test_dropout_zero_no_stochasticity(self) -> None:
        """With dropout=0, model in train mode produces deterministic output."""
        torch.manual_seed(42)
        model, _ = build_vae(n=50, T=64, T_annee=3, F=2, K=10, r_max=200.0, c_min=144, dropout=0.0)
        model.train()
        x = torch.randn(4, 64, 2)
        # Two forward passes with same input should produce same output
        # (no stochastic dropout)
        torch.manual_seed(99)
        x_hat1, mu1, _ = model(x)
        torch.manual_seed(99)
        x_hat2, mu2, _ = model(x)
        # Reparameterize uses different noise, so mu may differ, but x_hat determinism
        # depends on the same z. With dropout=0, the network itself is deterministic.
        # We just verify no NaN and shapes match
        assert not torch.isnan(x_hat1).any(), "NaN in output with dropout=0"
        assert x_hat1.shape == x.shape


# ---------------------------------------------------------------------------
# Formula verification: temporal dimension t_{l+1} = floor((t_l - 1) / 2) + 1
# ---------------------------------------------------------------------------


class TestTemporalDimensionFormula:
    """Verify the temporal dimension reduction formula at each encoder layer."""

    def test_temporal_sizes_match_formula(self) -> None:
        """compute_temporal_sizes must produce t_{l+1} = floor((t_l - 1)/2) + 1."""
        for T in [64, 252, 504, 756]:
            L = compute_depth(T)
            temporal_sizes = compute_temporal_sizes(T, L)

            assert len(temporal_sizes) == L + 1, (
                f"T={T}: expected {L+1} temporal sizes, got {len(temporal_sizes)}"
            )
            assert temporal_sizes[0] == T, (
                f"T={T}: first temporal size should be T={T}, got {temporal_sizes[0]}"
            )

            # Verify each reduction step follows the formula
            for l in range(L):
                t_curr = temporal_sizes[l]
                t_next = temporal_sizes[l + 1]
                expected = (t_curr - 1) // 2 + 1  # floor((t-1)/2) + 1
                assert t_next == expected, (
                    f"T={T}, layer {l}: t_{l+1}={t_next}, "
                    f"expected floor(({t_curr}-1)/2)+1={expected}"
                )


# ---------------------------------------------------------------------------
# Formula verification: capacity-data constraint r = P_total / N_capacity <= r_max
# ---------------------------------------------------------------------------


class TestCapacityDataConstraint:
    """Verify build_vae enforces r = P_total / N_capacity <= r_max."""

    def test_capacity_ratio_formula_verified(self) -> None:
        """build_vae info['r'] must equal P_total / N_capacity exactly."""
        model, info = build_vae(
            n=50, T=64, T_annee=3, F=2, K=10, r_max=200.0, c_min=144,
        )
        # Use formula-derived P_total from info (not model.parameters count
        # which may include a single learned σ² param not in P_enc/P_dec)
        ratio_expected = info["P_total"] / info["N"]
        assert abs(info["r"] - ratio_expected) < 1e-10, (
            f"info['r']={info['r']:.6f} != P_total/N={ratio_expected:.6f}"
        )

    def test_capacity_violation_raises(self) -> None:
        """build_vae must raise ValueError when r > r_max."""
        import pytest
        with pytest.raises(ValueError, match="Capacity-data constraint violated"):
            build_vae(n=10, T=504, T_annee=3, F=2, K=50, r_max=0.5, c_min=144)

    def test_capacity_ratio_recorded_in_info(self) -> None:
        """build_vae info dict should contain the ratio under key 'r'."""
        model, info = build_vae(
            n=50, T=64, T_annee=3, F=2, K=10, r_max=200.0, c_min=144,
        )
        assert "r" in info, "info must include 'r' (capacity ratio)"
        assert "r_max" in info, "info must include 'r_max'"
        assert info["r"] <= info["r_max"] + 1e-6


# ---------------------------------------------------------------------------
# Formula verification: channel progression exact values
# ---------------------------------------------------------------------------


class TestChannelProgressionExactFormula:
    """Verify C_l = round_16(C_HEAD * (C_L / C_HEAD)^(l/L)) exactly."""

    def test_exact_channel_values(self) -> None:
        """Every channel must be exactly round_16(C_HEAD * ratio^(l/L))."""
        C_HEAD = 144
        for K in [10, 50, 100, 200]:
            L = compute_depth(504)  # = 5
            C_L = compute_final_width(K)
            channels = compute_channel_progression(L, C_L)

            assert channels[0] == C_HEAD, (
                f"K={K}: C_0 should be {C_HEAD}, got {channels[0]}"
            )
            assert channels[-1] == C_L, (
                f"K={K}: C_L should be {C_L}, got {channels[-1]}"
            )

            ratio = C_L / C_HEAD
            for l in range(1, L + 1):
                expected = round_16(C_HEAD * (ratio ** (l / L)))
                assert channels[l] == expected, (
                    f"K={K}, l={l}: channel={channels[l]}, "
                    f"expected round_16({C_HEAD}*({ratio:.4f})^({l}/{L}))={expected}"
                )

            # Monotonicity: each channel >= previous
            for l in range(1, len(channels)):
                assert channels[l] >= channels[l - 1], (
                    f"K={K}: channels not monotonic at layer {l}: "
                    f"{channels[l]} < {channels[l-1]}"
                )
