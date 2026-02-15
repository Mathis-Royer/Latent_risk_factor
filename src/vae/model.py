"""
VAEModel: top-level VAE wrapper combining encoder and decoder.

Handles:
- Input transpose: (B, T, F) → (B, F, T) for Conv1d
- Reparameterization trick for training
- Deterministic encode() for inference
- Learned scalar observation noise σ² (INV-002)

Reference: ISD Section MOD-002 — Sub-task 4.
"""

import torch
import torch.nn as nn

from src.vae.encoder import Encoder
from src.vae.decoder import Decoder


class VAEModel(nn.Module):
    """
    Variational Autoencoder for latent risk factor discovery.

    Attributes:
        encoder: Encoder network
        decoder: Decoder network
        K: int — latent dimension
        log_sigma_sq: nn.Parameter — scalar (INV-002), init 0.0 (σ²=1.0)
        learn_obs_var: bool — True for Mode P/A, False for Mode F
    """

    def __init__(
        self,
        F: int,
        K: int,
        channels: list[int],
        T: int,
        T_compressed: int,
        learn_obs_var: bool = True,
        dropout: float = 0.1,
        sigma_sq_min: float = 1e-4,
        sigma_sq_max: float = 10.0,
    ) -> None:
        """
        :param F (int): Number of input features
        :param K (int): Latent dimension
        :param channels (list[int]): Channel progression [C_HEAD, ..., C_L]
        :param T (int): Window length
        :param T_compressed (int): Encoder's last temporal size
        :param learn_obs_var (bool): Whether σ² is learned (Mode P/A)
        :param dropout (float): Dropout rate for residual blocks
        :param sigma_sq_min (float): Lower clamp for observation variance σ²
        :param sigma_sq_max (float): Upper clamp for observation variance σ²
        """
        super().__init__()

        self.F = F
        self.K = K
        self.T = T
        self.learn_obs_var = learn_obs_var
        self.sigma_sq_min = sigma_sq_min
        self.sigma_sq_max = sigma_sq_max

        # Encoder and decoder
        self.encoder = Encoder(F=F, K=K, channels=channels, dropout=dropout)
        self.decoder = Decoder(
            F=F, K=K, channels=channels, T=T, T_compressed=T_compressed,
            dropout=dropout,
        )

        # Observation noise: scalar σ² = exp(log_sigma_sq), init at 1.0
        # INV-002: MUST be scalar (ndim == 0)
        self.log_sigma_sq = nn.Parameter(
            torch.tensor(0.0),  # σ² = exp(0) = 1.0
            requires_grad=learn_obs_var,
        )

    @property
    def obs_var(self) -> torch.Tensor:
        """
        Observation variance σ² = clamp(exp(log_sigma_sq), sigma_sq_min, sigma_sq_max).

        INV-002: scalar, clamped to [sigma_sq_min, sigma_sq_max].

        :return sigma_sq (torch.Tensor): Scalar observation variance
        """
        return torch.clamp(torch.exp(self.log_sigma_sq), min=self.sigma_sq_min, max=self.sigma_sq_max)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + exp(0.5 * log_var) * ε, ε ~ N(0,1).

        :param mu (torch.Tensor): Latent mean (B, K)
        :param log_var (torch.Tensor): Latent log-variance (B, K)

        :return z (torch.Tensor): Sampled latent vector (B, K)
        """
        # Clamp log_var to prevent exp overflow under AMP float16/bfloat16.
        # log_var=20 → std=exp(10)≈22026 (safe); log_var=88 → exp(44) overflows float16.
        log_var = log_var.clamp(-20.0, 20.0)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass with reparameterization (for training).

        Input arrives as (B, T, F) from data_pipeline (CONV-05).
        Transposed to (B, F, T) for Conv1d, back to (B, T, F) for output.

        :param x (torch.Tensor): Input windows (B, T, F)

        :return x_hat (torch.Tensor): Reconstruction (B, T, F)
        :return mu (torch.Tensor): Latent mean (B, K)
        :return log_var (torch.Tensor): Latent log-variance (B, K)
        """
        # Transpose: (B, T, F) → (B, F, T)
        x_enc = x.transpose(1, 2)

        # Encode
        mu, log_var = self.encoder(x_enc)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        x_hat_enc = self.decoder(z)  # (B, F, T)

        # Transpose back: (B, F, T) → (B, T, F)
        x_hat = x_hat_enc.transpose(1, 2)

        return x_hat, mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic encoding for inference. Returns mu only (no sampling).

        CRITICAL: Use this for inference (MOD-006), NOT forward().

        :param x (torch.Tensor): Input windows (B, T, F)

        :return mu (torch.Tensor): Latent mean (B, K), deterministic
        """
        # Transpose: (B, T, F) → (B, F, T)
        x_enc = x.transpose(1, 2)

        # Encode (deterministic — mu only)
        mu, _ = self.encoder(x_enc)
        return mu
