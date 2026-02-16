"""
VAE Decoder: Linear projection + Transposed residual body + Output head.

Architecture:
  Linear(K, C_L × T_comp) → reshape (B, C_L, T_comp)
    → L × TransposeResBlock (reversed channel progression)
    → trim/pad to target T → Conv1d(C_HEAD, F, kernel_size=1)

Output: (B, F, T) — channels-first.

Reference: ISD Section MOD-002 — Sub-task 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

# Fixed hyperparameters
C_HEAD = 144
K_BODY = 7


class TransposeResBlock(nn.Module):
    """
    Transposed residual block for the decoder.

    ConvTranspose1d (stride=2, output_padding=1) + Conv1d (stride=1).
    Skip: ConvTranspose1d (1×1, stride=2, output_padding=1).
    Each with BatchNorm + GELU + Dropout.
    Doubles temporal dimension at each block.
    """

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.1) -> None:
        """
        :param c_in (int): Input channels
        :param c_out (int): Output channels
        :param dropout (float): Dropout rate
        """
        super().__init__()

        # Main path: upsample then refine
        self.conv_t = nn.ConvTranspose1d(
            c_in, c_out, kernel_size=K_BODY, stride=2,
            padding=K_BODY // 2, output_padding=1, bias=True,
        )
        self.bn1 = nn.BatchNorm1d(c_out)

        self.conv2 = nn.Conv1d(
            c_out, c_out, kernel_size=K_BODY, stride=1,
            padding=K_BODY // 2, bias=True,
        )
        self.bn2 = nn.BatchNorm1d(c_out)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Skip connection: upsample with 1×1 transposed conv
        self.skip = nn.Sequential(
            nn.ConvTranspose1d(
                c_in, c_out, kernel_size=1, stride=2,
                output_padding=1, bias=True,
            ),
            nn.BatchNorm1d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x (torch.Tensor): Input (B, c_in, T_in)

        :return out (torch.Tensor): Output (B, c_out, T_out) where T_out ≈ 2*T_in
        """
        identity = self.skip(x)

        out = self.conv_t(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Match temporal dimensions (transposed convs may differ by 1)
        if out.shape[2] != identity.shape[2]:
            min_len = min(out.shape[2], identity.shape[2])
            out = out[:, :, :min_len]
            identity = identity[:, :, :min_len]

        out = out + identity
        out = self.act(out)
        return out


class Decoder(nn.Module):
    """
    Full decoder: Linear → reshape → L × TransposeResBlock → trim/pad → 1×1 Conv.

    T_comp = encoder's last temporal size (e.g., 16 for T=504, L=5).
    Decoder produces 16→32→64→128→256→512 vs encoder's 504→252→126→63→32→16.
    Final trim/pad ensures output matches T exactly.
    """

    def __init__(
        self,
        F: int,
        K: int,
        channels: list[int],
        T: int,
        T_compressed: int,
        dropout: float = 0.1,
    ) -> None:
        """
        :param F (int): Number of output features
        :param K (int): Latent dimension
        :param channels (list[int]): Channel progression [C_HEAD, C_1, ..., C_L]
            (same as encoder — will be reversed internally)
        :param T (int): Target output temporal dimension
        :param T_compressed (int): Encoder's last temporal size
        :param dropout (float): Dropout rate for residual blocks
        """
        super().__init__()
        self.T = T
        self.T_compressed = T_compressed

        C_L = channels[-1]
        self.C_L = C_L

        # Initial projection: z → (C_L, T_comp)
        self.fc = nn.Linear(K, C_L * T_compressed)

        # Transposed residual body (reversed channel progression)
        reversed_channels = list(reversed(channels))
        blocks = []
        for i in range(1, len(reversed_channels)):
            blocks.append(
                TransposeResBlock(
                    reversed_channels[i - 1], reversed_channels[i], dropout=dropout,
                )
            )
        self.body = nn.Sequential(*blocks)

        # Output head: 1×1 conv from C_HEAD to F
        self.output_head = nn.Conv1d(C_HEAD, F, kernel_size=1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z (torch.Tensor): Latent vector (B, K)

        :return x_hat (torch.Tensor): Reconstruction (B, F, T), channels-first
        """
        B = z.shape[0]

        # Project and reshape
        h = self.fc(z)  # (B, C_L * T_comp)
        h = h.view(B, self.C_L, self.T_compressed)  # (B, C_L, T_comp)

        # Transposed residual body
        h = self.body(h)  # (B, C_HEAD, T_approx)

        # Trim or pad to exact target T
        if h.shape[2] > self.T:
            h = h[:, :, :self.T]
        elif h.shape[2] < self.T:
            pad_size = self.T - h.shape[2]
            h = F_torch.pad(h, (0, pad_size), mode="replicate")

        # Output head
        x_hat = self.output_head(h)  # (B, F, T)

        return x_hat
