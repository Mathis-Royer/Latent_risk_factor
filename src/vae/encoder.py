"""
VAE Encoder: InceptionHead + Residual body + Projection to latent space.

Architecture:
  InceptionHead(F) → L × ResBlock(c_in, c_out) → AdaptiveAvgPool1d(1)
    → Linear(C_L, K) [mu] + Linear(C_L, K) [log_var]

Input: (B, F, T) — channels-first for Conv1d.
Output: mu (B, K), log_var (B, K).

NOTE: Data arrives as (B, T, F) from data_pipeline (CONV-05).
Transpose to (B, F, T) is done in model.forward().

Reference: ISD Section MOD-002 — Sub-task 2.
"""

import torch
import torch.nn as nn


# Fixed hyperparameters (DVT Section 4.3)
K_HEAD_KERNELS = (5, 21, 63)
C_BRANCH = 48
C_HEAD = 3 * C_BRANCH  # 144
K_BODY = 7
DROPOUT = 0.1


class InceptionHead(nn.Module):
    """
    Three parallel Conv1d branches with kernels (5, 21, 63),
    each producing C_BRANCH=48 channels → concatenated to C_HEAD=144.

    Each branch: Conv1d + BatchNorm1d + GELU.
    Padding: k // 2 (same-length output).
    """

    def __init__(self, in_channels: int) -> None:
        """
        :param in_channels (int): Number of input features (F)
        """
        super().__init__()
        self.branches = nn.ModuleList()

        for k in K_HEAD_KERNELS:
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    C_BRANCH,
                    kernel_size=k,
                    padding=k // 2,
                    bias=True,
                ),
                nn.BatchNorm1d(C_BRANCH),
                nn.GELU(),
            )
            self.branches.append(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x (torch.Tensor): Input (B, F, T)

        :return out (torch.Tensor): Output (B, C_HEAD, T)
        """
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)  # (B, C_HEAD, T)


class ResBlock(nn.Module):
    """
    Residual block with two convolutions and a skip connection.

    First conv: stride=2 (downsamples temporal dimension).
    Second conv: stride=1.
    Skip: 1×1 conv with stride=2 (always active since stride changes dims).
    Dropout(0.1) after activation.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        """
        :param c_in (int): Input channels
        :param c_out (int): Output channels
        """
        super().__init__()

        # Main path
        self.conv1 = nn.Conv1d(
            c_in, c_out, kernel_size=K_BODY, stride=2, padding=K_BODY // 2,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(c_out)

        self.conv2 = nn.Conv1d(
            c_out, c_out, kernel_size=K_BODY, stride=1, padding=K_BODY // 2,
            bias=True,
        )
        self.bn2 = nn.BatchNorm1d(c_out)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT)

        # Skip connection (always active: stride=2 changes dimensions)
        self.skip = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=1, stride=2, bias=True),
            nn.BatchNorm1d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x (torch.Tensor): Input (B, c_in, T_in)

        :return out (torch.Tensor): Output (B, c_out, T_out) where T_out ≈ T_in/2
        """
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Match temporal dimensions if needed (off-by-one from stride)
        if out.shape[2] != identity.shape[2]:
            min_len = min(out.shape[2], identity.shape[2])
            out = out[:, :, :min_len]
            identity = identity[:, :, :min_len]

        out = out + identity
        out = self.act(out)
        return out


class Encoder(nn.Module):
    """
    Full encoder: InceptionHead → L × ResBlock → AdaptiveAvgPool1d(1)
      → Linear(C_L, K) [mu] + Linear(C_L, K) [log_var]
    """

    def __init__(
        self,
        F: int,
        K: int,
        channels: list[int],
    ) -> None:
        """
        :param F (int): Number of input features
        :param K (int): Latent dimension
        :param channels (list[int]): Channel progression [C_HEAD, C_1, ..., C_L]
        """
        super().__init__()
        self.K = K

        # Inception head: F → C_HEAD
        self.inception = InceptionHead(F)

        # Residual body: L blocks
        blocks = []
        for i in range(1, len(channels)):
            blocks.append(ResBlock(channels[i - 1], channels[i]))
        self.body = nn.Sequential(*blocks)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Projection to latent space
        C_L = channels[-1]
        self.fc_mu = nn.Linear(C_L, K)
        self.fc_log_var = nn.Linear(C_L, K)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x (torch.Tensor): Input (B, F, T), channels-first

        :return mu (torch.Tensor): Latent mean (B, K)
        :return log_var (torch.Tensor): Latent log-variance (B, K)
        """
        # Inception head
        h = self.inception(x)  # (B, C_HEAD, T)

        # Residual body
        h = self.body(h)  # (B, C_L, T_compressed)

        # Global average pooling
        h = self.pool(h)  # (B, C_L, 1)
        h = h.squeeze(-1)  # (B, C_L)

        # Projection
        mu = self.fc_mu(h)          # (B, K)
        log_var = self.fc_log_var(h)  # (B, K)

        return mu, log_var
