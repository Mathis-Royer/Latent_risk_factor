"""
VAE architecture construction: sizing rules and factory function.

The entire architecture is derived from 5 variable parameters:
(n, T, T_annee, F, K) with capacity-data constraint verification.

All fixed hyperparameters are from DVT Section 4.3.

Reference: ISD Section MOD-002 — Sub-tasks 1, 5, 6.
"""

import math

from src.vae.encoder import Encoder
from src.vae.decoder import Decoder
from src.vae.model import VAEModel


# ---------------------------------------------------------------------------
# Fixed hyperparameters (DVT Section 4.3)
# ---------------------------------------------------------------------------

K_HEAD = (5, 21, 63)         # Inception head kernels
C_BRANCH = 48                # Filters per Inception branch
C_HEAD = 3 * C_BRANCH        # = 144, total Inception output channels
K_BODY = 7                   # Residual body kernel size
STRIDE = 2                   # Per-block downsampling
ALPHA_PROJ = 1.3             # Projection ratio
C_MIN = 384                  # Minimum final layer width
DROPOUT = 0.1                # Dropout rate


def round_16(x: int) -> int:
    """
    Round to nearest multiple of 16 (for GPU memory alignment).

    :param x (int): Value to round

    :return rounded (int): Nearest multiple of 16 (at least 16)
    """
    return max(16, 16 * round(x / 16))


def compute_depth(T: int) -> int:
    """
    Sizing rule 1 — Encoder depth from window length.

    L(T) = max(3, ceil(log2(T / k_max)) + 2), k_max = 63.

    :param T (int): Window length in trading days

    :return L (int): Number of residual blocks
    """
    k_max = K_HEAD[-1]  # 63
    return max(3, math.ceil(math.log2(T / k_max)) + 2)


def compute_final_width(K: int) -> int:
    """
    Sizing rule 2 — Final layer width from latent capacity.

    C_L(K) = round_16(max(C_min, ceil(alpha_proj * 2K))).

    :param K (int): Latent capacity ceiling

    :return C_L (int): Final layer width (channels)
    """
    raw = math.ceil(ALPHA_PROJ * 2 * K)
    return round_16(max(C_MIN, raw))


def compute_channel_progression(L: int, C_L: int) -> list[int]:
    """
    Sizing rule 3 — Geometric interpolation from C_HEAD to C_L.

    C_l = round_16(C_HEAD * (C_L / C_HEAD)^(l/L)), l = 1, ..., L.

    :param L (int): Number of residual blocks
    :param C_L (int): Final layer width

    :return channels (list[int]): Channel widths [C_0=C_HEAD, C_1, ..., C_L]
    """
    channels = [C_HEAD]
    ratio = C_L / C_HEAD

    for l in range(1, L + 1):
        c = C_HEAD * (ratio ** (l / L))
        channels.append(round_16(int(round(c))))

    # Ensure final channel matches C_L
    channels[-1] = C_L

    # Ensure monotonically non-decreasing
    for i in range(1, len(channels)):
        if channels[i] < channels[i - 1]:
            channels[i] = channels[i - 1]

    return channels


def compute_temporal_sizes(T: int, L: int) -> list[int]:
    """
    Compute temporal dimension after each stride-2 block.

    At each block: t_out = (t_in - 1) // 2 + 1 (for stride-2 convolution).

    :param T (int): Input temporal dimension
    :param L (int): Number of residual blocks

    :return sizes (list[int]): Temporal sizes [T, t_1, ..., t_L]
    """
    sizes = [T]
    t = T
    for _ in range(L):
        t = (t - 1) // 2 + 1
        sizes.append(t)
    return sizes


def count_encoder_params(
    F: int,
    K: int,
    channels: list[int],
) -> int:
    """
    Analytical parameter count for the encoder.

    Inception head: sum over branches of (F * C_BRANCH * k + C_BRANCH + 2*C_BRANCH)
    Residual body: sum over blocks of conv1 + conv2 + skip + batchnorms
    Projection: C_L*K + K (mu) + C_L*K + K (log_var)

    :param F (int): Number of input features
    :param K (int): Latent dimension
    :param channels (list[int]): Channel progression [C_HEAD, C_1, ..., C_L]

    :return count (int): Total number of encoder parameters
    """
    C_L = channels[-1]

    # Inception head: 3 branches
    p_head = 0
    for k in K_HEAD:
        # Conv1d: F -> C_BRANCH, kernel k
        p_head += F * C_BRANCH * k + C_BRANCH
        # BatchNorm1d: 2 * C_BRANCH (weight + bias)
        p_head += 2 * C_BRANCH

    # Residual body
    p_body = 0
    for i in range(1, len(channels)):
        c_in = channels[i - 1]
        c_out = channels[i]

        # Conv1 (stride=2): c_in * c_out * K_BODY + c_out
        p_body += c_in * c_out * K_BODY + c_out
        # BN1: 2 * c_out
        p_body += 2 * c_out

        # Conv2 (stride=1): c_out * c_out * K_BODY + c_out
        p_body += c_out * c_out * K_BODY + c_out
        # BN2: 2 * c_out
        p_body += 2 * c_out

        # Skip (1x1 conv, always active since stride changes dimensions)
        p_body += c_in * c_out * 1 + c_out
        # Skip BN: 2 * c_out
        p_body += 2 * c_out

    # Projection heads: mu and log_var
    p_proj = C_L * K + K  # mu
    p_proj += C_L * K + K  # log_var

    return p_head + p_body + p_proj


def count_decoder_params(
    F: int,
    K: int,
    channels: list[int],
    T_compressed: int,
) -> int:
    """
    Analytical parameter count for the decoder.

    Initial projection: K * (C_L * T_comp) + (C_L * T_comp)
    Transposed body: same block structure, reversed channels
    Output head: C_HEAD * F + F

    :param F (int): Number of output features
    :param K (int): Latent dimension
    :param channels (list[int]): Channel progression (same as encoder)
    :param T_compressed (int): Encoder's last temporal size

    :return count (int): Total number of decoder parameters
    """
    C_L = channels[-1]

    # Initial projection: K -> C_L * T_comp
    p_proj = K * (C_L * T_compressed) + (C_L * T_compressed)

    # Transposed body (reversed channels)
    p_body = 0
    reversed_channels = list(reversed(channels))
    for i in range(1, len(reversed_channels)):
        c_in = reversed_channels[i - 1]
        c_out = reversed_channels[i]

        # ConvTranspose1d (stride=2): c_in * c_out * K_BODY + c_out
        p_body += c_in * c_out * K_BODY + c_out
        # BN: 2 * c_out
        p_body += 2 * c_out

        # Conv1d (stride=1): c_out * c_out * K_BODY + c_out
        p_body += c_out * c_out * K_BODY + c_out
        # BN: 2 * c_out
        p_body += 2 * c_out

        # Skip (ConvTranspose1d 1x1): c_in * c_out + c_out
        p_body += c_in * c_out * 1 + c_out
        # Skip BN: 2 * c_out
        p_body += 2 * c_out

    # Output head: 1x1 Conv from C_HEAD to F
    p_out = C_HEAD * F + F

    return p_proj + p_body + p_out


def build_vae(
    n: int,
    T: int,
    T_annee: int,
    F: int,
    K: int,
    s_train: int = 1,
    r_max: float = 5.0,
    beta: float = 1.0,
    learn_obs_var: bool = True,
) -> tuple[VAEModel, dict]:
    """
    Factory function: derive and instantiate the full VAE architecture.

    Mode selection via (learn_obs_var, beta):
      Mode P: learn_obs_var=True,  beta=1.0
      Mode F: learn_obs_var=False, beta=<1.0 (external β_t)
      Mode A: learn_obs_var=True,  beta=>1.0

    :param n (int): Number of stocks in the universe
    :param T (int): Window length in trading days
    :param T_annee (int): Total history length in years
    :param F (int): Number of features per timestep
    :param K (int): Latent capacity ceiling
    :param s_train (int): Training stride (window step)
    :param r_max (float): Maximum parameter/data ratio
    :param beta (float): KL weight (1.0 for Mode P)
    :param learn_obs_var (bool): Whether σ² is learned (True for P/A, False for F)

    :return model (VAEModel): Instantiated VAE model
    :return info (dict): Architecture info dictionary
    """
    # Derived quantities
    T_hist = T_annee * 252
    N_capacity = n * (T_hist - T + 1)  # Always computed at s=1
    N_train = n * ((T_hist - T) // s_train) + n

    # Sizing rules
    L = compute_depth(T)
    C_L = compute_final_width(K)
    channels = compute_channel_progression(L, C_L)
    temporal_sizes = compute_temporal_sizes(T, L)
    T_compressed = temporal_sizes[-1]

    # Parameter count
    P_enc = count_encoder_params(F, K, channels)
    P_dec = count_decoder_params(F, K, channels, T_compressed)
    P_total = P_enc + P_dec

    # Capacity-data constraint
    r = P_total / N_capacity
    if r > r_max:
        raise ValueError(
            f"Capacity constraint violated: r = {r:.4f} > r_max = {r_max}. "
            f"P_total = {P_total:,}, N = {N_capacity:,}. "
            f"Levers: increase n, increase T_annee, cap C_L at 384, or raise r_max."
        )

    # Instantiate model
    model = VAEModel(
        F=F,
        K=K,
        channels=channels,
        T=T,
        T_compressed=T_compressed,
        learn_obs_var=learn_obs_var,
    )

    info = {
        "L": L,
        "channels": channels,
        "temporal_sizes": temporal_sizes,
        "C_L": C_L,
        "T_compressed": T_compressed,
        "P_enc": P_enc,
        "P_dec": P_dec,
        "P_total": P_total,
        "N": N_capacity,
        "N_train": N_train,
        "s_train": s_train,
        "r": r,
        "r_max": r_max,
        "T_hist": T_hist,
    }

    return model, info
