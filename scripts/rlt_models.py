"""
RLT (RL Tokens) – model components for the liquidnets pipeline.

Adapted from Physical Intelligence's RLT paper (March 2026).

Core idea applied to our setting:
  1. Freeze the pretrained liquid policy (SharedBackbone + LiquidTrajectoryModel).
  2. Train a small encoder–decoder transformer that compresses the backbone's
     latent sequence into a single "RL token" via a reconstruction loss.
  3. Train a lightweight SAC actor and critic that operate on the RL token
     (+ the base policy's proposed action) to produce a *residual* correction.

This file defines all new modules.  Training loops live in separate scripts.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# 1.  RL-token encoder / decoder
# ---------------------------------------------------------------------------

class RLTokenEncoder(nn.Module):
    """
    Encoder: maps the backbone latent sequence (B, T, D_backbone) -> (B, D_token).

    Architecture: small 2-layer Transformer encoder whose final CLS-like output
    is projected to a compact RL token vector.
    """

    def __init__(self, input_dim: int = 256, token_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.token_dim = token_dim

        # Learnable query that will become the RL token
        self.rl_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        self.pos_emb = nn.Embedding(17, input_dim)  # up to 16 seq + 1 query
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=input_dim * 4, dropout=dropout, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(input_dim, token_dim)

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """latent_seq: (B, T, D_backbone) -> rl_token: (B, D_token)."""
        b, t, _ = latent_seq.shape
        query = self.rl_query.expand(b, -1, -1)        # (B, 1, D)
        out = self.cross_attn(query, latent_seq)        # (B, 1, D)
        return self.proj(out.squeeze(1))                # (B, D_token)


class RLTokenDecoder(nn.Module):
    """
    Decoder: reconstructs the backbone latent sequence from the RL token.

    (B, D_token) -> (B, T, D_backbone)
    """

    def __init__(self, token_dim: int = 64, output_dim: int = 256,
                 seq_length: int = 16, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_length = seq_length
        self.output_dim = output_dim

        self.token_proj = nn.Linear(token_dim, output_dim)
        self.pos_emb = nn.Embedding(seq_length, output_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim, nhead=num_heads,
            dim_feedforward=output_dim * 4, dropout=dropout, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, rl_token: torch.Tensor) -> torch.Tensor:
        """rl_token: (B, D_token) -> reconstructed: (B, T, D_backbone)."""
        b = rl_token.shape[0]
        memory = self.token_proj(rl_token).unsqueeze(1)  # (B, 1, D)
        device = rl_token.device
        pos = self.pos_emb(torch.arange(self.seq_length, device=device))
        queries = pos.unsqueeze(0).expand(b, -1, -1)     # (B, T, D)
        decoded = self.cross_attn(queries, memory)        # (B, T, D)
        return self.output_proj(decoded)                  # (B, T, D_backbone)


class RLTokenBottleneck(nn.Module):
    """Encoder + Decoder trained jointly via reconstruction loss."""

    def __init__(self, backbone_dim: int = 256, token_dim: int = 64,
                 seq_length: int = 16, **kwargs):
        super().__init__()
        self.encoder = RLTokenEncoder(input_dim=backbone_dim, token_dim=token_dim, **kwargs)
        self.decoder = RLTokenDecoder(token_dim=token_dim, output_dim=backbone_dim,
                                      seq_length=seq_length, **kwargs)

    def forward(self, latent_seq: torch.Tensor):
        """
        latent_seq: (B, T, D) -> (rl_token, reconstructed)
        """
        rl_token = self.encoder(latent_seq)         # (B, D_token)
        recon = self.decoder(rl_token)              # (B, T, D)
        return rl_token, recon


# ---------------------------------------------------------------------------
# 2.  SAC actor  (residual on top of base policy action)
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ResidualActor(nn.Module):
    """
    Takes: RL token + base policy action chunk (flattened).
    Outputs: Gaussian parameters for residual delta to add to the base action.

    ref_dropout_prob: during training, randomly zero-out the reference action
    input to prevent the actor from simply copying it (per RLT paper).
    """

    def __init__(self, token_dim: int = 64, action_dim: int = 7,
                 seq_length: int = 16, hidden_dim: int = 256,
                 ref_dropout_prob: float = 0.1):
        super().__init__()
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.ref_dropout_prob = ref_dropout_prob
        flat_action = action_dim * seq_length             # 112 for robomimic
        in_dim = token_dim + flat_action

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, flat_action)
        self.log_std_head = nn.Linear(hidden_dim, flat_action)

        # Initialize residual output near zero so initial policy ≈ base policy
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.constant_(self.log_std_head.bias, -2.0)

    def forward(self, rl_token: torch.Tensor, base_action_flat: torch.Tensor,
                deterministic: bool = False):
        """
        rl_token:         (B, D_token)
        base_action_flat: (B, action_dim * seq_length)   — base policy output, normalized
        Returns:
            action: (B, action_dim * seq_length)  — final corrected action
            log_prob: (B,)
            residual_mu: (B, action_dim * seq_length)
        """
        ref = base_action_flat
        # Reference-action dropout
        if self.training and self.ref_dropout_prob > 0:
            mask = (torch.rand(ref.shape[0], 1, device=ref.device) > self.ref_dropout_prob).float()
            ref = ref * mask

        x = torch.cat([rl_token, ref], dim=-1)
        h = self.trunk(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            residual = mu
            log_prob = torch.zeros(rl_token.shape[0], device=rl_token.device)
        else:
            dist = Normal(mu, std)
            residual_raw = dist.rsample()
            # Squash through tanh for bounded exploration
            residual = torch.tanh(residual_raw)
            log_prob = dist.log_prob(residual_raw) - torch.log(1 - residual.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)

        # Residual scaling: limit the correction magnitude
        residual_scale = 0.3   # max ±0.3 in normalized action space
        action = (base_action_flat + residual_scale * residual).clamp(-1.0, 1.0)
        return action, log_prob, mu


# ---------------------------------------------------------------------------
# 3.  SAC critic  (twin Q)
# ---------------------------------------------------------------------------

class TwinQCritic(nn.Module):
    """
    Two independent Q-networks for SAC.
    Input: RL token + action chunk (flattened).
    Output: two scalar Q-values.
    """

    def __init__(self, token_dim: int = 64, action_dim: int = 7,
                 seq_length: int = 16, hidden_dim: int = 256):
        super().__init__()
        flat_action = action_dim * seq_length
        in_dim = token_dim + flat_action

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rl_token: torch.Tensor, action_flat: torch.Tensor):
        x = torch.cat([rl_token, action_flat], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 4.  Replay buffer  (simple, numpy-backed)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size ring buffer storing (rl_token, base_action, reward, next_rl_token, next_base_action, done)."""

    def __init__(self, capacity: int, token_dim: int, action_flat_dim: int):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.tokens = torch.zeros(capacity, token_dim)
        self.actions = torch.zeros(capacity, action_flat_dim)
        self.base_actions = torch.zeros(capacity, action_flat_dim)
        self.rewards = torch.zeros(capacity)
        self.next_tokens = torch.zeros(capacity, token_dim)
        self.next_base_actions = torch.zeros(capacity, action_flat_dim)
        self.dones = torch.zeros(capacity)

    def push(self, token, base_action, action, reward, next_token, next_base_action, done):
        """All inputs are 1-D tensors (single transition)."""
        i = self.idx % self.capacity
        self.tokens[i] = token
        self.base_actions[i] = base_action
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_tokens[i] = next_token
        self.next_base_actions[i] = next_base_action
        self.dones[i] = float(done)
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idxs = torch.randint(0, self.size, (batch_size,))
        return {
            "token": self.tokens[idxs].to(device),
            "base_action": self.base_actions[idxs].to(device),
            "action": self.actions[idxs].to(device),
            "reward": self.rewards[idxs].to(device),
            "next_token": self.next_tokens[idxs].to(device),
            "next_base_action": self.next_base_actions[idxs].to(device),
            "done": self.dones[idxs].to(device),
        }
