"""
Football Transformer — Event Detection Model.

Architecture:
  1. PlayerEncoder    — per-frame MLP encodes each player's features → embedding
  2. SpatialAttention — cross-attention over all players in one frame
  3. TemporalEncoder  — Transformer over the sequence of frames
  4. EventHead        — MLP classifier → event probabilities

Input shape:  (batch, seq_len, max_players, player_features)
Output shape: (batch, num_events)

Events:
  0  background    — nothing notable happening
  1  pass          — player passes the ball
  2  shot          — attempt on goal
  3  cross         — ball delivered from wide
  4  tackle        — defensive sliding challenge
  5  interception  — player intercepts the pass
  6  duel          — 50/50 contest
  7  turnover      — ball lost in open play
  8  foul          — illegal challenge

Design decisions:
- Max 23 "slots" per frame: 22 players + 1 ball.
  Slots that are empty are masked out in attention.
- Positional encoding is sinusoidal over the time dimension.
- The ball is treated as a special player with a learnable role embedding.
- Model is small enough to train on a CPU in a few hours with StatsBomb data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FootballTransformerConfig:
    # Input
    player_features: int = 7    # x, y, speed, accel, direction_sin, direction_cos, is_ball
    max_players: int = 23       # 22 players + 1 ball slot
    seq_len: int = 25           # frames in one sequence (1 sec at 25fps)

    # Architecture
    player_embed_dim: int = 64  # per-player embedding size
    hidden_dim: int = 128       # main transformer hidden size
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    ffn_dim: int = 256          # feedforward expansion

    # Output
    num_events: int = 9

    @property
    def event_names(self) -> list[str]:
        return [
            "background",
            "pass",
            "shot",
            "cross",
            "tackle",
            "interception",
            "duel",
            "turnover",
            "foul",
        ]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PlayerEncoder(nn.Module):
    """
    MLP that encodes per-player features → player embedding.
    Applied independently to each player in each frame.
    """

    def __init__(self, in_features: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., player_features) → (..., out_dim)
        return self.net(x)


class SpatialAttention(nn.Module):
    """
    Multi-head self-attention over players within a single frame.
    Aggregates to a single frame embedding by taking the mean of
    attended player embeddings.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,          # (batch, max_players, dim)
        mask: torch.Tensor | None, # (batch, max_players) True = padding
    ) -> torch.Tensor:             # (batch, dim)
        residual = x
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm(x + residual)

        # Masked mean pooling → one vector per frame
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)  # (batch, max_players, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return x  # (batch, dim)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the time dimension."""

    def __init__(self, dim: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FootballTransformer(nn.Module):
    """
    Full Football Transformer for event detection.

    Forward pass:
        players:  (B, T, P, F)  — tracking sequences
        masks:    (B, T, P)     — True where slot is empty (padding)

    Returns:
        logits:   (B, num_events)  — raw (un-softmaxed) class scores
    """

    def __init__(self, config: FootballTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or FootballTransformerConfig()
        cfg = self.config

        # 1. Per-player feature encoder
        self.player_encoder = PlayerEncoder(cfg.player_features, cfg.player_embed_dim)

        # 2. Project player embeddings to hidden_dim if different
        self.proj = (
            nn.Linear(cfg.player_embed_dim, cfg.hidden_dim)
            if cfg.player_embed_dim != cfg.hidden_dim
            else nn.Identity()
        )

        # 3. Spatial attention (per frame)
        self.spatial_attn = SpatialAttention(cfg.hidden_dim, cfg.num_heads, cfg.dropout)

        # 4. Positional encoding + Transformer over time
        self.pos_enc = SinusoidalPositionalEncoding(cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,   # Pre-LN for stable training
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # 5. Classification head — use [CLS]-style: mean pool over time
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.num_events),
        )

        self._init_weights()

    def forward(
        self,
        players: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, P, F = players.shape

        # --- 1. Encode each player in each frame ---
        x = self.player_encoder(players)           # (B, T, P, player_embed_dim)
        x = self.proj(x)                           # (B, T, P, hidden_dim)

        # --- 2. Spatial attention: aggregate players → one vector per frame ---
        x = x.view(B * T, P, -1)                  # (B*T, P, hidden_dim)
        if masks is not None:
            flat_masks = masks.view(B * T, P)
        else:
            flat_masks = None
        x = self.spatial_attn(x, flat_masks)       # (B*T, hidden_dim)
        x = x.view(B, T, -1)                       # (B, T, hidden_dim)

        # --- 3. Temporal Transformer ---
        x = self.pos_enc(x)                        # add positional encoding
        x = self.temporal_encoder(x)               # (B, T, hidden_dim)

        # --- 4. Mean pool over time → classify ---
        x = x.mean(dim=1)                          # (B, hidden_dim)
        logits = self.classifier(x)                # (B, num_events)
        return logits

    def predict(self, players: torch.Tensor, masks: torch.Tensor | None = None) -> dict:
        """Convenience method: returns class index + probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(players, masks)
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

        names = self.config.event_names
        return {
            "event": names[pred[0].item()],
            "confidence": probs[0, pred[0]].item(),
            "probabilities": {
                name: round(probs[0, i].item(), 4)
                for i, name in enumerate(names)
            },
        }

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config: FootballTransformerConfig | None = None) -> FootballTransformer:
    return FootballTransformer(config)


def load_model(checkpoint_path: str, device: str = "cpu") -> FootballTransformer:
    """Load a saved checkpoint (supports both Lightning and plain torch.save formats)."""
    from fie.models.event_detection.model import FootballTransformerConfig  # noqa: F811
    import torch.serialization as _ts
    _ts.add_safe_globals([FootballTransformerConfig])

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # PyTorch Lightning checkpoint: weights are under 'state_dict'
    # with 'model.' prefix on every key.
    if "state_dict" in ckpt and "model_state_dict" not in ckpt:
        cfg = ckpt.get("hyper_parameters", {}).get("cfg", FootballTransformerConfig())
        if not isinstance(cfg, FootballTransformerConfig):
            cfg = FootballTransformerConfig()
        model = FootballTransformer(cfg)
        # Strip 'model.' prefix added by the LightningModule wrapper
        raw = ckpt["state_dict"]
        state = {k.removeprefix("model."): v for k, v in raw.items()}
        model.load_state_dict(state)
    else:
        # Plain checkpoint saved with torch.save({'model_state_dict': ..., 'config': ...})
        cfg = ckpt.get("config", FootballTransformerConfig())
        model = FootballTransformer(cfg)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model.to(device)
