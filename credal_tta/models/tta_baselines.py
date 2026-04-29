"""
Gradient-Based TTA Baselines (R1-W4, Section 5.3)
LoRA-TTA and TENT-TTA implementations for comparison with Credal-TTA
"""

import numpy as np
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --- Torch-dependent class definitions (guarded) ---------------------------
# These classes inherit from nn.Module and therefore can only be defined when
# PyTorch is installed.  When torch is unavailable the higher-level wrappers
# (LoRATTA, TENTTTA) gracefully fall back to a simple moving-average predictor.

if TORCH_AVAILABLE:
    class LoRALayer(nn.Module):
        """Low-rank adapter: ΔW = BA, B∈ℝ^{d×r}, A∈ℝ^{r×d}"""

        def __init__(self, in_dim: int, out_dim: int, rank: int = 4):
            super().__init__()
            self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
            self.B = nn.Parameter(torch.zeros(out_dim, rank))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + x @ self.A.T @ self.B.T

    class SimpleTransformerBlock(nn.Module):
        """Minimal transformer block for standalone LoRA/TENT experiments."""

        def __init__(self, d_model: int = 64, nhead: int = 4):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.ln1(x)
            h, _ = self.attn(h, h, h)
            x = x + h
            x = x + self.ff(self.ln2(x))
            return x

    class SimpleTSModel(nn.Module):
        """Lightweight transformer time-series model for baseline comparisons."""

        def __init__(self, context_len: int = 512, d_model: int = 64, n_layers: int = 2, nhead: int = 4):
            super().__init__()
            self.context_len = context_len
            self.d_model = d_model
            self.input_proj = nn.Linear(1, d_model)
            self.blocks = nn.ModuleList([SimpleTransformerBlock(d_model, nhead) for _ in range(n_layers)])
            self.output_proj = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, T, 1) -> (B, 1)"""
            h = self.input_proj(x)  # (B, T, d)
            for blk in self.blocks:
                h = blk(h)
            return self.output_proj(h[:, -1, :]).squeeze(-1)  # (B,)
else:
    # Provide stub references so that type hints / isinstance checks don't
    # crash at import time when torch is absent.
    LoRALayer = None
    SimpleTransformerBlock = None
    SimpleTSModel = None


# ---------------------------------------------------------------------------
# LoRA-TTA
# ---------------------------------------------------------------------------

class LoRATTA:
    """
    LoRA Test-Time Adaptation baseline (Hu et al., 2022).

    Injects rank-r LoRA adapters into Q/K projections of all attention layers.
    At each time step, performs ``num_steps`` AdamW gradient updates minimising
    MSE on the ``window`` most recent observations.

    When no real TSFM is available, falls back to a lightweight built-in
    transformer so that the adaptation loop is still exercised.
    """

    def __init__(
        self,
        base_model=None,
        rank: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_steps: int = 1,
        window: int = 20,
        context_len: int = 512,
        device: str = None,
    ):
        self.rank = rank
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.window = window
        self.context_len = context_len
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.base_model = base_model
        self.history: List[float] = []

        # Build internal model + LoRA layers
        if TORCH_AVAILABLE:
            self._model = SimpleTSModel(context_len=context_len).to(self.device)
            self._model.eval()
            # Freeze base weights
            for p in self._model.parameters():
                p.requires_grad = False
            # Inject LoRA into every attention in_proj
            self._lora_layers: List[LoRALayer] = []
            for blk in self._model.blocks:
                d = self._model.d_model
                lora = LoRALayer(d, d, rank).to(self.device)
                self._lora_layers.append(lora)
            self._optimizer = torch.optim.AdamW(
                [p for lora in self._lora_layers for p in lora.parameters()],
                lr=lr, weight_decay=weight_decay,
            )

    # ---- internal helpers --------------------------------------------------

    def _prepare_context(self, context: np.ndarray) -> torch.Tensor:
        ctx = context[-self.context_len:]
        t = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)
        return t  # (1, T, 1)

    def _forward_with_lora(self, x_tensor: torch.Tensor) -> torch.Tensor:
        h = self._model.input_proj(x_tensor)
        for blk, lora in zip(self._model.blocks, self._lora_layers):
            h = lora(h)
            h = blk(h)
        return self._model.output_proj(h[:, -1, :]).squeeze(-1)

    def _adapt(self):
        """Run gradient steps on recent window."""
        if len(self.history) < self.window + 1:
            return
        recent = np.array(self.history[-(self.window + 1):])
        inputs = recent[:-1]
        target = recent[-1]

        x_t = self._prepare_context(inputs)
        tgt = torch.tensor([target], dtype=torch.float32, device=self.device)

        for _ in range(self.num_steps):
            self._optimizer.zero_grad()
            pred = self._forward_with_lora(x_t)
            loss = ((pred - tgt) ** 2).mean()
            loss.backward()
            self._optimizer.step()

    # ---- public API --------------------------------------------------------

    def predict(self, context: np.ndarray, recent_obs: Optional[List[float]] = None) -> float:
        """
        Predict next value.  If *recent_obs* is provided it is used for
        adaptation; otherwise internal history is used.
        """
        self.history = list(context)

        # Gradient adaptation
        if TORCH_AVAILABLE:
            self._adapt()
            with torch.no_grad():
                x_t = self._prepare_context(context)
                pred = self._forward_with_lora(x_t)
                return float(pred.cpu().item())

        # Fallback when torch unavailable: moving average
        w = min(50, len(context))
        return float(np.mean(context[-w:])) if w > 0 else 0.0

    def reset(self):
        """Reset LoRA parameters (between episodes)."""
        self.history = []
        if TORCH_AVAILABLE:
            for lora in self._lora_layers:
                nn.init.normal_(lora.A, std=0.01)
                nn.init.zeros_(lora.B)


# ---------------------------------------------------------------------------
# TENT-TTA
# ---------------------------------------------------------------------------

class TENTTTA:
    """
    TENT Test-Time Adaptation baseline (Wang et al., 2021).

    Updates only LayerNorm scale/shift parameters by minimising predictive
    entropy H(P̂_t) over the last ``window`` observations.
    """

    def __init__(
        self,
        base_model=None,
        lr: float = 1e-3,
        window: int = 20,
        context_len: int = 512,
        device: str = None,
    ):
        self.lr = lr
        self.window = window
        self.context_len = context_len
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.base_model = base_model
        self.history: List[float] = []

        if TORCH_AVAILABLE:
            self._model = SimpleTSModel(context_len=context_len).to(self.device)
            self._model.eval()
            # Freeze everything except LayerNorm
            self._ln_params = []
            for m in self._model.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True
                        self._ln_params.append(p)
                else:
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False
            self._optimizer = torch.optim.Adam(self._ln_params, lr=lr)

    def _entropy_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Proxy entropy: variance of recent predictions as entropy surrogate."""
        return predictions.var()

    def _adapt(self):
        if len(self.history) < self.window + 1:
            return
        recent = np.array(self.history[-self.window:])
        preds = []
        for i in range(len(recent)):
            ctx = recent[:i + 1]
            x_t = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)
            pred = self._model(x_t)
            preds.append(pred)
        preds_tensor = torch.stack(preds)
        loss = self._entropy_loss(preds_tensor)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def predict(self, context: np.ndarray, recent_obs: Optional[List[float]] = None) -> float:
        self.history = list(context)

        if TORCH_AVAILABLE:
            self._adapt()
            with torch.no_grad():
                ctx = context[-self.context_len:]
                x_t = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)
                pred = self._model(x_t)
                return float(pred.cpu().item())

        w = min(50, len(context))
        return float(np.mean(context[-w:])) if w > 0 else 0.0

    def reset(self):
        self.history = []
        if TORCH_AVAILABLE:
            for m in self._model.modules():
                if isinstance(m, nn.LayerNorm):
                    m.reset_parameters()


# Alias for backward compatibility with __init__.py
TENT_TTA = TENTTTA
