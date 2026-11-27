"""Export a ~1.85M-parameter dummy 1D-CNN + Transformer model to ONNX for browser latency probes."""

import argparse
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn

from learnsign.features.derived import DERIVED_FEATURES_PER_FRAME
from learnsign.features.schema import FEATURES_PER_FRAME

DEFAULT_PARAMS_TARGET: Final[int] = 1_850_000
MIN_EXPECTED_PARAMS: Final[int] = 1_700_000
MAX_EXPECTED_PARAMS: Final[int] = 2_000_000
DEFAULT_WINDOW_FRAMES: Final[int] = 64
DEFAULT_NUM_CLASSES: Final[int] = 250
DEFAULT_BENCH_FEATURES: Final[int] = FEATURES_PER_FRAME + DERIVED_FEATURES_PER_FRAME
DEFAULT_EXPORT_SEED: Final[int] = 42

DEFAULT_D_MODEL: Final[int] = 192
DEFAULT_NUM_HEADS: Final[int] = 6
DEFAULT_FFN_DIM: Final[int] = 568
DEFAULT_NUM_LAYERS: Final[int] = 4
ONNX_OPSET_VERSION: Final[int] = 18


class _AttentionEncoderBlock(nn.Module):
    """Pre-norm self-attention + FFN block implemented with explicit ONNX-friendly ops."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and feedforward residual blocks to `(B, T, D)` tensor."""
        batch_size, seq_len, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        x = x + self.out_proj(context)
        return x + self.ffn(self.norm2(x))


class DummySignModel(nn.Module):
    """1D-CNN + Transformer probe architecture sized to ~1.85M parameters."""

    def __init__(
        self,
        in_features: int = DEFAULT_BENCH_FEATURES,
        num_classes: int = DEFAULT_NUM_CLASSES,
        d_model: int = DEFAULT_D_MODEL,
        num_heads: int = DEFAULT_NUM_HEADS,
        ffn_dim: int = DEFAULT_FFN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_features, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            *[_AttentionEncoderBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass on `(B, T, F)` sequence and return `(B, num_classes)` logits."""
        # Conv1d operates on (B, F, T) channels-first tensors.
        h = self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class _Fp16IoWrapper(nn.Module):
    """Cast float32 browser/test inputs into fp16 weights and return float32 logits."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.to(torch.float16)).to(torch.float32)


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count for `model`."""
    return sum(p.numel() for p in model.parameters())


def _select_ffn_dim_for_target(
    params_target: int,
    in_features: int,
    num_classes: int,
) -> int:
    """Solve `ffn_dim` linearly so total params land close to `params_target`."""
    probe = DummySignModel(
        in_features=in_features,
        num_classes=num_classes,
        d_model=DEFAULT_D_MODEL,
        num_heads=DEFAULT_NUM_HEADS,
        ffn_dim=1,
        num_layers=DEFAULT_NUM_LAYERS,
    )
    base_params = count_parameters(probe)
    # Each FFN layer adds (2 * d_model + 1) params per hidden unit (ffn_dim=1 is already included).
    params_per_ffn_unit = DEFAULT_NUM_LAYERS * (2 * DEFAULT_D_MODEL + 1)
    ffn_dim = max(64, round(1 + (params_target - base_params) / params_per_ffn_unit))
    return ffn_dim


def export_dummy_onnx(
    path: str | Path,
    params_target: int = DEFAULT_PARAMS_TARGET,
    window: int = DEFAULT_WINDOW_FRAMES,
    features: int = DEFAULT_BENCH_FEATURES,
    seed: int = DEFAULT_EXPORT_SEED,
    fp16: bool = True,
) -> int:
    """Build a ~`params_target` 1D-CNN + Transformer model, export ONNX, and return param count."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    ffn_dim = _select_ffn_dim_for_target(
        params_target=params_target,
        in_features=features,
        num_classes=DEFAULT_NUM_CLASSES,
    )
    core_model = DummySignModel(
        in_features=features,
        num_classes=DEFAULT_NUM_CLASSES,
        d_model=DEFAULT_D_MODEL,
        num_heads=DEFAULT_NUM_HEADS,
        ffn_dim=ffn_dim,
        num_layers=DEFAULT_NUM_LAYERS,
    )
    core_model.eval()
    param_count = count_parameters(core_model)

    export_model: nn.Module = _Fp16IoWrapper(core_model).eval() if fp16 else core_model
    dummy_input = torch.randn(1, window, features, dtype=torch.float32)

    # Why external_data=False: keeps the entire model inside one static `.onnx` file for
    # Cloudflare Pages hosting and simple fetch-by-URL in onnxruntime-web.
    torch.onnx.export(
        export_model,
        (dummy_input,),
        str(out_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=True,
        dynamo=True,
        external_data=False,
    )
    return param_count


def main() -> None:
    """CLI entrypoint to regenerate the static dummy ONNX probe file."""
    parser = argparse.ArgumentParser(description="Export dummy Step 0 latency-probe ONNX model.")
    parser.add_argument("--out", required=True, help="Output path for the exported .onnx file.")
    parser.add_argument("--params", type=int, default=DEFAULT_PARAMS_TARGET)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--features", type=int, default=DEFAULT_BENCH_FEATURES)
    parser.add_argument("--seed", type=int, default=DEFAULT_EXPORT_SEED)
    parser.add_argument("--fp32", action="store_true", help="Export fp32 instead of fp16.")
    args = parser.parse_args()

    params = export_dummy_onnx(
        args.out,
        params_target=args.params,
        window=args.window,
        features=args.features,
        seed=args.seed,
        fp16=not args.fp32,
    )
    size_bytes = Path(args.out).stat().st_size
    print(f"Exported {args.out}: params={params:,}, bytes={size_bytes:,}")


if __name__ == "__main__":
    main()
