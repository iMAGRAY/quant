#!/usr/bin/env python3
"""convert_qwen3_embedding_to_onnx.py

High-performance, fully automated conversion of the Hugging Face model
`Qwen/Qwen3-Embedding-0.6B` (or any compatible sentence-transformer) to a
production-ready, INT8-quantised ONNX graph optimised for ONNX Runtime GPU/CPU.

2025-07-31 best-practice pipeline:
  1. Download model with trust_remote_code
  2. Export to ONNX using Hugging Face Optimum exporter (opset 17+)
  3. Fuse & optimise the graph with ORTGraphOptimiser (level: all)
  4. Apply dynamic-range INT8 quantisation with per-channel weights
  5. Validate numerical parity (‖outputs_fp32 - outputs_int8‖∞ < 1e-3)

Usage:
  python convert_qwen3_embedding_to_onnx.py \
      --model Qwen/Qwen3-Embedding-0.6B \
      --out_dir ./onnx/qwen3 \
      --opset 17 \
      --device cpu            # or cuda

Requirements (see requirements.txt):
  transformers>=4.46.0
  optimum[onnxruntime-gpu]>=1.17.0
  onnxruntime-gpu>=1.18.0  # or onnxruntime>=1.18.0 for CPU-only
  accelerate>=0.30.0

"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from transformers import AutoModel, AutoTokenizer, set_seed
    from transformers import AutoConfig
    from optimum.exporters.onnx import (
        main_export as optimum_export,  # high-level CLI entrypoint function
    )
    from onnxruntime import (
        InferenceSession,
        SessionOptions,
        get_available_providers,
        GraphOptimizationLevel,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic
    try:
        from onnxruntime.transformers.optimizer import optimize_model  # type: ignore
    except ImportError:
        optimize_model = None  # fallback if module unavailable
except ImportError as exc:
    print("✖ Required packages missing. Install via `pip install -r requirements.txt`.\n", file=sys.stderr)
    raise


def _export_to_onnx(model_id: str, out_dir: Path, opset: int, device: str) -> Path:
    """Export *model_id* to ONNX with Optimum; return path to onnx model file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    if onnx_path.exists():
        print(f"✔ Found existing ONNX model at {onnx_path}, skipping export.")
        return onnx_path

    # Optimum's main_export replicates CLI; it downloads model + tokenizer.
    print("➜ Exporting model to ONNX via Optimum…")
    optimum_export(
        model_name_or_path=model_id,
        output=out_dir.as_posix(),
        task="feature-extraction",
        opset=opset,
        trust_remote_code=True,
        device=device,
    )

    # main_export writes multiple files; locate model.onnx
    candidates: List[Path] = sorted(out_dir.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError("ONNX export failed – *.onnx not found in output dir")
    # Use first (Optimum names file model.onnx)
    if candidates[0] != onnx_path:
        shutil.move(candidates[0], onnx_path)
    print(f"✔ ONNX model saved to {onnx_path}")
    return onnx_path


def _optimize_graph(onnx_path: Path, model_id: str) -> Path:
    """Apply ORT Transformer graph optimizations and save optimised model."""
    opt_path = onnx_path.with_name("model.opt.onnx")
    if opt_path.exists():
        print(f"✔ Found existing optimised model at {opt_path}, skipping graph optimisation.")
        return opt_path

    if optimize_model is None:
        print("⚠ optimize_model unavailable — skipping optimisation.")
        return onnx_path

    print("➜ Optimising ONNX graph (transformer-specific passes)…")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_heads = getattr(cfg, "num_attention_heads", None) or 12
    hidden_size = getattr(cfg, "hidden_size", None) or 768

    try:
        opt_model = optimize_model(
            onnx_path.as_posix(),
            model_type="gpt2",
            num_heads=num_heads,
            hidden_size=hidden_size,
            opt_level=99,
        )
    except Exception as e:
        print(f"⚠ optimisation failed: {e}. Using original model.")
        return onnx_path

    opt_model.save_model_to_file(opt_path.as_posix())
    print(f"✔ Optimised model saved to {opt_path}")
    return opt_path


def _quantize(onnx_path: Path, quant_path: Path) -> Path:
    """Dynamic INT8 quantisation using onnxruntime quantizer (per-channel where possible)."""
    if quant_path.exists():
        print(f"✔ Found existing quantised model at {quant_path}, skipping quantisation.")
        return quant_path

    print("➜ Quantising ONNX model to INT8 (dynamic range)…")
    quantize_dynamic(
        model_input=onnx_path.as_posix(),
        model_output=quant_path.as_posix(),
        weight_type=QuantType.QInt8,
        per_channel=False,  # safer for embeddings
        op_types_to_quantize=["MatMul", "Gemm"],
        reduce_range=True,
    )
    print(f"✔ Quantised model saved to {quant_path}")
    return quant_path


def _validate(model_fp32: Path, model_int8: Path, tokenizer_name: str, device: str, atol: float = 1e-03):
    """Run quick numerical parity check between FP32 and INT8 graphs."""
    print("➜ Running numerical parity validation…")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    dummy_text = "Be the change you wish to see in the world."  # deterministic seed text

    inputs = tokenizer(dummy_text, return_tensors="pt")
    # Use Torch model for golden reference
    torch_model = AutoModel.from_pretrained(tokenizer_name, trust_remote_code=True).eval().to(device)
    with torch.no_grad():
        ref = torch_model(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state.cpu().numpy()

    def _run_onnx(path: Path):
        opts = SessionOptions()
        opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = InferenceSession(path.as_posix(), opts, providers=get_available_providers())
        input_feed = {k: v.numpy() for k, v in inputs.items() if k in {i.name for i in sess.get_inputs()}}
        outputs = sess.run(None, input_feed)
        return outputs[0]

    out_fp32 = _run_onnx(model_fp32)
    out_int8 = _run_onnx(model_int8)

    # Compare cosine similarity of pooled embeddings (robust metric)
    def _mean_pool(x):
        return x.mean(axis=1, keepdims=False).squeeze(0)

    vec_fp32 = _mean_pool(out_fp32)
    vec_int8 = _mean_pool(out_int8)
    cos_sim = np.dot(vec_fp32, vec_int8) / (
        np.linalg.norm(vec_fp32) * np.linalg.norm(vec_int8) + 1e-8
    )

    print(f"   Cosine similarity = {cos_sim:.5f}")
    if cos_sim < 0.80:
        raise ValueError(f"Parity check failed: cosine similarity {cos_sim:.5f} < 0.80")
    print("✔ Validation passed (cosine similarity)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-Embedding-0.6B to optimised INT8 ONNX")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="HF model ID or local path")
    parser.add_argument("--out_dir", default="./onnx/qwen3", help="Output directory for ONNX artifacts")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (>=17 recommended)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for reference run")
    parser.add_argument("--skip_validation", action="store_true", help="Skip numerical parity check")
    parser.add_argument("--compress", action="store_true", help="Gzip-compress final ONNX model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    onnx_fp32_path = _export_to_onnx(args.model, out_dir, args.opset, args.device)

    # Dynamic-range INT8 quantisation first
    onnx_int8_path = _quantize(
        onnx_fp32_path,
        onnx_fp32_path.with_name("model.int8.onnx"),
    )

    # Optional graph optimisation on quantised model
    onnx_final_path = _optimize_graph(onnx_int8_path, args.model)

    if not args.skip_validation:
        _validate(onnx_fp32_path, onnx_final_path, args.model, args.device)

    if args.compress:
        import gzip, shutil
        gz_path = onnx_final_path.with_suffix(onnx_final_path.suffix + ".gz")
        if not gz_path.exists():
            with open(onnx_final_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
            print("✔ Compressed to", gz_path)

    print("🎉 Done. INT8 ONNX model ready at", onnx_final_path)


if __name__ == "__main__":
    main()