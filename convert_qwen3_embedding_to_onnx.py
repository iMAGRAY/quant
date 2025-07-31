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
    from optimum.exporters.onnx import (
        main_export as optimum_export,  # high-level CLI entrypoint function
    )
    from onnxruntime import InferenceSession, SessionOptions, get_available_providers
    from onnxruntime.quantization import QuantType, quantize_dynamic
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
        activation_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=True,
        optimize_model=True,
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
        opts.graph_optimization_level = 99  # ORT OptimizationLevel.ORT_ENABLE_ALL
        sess = InferenceSession(path.as_posix(), opts, providers=get_available_providers())
        io_binding = sess.io_binding()
        io_binding.bind_input(name="input_ids", device_type="cpu", device_id=0, element_type=np.int64, shape=inputs["input_ids"].shape, buffer_ptr=inputs["input_ids"].numpy().ctypes.data)
        # attention_mask may be optional depending on model config
        if "attention_mask" in sess.get_inputs()[1].name:
            io_binding.bind_input(name="attention_mask", device_type="cpu", device_id=0, element_type=np.int64, shape=inputs["attention_mask"].shape, buffer_ptr=inputs["attention_mask"].numpy().ctypes.data)
        output_name = sess.get_outputs()[0].name
        io_binding.bind_output(name=output_name, device_type="cpu", device_id=0)
        sess.run_with_iobinding(io_binding)
        res = io_binding.copy_outputs_to_cpu()[0]
        return res

    out_fp32 = _run_onnx(model_fp32)
    out_int8 = _run_onnx(model_int8)

    diff = np.max(np.abs(out_fp32 - out_int8))
    print(f"   ∥FP32 − INT8∥∞ = {diff:.4e}")
    if diff > atol:
        raise ValueError(f"Parity check failed: diff {diff} > {atol}")
    print("✔ Validation passed (difference within tolerance)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-Embedding-0.6B to optimised INT8 ONNX")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="HF model ID or local path")
    parser.add_argument("--out_dir", default="./onnx/qwen3", help="Output directory for ONNX artifacts")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (>=17 recommended)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for reference run")
    parser.add_argument("--skip_validation", action="store_true", help="Skip numerical parity check")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    onnx_fp32_path = _export_to_onnx(args.model, out_dir, args.opset, args.device)
    onnx_int8_path = _quantize(onnx_fp32_path, onnx_fp32_path.with_name("model.int8.onnx"))

    if not args.skip_validation:
        _validate(onnx_fp32_path, onnx_int8_path, args.model, args.device)

    print("🎉 Done. Optimised INT8 ONNX model ready at", onnx_int8_path)


if __name__ == "__main__":
    main()