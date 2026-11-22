#!/usr/bin/env python3
"""Recover a WordCNN checkpoint from Moonshine's fused fp32 ONNX model.

This is a provenance-preserving compatibility bridge, not an upstream artifact.
It unfolds each fused Conv+BatchNorm ONNX operator into the published PyTorch
architecture and refuses to save unless PyTorch logits match ONNX reference
logits for deterministic test inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
import torch
from torch import nn


def recover(model: nn.Module, graph: onnx.ModelProto) -> None:
    arrays = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    conv_nodes = [node for node in graph.graph.node if node.op_type == "Conv"]
    conv_modules = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    bn_modules = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    if not (len(conv_nodes) == len(conv_modules) == len(bn_modules)):
        raise RuntimeError(
            f"architecture mismatch: ONNX convs={len(conv_nodes)}, "
            f"PyTorch convs={len(conv_modules)}, batchnorms={len(bn_modules)}"
        )

    with torch.no_grad():
        for node, conv, bn in zip(conv_nodes, conv_modules, bn_modules, strict=True):
            if len(node.input) != 3:
                raise RuntimeError(f"expected fused Conv weight+bias: {node.name or node.output[0]}")
            weight = torch.from_numpy(arrays[node.input[1]].copy())
            bias = torch.from_numpy(arrays[node.input[2]].copy())
            if tuple(weight.shape) != tuple(conv.weight.shape):
                raise RuntimeError(f"shape mismatch at {node.output[0]}: {weight.shape} != {conv.weight.shape}")
            conv.weight.copy_(weight)
            # Reproduce y = conv(x) + fused_bias using an identity BatchNorm.
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.weight.fill_(math.sqrt(1.0 + bn.eps))
            bn.bias.copy_(bias)
            bn.num_batches_tracked.zero_()

        linear = model.classifier[1]
        linear.weight.copy_(torch.from_numpy(arrays["classifier.1.weight"].copy()))
        linear.bias.copy_(torch.from_numpy(arrays["classifier.1.bias"].copy()))


def verify(model: nn.Module, graph: onnx.ModelProto, input_name: str) -> tuple[float, float]:
    evaluator = ReferenceEvaluator(graph)
    rng = np.random.default_rng(0xE8)
    max_abs = 0.0
    max_rel = 0.0
    for scale in (0.0, 0.25, 1.0):
        sample = (rng.standard_normal((1, 1, 64, 128)) * scale).astype(np.float32)
        expected = evaluator.run(None, {input_name: sample})[0]
        with torch.no_grad():
            actual = model(torch.from_numpy(sample)).cpu().numpy()
        delta = np.abs(actual - expected)
        max_abs = max(max_abs, float(delta.max()))
        max_rel = max(max_rel, float((delta / np.maximum(np.abs(expected), 1.0e-6)).max()))
    return max_abs, max_rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--moonshine-training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.moonshine_training.resolve()))
    from stt_training.model import WordCNN  # pylint: disable=import-outside-toplevel

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    classes = list(metadata["classes"])
    model = WordCNN(
        num_classes=len(classes),
        stem_stride=tuple(metadata["stem_stride"]),
        pad_to_odd=bool(metadata["pad_to_odd"]),
    ).eval()
    graph = onnx.load(args.onnx)
    recover(model, graph)
    max_abs, max_rel = verify(model, graph, metadata["input_name"])
    print(f"parity max_abs={max_abs:.9g} max_rel={max_rel:.9g}")
    if max_abs > args.atol:
        raise RuntimeError(f"parity failed: max_abs {max_abs} exceeds {args.atol}")

    checkpoint = {
        "model_state": model.state_dict(),
        "classes": classes,
        "args": {
            "sample_rate": metadata["sample_rate"],
            "clip_seconds": metadata["clip_seconds"],
            "n_mels": metadata["n_mels"],
            "target_frames": metadata["target_frames"],
            "hop_length": metadata["hop_length"],
            "stem_stride": metadata["stem_stride"],
            "pad_to_odd": metadata["pad_to_odd"],
            "width_mult": 1.0,
        },
        "provenance": {
            "method": "recovered from fused fp32 ONNX with numerical parity gate",
            "source_onnx": args.onnx.name,
            "source_checkpoint_claim": metadata.get("source_checkpoint"),
            "source_run": metadata.get("source_run"),
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
