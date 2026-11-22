#!/usr/bin/env python3
"""Evaluate the fp32 checkpoint on the deterministic SDS source corpus."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "speech-app" / "model" / "moonshine" / "model.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("speech_model", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.get_model()
    checkpoint = torch.load(module.CHECKPOINT, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]

    passed = known_passed = unknown_passed = count = 0
    for path in sorted(module.CALIBRATION_ROOT.rglob("*.wav")):
        features = module._log_mel(module._read_pcm16(path))
        with torch.no_grad():
            probabilities = torch.softmax(model(features), dim=-1)[0]
        confidence, index = probabilities.max(dim=0)
        predicted = classes[int(index)] if float(confidence) >= 0.5 else "UNKNOWN"
        expected_word = path.parent.name
        expected = "UNKNOWN" if expected_word == "banana" else expected_word
        ok = predicted == expected
        count += 1
        passed += int(ok)
        known_passed += int(ok and expected != "UNKNOWN")
        unknown_passed += int(ok and expected == "UNKNOWN")
        if not ok:
            print(
                f"{path.parts[-3]} {expected_word} -> {predicted} "
                f"confidence={float(confidence):.4f}"
            )

    print(
        f"FP32 total={passed}/{count} known={known_passed}/{count - 3} "
        f"unknown={unknown_passed}/3"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
