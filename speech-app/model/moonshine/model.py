"""Moonshine Micro WordCNN adapter for the CMSIS ExecuTorch MLOps flow."""

from pathlib import Path
import math
import sys
import wave

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moonshine_model import WordCNN

INPUT_SHAPE = (1, 1, 64, 128)
CHECKPOINT = Path(__file__).with_name("word_cnn.pt")
CALIBRATION_ROOT = Path(__file__).resolve().parents[2] / "test" / "tts-model-vocab"


def get_model(input_shape: tuple[int, ...] = INPUT_SHAPE) -> torch.nn.Module:
    if tuple(input_shape) != INPUT_SHAPE:
        raise ValueError(f"Moonshine expects {INPUT_SHAPE}, got {input_shape}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = WordCNN(
        num_classes=len(checkpoint["classes"]),
        width_mult=float(checkpoint["args"]["width_mult"]),
        stem_stride=tuple(checkpoint["args"]["stem_stride"]),
        pad_to_odd=bool(checkpoint["args"]["pad_to_odd"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def get_example_inputs(
    input_shape: tuple[int, ...] = INPUT_SHAPE,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    return (torch.zeros(input_shape, dtype=torch.float32),), {}


def get_calibration_inputs(
    input_shape: tuple[int, ...] = INPUT_SHAPE, calibration_samples: int = 63
) -> list[tuple[tuple[torch.Tensor, ...], dict]]:
    if tuple(input_shape) != INPUT_SHAPE:
        raise ValueError(f"Moonshine expects {INPUT_SHAPE}, got {input_shape}")
    paths = sorted(CALIBRATION_ROOT.rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(
            f"no representative calibration WAVs under {CALIBRATION_ROOT}"
        )
    count = min(calibration_samples, len(paths))
    # Even spacing covers voices and labels instead of taking one directory's
    # alphabetic prefix. The generated corpus is deterministic.
    selected = [paths[(i * len(paths)) // count] for i in range(count)]
    return [((_log_mel(_read_pcm16(path)),), {}) for path in selected]


def _read_pcm16(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (16000, 1, 2):
            raise ValueError(f"{path}: expected mono PCM16/16 kHz, got {actual}")
        pcm = bytearray(wav.readframes(wav.getnframes()))
    waveform = torch.frombuffer(pcm, dtype=torch.int16).to(torch.float32) / 32768.0
    # Training uses fixed one-second clips. Front-align and zero-pad exactly as
    # the embedded path does before feature extraction.
    fixed = torch.zeros(16000, dtype=torch.float32)
    fixed[: min(16000, waveform.numel())] = waveform[:16000]
    return fixed


def _hz_to_mel(hz: float) -> float:
    return hz / (200.0 / 3.0) if hz < 1000.0 else 15.0 + math.log(hz / 1000.0) / (math.log(6.4) / 27.0)


def _mel_to_hz(mel: float) -> float:
    return (200.0 / 3.0) * mel if mel < 15.0 else 1000.0 * math.exp((math.log(6.4) / 27.0) * (mel - 15.0))


def _mel_filterbank() -> torch.Tensor:
    n_mels, n_freq = 64, 257
    lo, hi = _hz_to_mel(20.0), _hz_to_mel(8000.0)
    edges = [_mel_to_hz(lo + (hi - lo) * i / (n_mels + 1)) for i in range(n_mels + 2)]
    bins = torch.linspace(0.0, 8000.0, n_freq)
    bank = torch.zeros((n_mels, n_freq), dtype=torch.float32)
    for mel in range(n_mels):
        left, center, right = edges[mel : mel + 3]
        rising = (bins - left) / (center - left)
        falling = (right - bins) / (right - center)
        bank[mel] = torch.clamp(torch.minimum(rising, falling), min=0.0) * (2.0 / (right - left))
    return bank


def _log_mel(waveform: torch.Tensor) -> torch.Tensor:
    spectrum = torch.stft(
        waveform,
        n_fft=512,
        hop_length=125,
        win_length=512,
        window=torch.hann_window(512, periodic=True),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    mel = _mel_filterbank() @ spectrum.abs().square()
    mel = torch.log(mel + 1.0e-6)[..., :128]
    if mel.shape[-1] < 128:
        mel = torch.nn.functional.pad(mel, (0, 128 - mel.shape[-1]))
    mel = (mel - mel.mean()) / mel.std().clamp_min(1.0e-3)
    return mel.reshape(INPUT_SHAPE)
