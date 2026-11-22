#!/usr/bin/env python3
"""Build one natural, continuous WAV/SDS test session per synthetic speaker."""

from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path

VOICES = {
    "speaker-a": "zipvoice_american_female",
    "speaker-b": "zipvoice_english_male",
    "speaker-c": "zipvoice_indian_female",
}
SAMPLE_RATE = 16000
SILENCE_MS = 500
SDS_BLOCK_MS = 10
UNKNOWN_PROMPT = "banana"


def read_clip(source: Path) -> bytes:
    with wave.open(str(source), "rb") as wav:
        actual = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if actual != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{source}: expected mono PCM16/16 kHz, got {actual}")
        return wav.readframes(wav.getnframes())


def write_wav(output: Path, pcm: bytes) -> None:
    with wave.open(str(output), "wb") as wav:
        wav.setframerate(SAMPLE_RATE)
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.writeframes(pcm)


def write_sds(output: Path, pcm: bytes) -> None:
    block_bytes = SAMPLE_RATE * 2 * SDS_BLOCK_MS // 1000
    remainder = len(pcm) % block_bytes
    if remainder:
        pcm += b"\0" * (block_bytes - remainder)
    with output.open("wb") as stream:
        for index, offset in enumerate(range(0, len(pcm), block_bytes)):
            payload = pcm[offset : offset + block_bytes]
            stream.write(struct.pack("<II", index * SDS_BLOCK_MS, len(payload)))
            stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-dir", type=Path, default=Path("tts-model-vocab"))
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--words-file", type=Path, default=Path("words.txt"))
    args = parser.parse_args()

    words = args.words_file.read_text(encoding="utf-8").split()
    if len(words) != 21 or words[-1] != UNKNOWN_PROMPT:
        raise ValueError(
            f"expected 20 model commands followed by {UNKNOWN_PROMPT!r}, found {words}"
        )
    expected = [*words[:-1], "UNKNOWN"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    silence = b"\0" * (SAMPLE_RATE * 2 * SILENCE_MS // 1000)

    manifest = [
        "speaker\tvoice\tprompt_order\texpected_class_order\tsilence_ms\twav\tsds"
    ]
    for session_number, (alias, voice) in enumerate(VOICES.items(), start=1):
        parts = [silence]
        for word in words:
            source = args.tts_dir / voice / word / f"{voice}_nohash_0.wav"
            if not source.is_file():
                raise FileNotFoundError(source)
            parts.extend((read_clip(source), silence))
        pcm = b"".join(parts)
        wav_name = f"AudioIn.{session_number}.wav"
        sds_name = f"AudioIn.{session_number}.sds"
        write_wav(args.output_dir / wav_name, pcm)
        write_sds(args.output_dir / sds_name, pcm)
        manifest.append(
            f"{alias}\t{voice}\t{' '.join(words)}\t{' '.join(expected)}\t"
            f"{SILENCE_MS}\t{wav_name}\t{sds_name}"
        )

    (args.output_dir / "sessions.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"created 3 speaker sessions in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
