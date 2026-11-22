# SpeechApp demo

This solution is the implementation workspace for the specification in
[`../SpeechApp.md`](../SpeechApp.md). It is derived from the repository's
`cmsis-u55` template.

Targets:

- `FVP`: Corstone-300 Ethos-U55; SDS playback only.
- `Board`: AppKit-E8-AIML, M55_HE, on-board J-Link and USB V-Com UART.

The checked-in Moonshine WordCNN checkpoint is reconstructed from the pinned
upstream fp32 ONNX model by `tools/recover_moonshine_checkpoint.py`. Its
ONNX/PyTorch parity gate passed at `1.43e-6` maximum absolute logit error.
ExecuTorch 1.6 and Vela 5.2 delegate all 55 network operators to Ethos-U55-128.
Both Board and FVP images build; FVP places the model in its secondary
DDR-backed ROM region.

The final FVP regression processed all three synthetic speaker sessions:
7,014 ten-millisecond input blocks, 63 segmented utterances, 63 ModelOut
records, and clean SDS end-of-playback. The current recovered 51-class model
scores 38/63 (60.3%) on this independent corpus. This is the documented model
accuracy baseline, not a transport or runtime failure; see `../Missing.md`.

SDS 3.1 uses USB/SDSIO-Server on Board and VSI playback on FVP. Test metadata
and the 20-keyword, three-speaker coverage definition are under `test/`.

The Board target continuously captures PDM4 at 2.4 MHz/48 kHz, applies the
hardware DC blocker and 6.5 kHz FIR, then applies a 63-tap 7 kHz FIR and 3:1
software decimation before feeding 16 kHz PCM to SDS, segmentation, feature
generation, and inference. SDS recording starts only when SDSIO-Server asserts
record start. `test/recordings/AudioIn.37.sds` is the accepted live-audio
reference (703 contiguous 10 ms blocks, no gaps or clipping, quiet RMS 11.4).

Board setup is proven with SETOOLS, on-board J-Link, SW4 in U4 for runtime
UART, and the USB V-Com interface at 115200 baud.

