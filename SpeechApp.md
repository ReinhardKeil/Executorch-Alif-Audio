# SpeechApp: Moonshine Micro speech demo for Alif E8 AppKit

## 1. Purpose and scope

Create a bare-metal CMSIS application for the Alif Semiconductor AppKit-E8-AIML that listens to one on-board PDM microphone on the M55_HE core, recognizes isolated spoken tokens with Moonshine Micro, and prints the recognized label and confidence over UART. The application shall also run on an Arm Corstone-300 Ethos-U55 FVP with deterministic SDS audio input.

This is an **isolated-token command recognizer**, not continuous free-form transcription. Moonshine Micro's checked-in SpellingCNN recognizes 51 isolated letters, digits, and command words from an approximately one-second, 16 kHz mono clip. The demo will expose a useful 20-keyword acceptance subset while retaining the upstream model's full label table.

The application shall use ExecuTorch from the workspace's `PyTorch::ExecuTorch@1.6.0` pack. Moonshine Micro's checked-in TFLite file is a reference artifact, not the deployed program. Recreate/load the Moonshine PyTorch architecture and weights from a pinned upstream revision, export with `torch.export`, quantize through the ExecuTorch PT2E flow, lower supported partitions to Ethos-U55 with the ExecuTorch Arm backend and Vela, and generate the embedded `.pte` AI layer with this workspace's `model_conversion/create_ai_layer.py`. The CPU/CMSIS-NN path remains the numerical reference and fallback for unsupported operations.

## 2. Selected workspace baseline

Use `cmsis-u55/cmsis-u55.csolution.yml` as the source template. It is the correct existing solution because it is the CMSIS Cortex-M plus Ethos-U55 template, already contains the Corstone-300 U55 FVP target, and has an SDS/VSI-capable FVP board layer. Do not modify the existing TinyCNN example in place. Create a sibling solution:

```text
speech-app/
|-- SpeechApp.csolution.yml
|-- SpeechApp.cproject.yml
|-- cdefault.yml
|-- README.md
|-- src/
|   |-- app_main.cpp
|   |-- audio_pipeline.cpp
|   |-- audio_pipeline.h
|   |-- recognizer.cpp
|   |-- recognizer.h
|   |-- uart_reporter.cpp
|   `-- uart_reporter.h
|-- config/
|   |-- speech_config.h
|   `-- RTE_Device_AppKit_E8_HE.h
|-- board/
|   |-- fvp/
|   |   |-- Board.clayer.yml
|   |   `-- fvp_config.txt
|   `-- appkit-e8/
|       `-- Board.clayer.yml
|-- moonshine/
|   |-- MoonshineMicro.clayer.yml
|   |-- LICENSES.md
|   `-- generated/
|       |-- model_data.cc
|       |-- model_data.h
|       |-- classes.cc
|       `-- classes.h
`-- test/
    |-- SpeechAppTest.cproject.yml
    |-- speech.sdsio.yml
    |-- manifest.yml
    |-- expected.csv
    |-- sds/
    `-- tools/
```

Vendored Moonshine sources shall be pinned to an immutable commit and retain their upstream licenses. Include only the required feature-generation, VAD/segmenter, label metadata, PyTorch model/export definitions, and test assets. Do not vendor Moonshine's TFLM runtime. TTS and Wi-Fi examples are out of scope.

## 3. CMSIS solution and targets

`SpeechApp.csolution.yml` shall declare these packs explicitly:

- `AlifSemiconductor::Ensemble` pinned to the agreed 2.3.0 release candidate.
- `ARM::CMSIS`, `ARM::CMSIS-Compiler`, `ARM::CMSIS-NN`, and `ARM::CMSIS-Ethos-U` at versions resolved and recorded by `cbuild setup`.
- `ARM::SDS@3.1.0` (using the exact installed pack identity reported by the pack catalog) and its AudioIn/SDSIO components.

Create exactly two target types:

### `FVP`

- Board/device: the existing Corstone-300 SSE-300 Ethos-U55 FVP configuration from `cmsis-u55`.
- CPU: Cortex-M55; NPU: Ethos-U55-128. Execute the Vela-lowered Moonshine `.pte`; retain a CPU build variant for parity and fallback testing.
- Input: CMSIS vStream AudioIn supplied by SDS/VSI replay.
- Output: simulator UART/stdout.
- Debugger: `Arm-FVP`, model `FVP_Corstone_SSE-300_Ethos-U55`, finite simulation limit long enough for the full test manifest.
- Supported mode: `SDS_PLAYBACK` only. The FVP shall not expose live-microphone or SDS-record modes.
- Purpose: deterministic playback tests, sanitizer-like bounds assertions where supported, and CI regression.

### `Board`

- Board: `Alif Semiconductor::AppKit-E8-AIML`.
- Device/core: `Alif Semiconductor::AE822FA0E5597BS0:M55_HE` unless the connected board's fitted device reports otherwise.
- Board layer: derive from the DFP's `Boards/AppKit-e8/Layers/M55_HE/Board_HE.clayer.yml`, retaining the AppKit clock, pin, secure-enclave initialization, UART retarget, PDM, and `CMSIS Driver:vStream:AudioIn&AppKit-E8-AIML` components.
- Input: one PDM microphone through `Driver_vStreamAudioIn`.
- Output: DFP retarget stdout over the board's USB V-Com UART, 115200-8-N-1 unless the board layer specifies a different fixed setting.
- Debugger/programming: the on-board J-Link through CMSIS Debugger, using the SETOOLS-generated device configuration.
- Hardware setup: keep SW4 in the `SE` position. Use the SETOOLS installation under `C:\Alif` for device provisioning/configuration.
- Purpose: live microphone demonstration plus the same SDS replay test mode used on FVP.

Both targets shall build `Debug` and `Release`. `Debug` enables assertions, input statistics, timing, and per-stage diagnostics. `Release` emits only startup information, recognition results, and errors.

## 4. Audio input contract

The recognizer consumes signed 16-bit, mono PCM at 16,000 samples/s.

The AppKit DFP vStream AudioIn driver captures PDM channel 4 or 5 and supports one or two channels. Configure it for mono (`AUDIO_CFG_CHANNELS=1`) on the accepted channel 4 (`AUDIO_CFG_PDM_CHANNEL=4`). Run the T5838 in its high-quality range using PDM mode 6: 2.4 MHz PDM clock, decimation 50, and 48 kHz PCM. Enable the hardware DC-blocking IIR with coefficient 4 and the measured 35-tap, 6.5 kHz hardware FIR. Low-pass filter the 48 kHz PCM with the 63-tap Q15 7 kHz application FIR and decimate 3:1 to the recognizer's 16 kHz input.

Use the interrupt-driven vStream producer and release each completed block promptly. Provision 128 raw blocks of 480 samples (1.28 seconds, 122,880 bytes) so inference and SDS control-plane latency cannot stall capture:

- One raw block is 480 samples / 10 ms / 960 bytes at 48 kHz.
- Each raw block produces exactly 160 samples / 10 ms / 320 bytes at 16 kHz after filtering and decimation.
- The ISR/vStream callback only records data availability and errors.
- A high-priority capture task calls `GetBlock`, copies the block into the application queue, then calls `ReleaseBlock` promptly.
- No feature extraction, inference, UART formatting, or allocation occurs in interrupt context.
- Overflow, end-of-stream, and driver errors are counted and printed as machine-readable errors.
- The microphone stream is not stopped while VAD, feature generation, ExecuTorch inference, UART reporting, or SDS recording is active.
- If the application queue fills, discard the oldest queued audio, increment a discontinuity counter, reset the VAD/segment under construction, and report `AUDIO_DISCONTINUITY`; never classify a segment that crosses a dropped block.

For a stereo-only fallback, select one channel while copying into the mono ring; do not average channels until polarity, gain, and phase have been measured.

Ignore the first 100 blocks (one second) for segmentation while the microphone and filters settle. Continue capture during this warm-up. Report periodic peak and RMS diagnostics; warn but do not prevent operation unless the stream is silent or saturated.

### Runtime modes

The same firmware image shall support these mutually exclusive audio-source modes, selected at startup by build default plus an SDS user flag/control setting:

- **`LIVE_CONTINUOUS` (default on Board):** initialize and start vStream AudioIn once, remain continuously connected to the microphone, and run repeated VAD → recognition cycles indefinitely. After each result, return immediately to listening without stopping or reinitializing PDM. The capture task continues filling the PCM queue while inference runs.
- **`SDS_RECORD` (Board only):** keep continuous microphone capture and inference active, and tee every raw PCM block to an SDS output stream. SDSIO-Server controls start/stop of persistence; firmware does not invent a separate record protocol. A slow or disconnected SDS sink may drop recording data with an SDS error counter but must not block microphone capture or inference.
- **`SDS_PLAYBACK` (Board and FVP):** feed recorded PCM from the SDS input stream into the identical PCM queue used by live capture. On Board, the physical microphone path shall not be mixed with playback; it may remain initialized but its blocks are discarded while playback owns the algorithm input. On FVP this is the normal input mode.

Mode transitions occur only at a segment boundary. On transition, flush the application PCM queue, reset feature/VAD state, close the previous SDS stream as required, and emit a `MODE` UART record. A transition must not require resetting the target.

## 5. Recognition pipeline

The live pipeline is:

```text
PDM4 microphone -> 2.4 MHz/48 kHz vStream AudioIn
 -> 6.5 kHz hardware FIR + DC blocker
 -> 7 kHz application FIR -> decimate 3:1 -> 16 kHz mono blocks
 -> 512-sample log-mel hops -> TinyVadCNN -> segmenter
 -> approximately 1 s speech clip -> 64 x 128 normalized log-mel
 -> int8 SpellingCNN -> logits -> argmax/softmax -> UART
```

Use Moonshine Micro's feature-generation constants and generated tables unchanged. The VAD runs every 32 ms. Preserve pre-roll and trailing silence according to the upstream `VadSegmenter`; pad or crop the final clip exactly as the upstream RP2350 example does so desktop, FVP, and board results remain comparable.

Use the workspace's embedded ExecuTorch module wrapper and a statically provisioned method allocator/memory pool placed in HE-accessible SRAM. Size it from the generated ExecuTorch/Vela reports and measured peak use, then retain at least 10% headroom. Audio rings, feature buffers, stacks, UART buffers, `.pte` program data, Ethos-U command streams, and tensor arenas shall be reported separately. Place the embedded `.pte` and weights in MRAM or directly readable external flash according to the SETOOLS memory map.

The exported graphs are expected to contain convolution/depthwise convolution, add, reduction/global-pool, linear, reshape/view, padding, and quantize/dequantize edges. Before firmware integration, generate an operator/delegation report for both SpellingCNN and TinyVadCNN. The build shall fail if an unapproved portable fallback operator appears, an expected NPU partition is not delegated, or Vela rejects a graph. Startup shall fail clearly on program verification, method lookup, input-shape, allocator, or Ethos-U driver errors.

Keep log-mel feature generation and VAD segmentation in C/C++; the TinyVadCNN and SpellingCNN neural networks are separate methods/programs in one generated AI layer so they share one ExecuTorch runtime. Invoke VAD once per 32 ms hop and STT once per completed speech segment. The application must not run both methods concurrently, allowing planned memory reuse.

Use a confidence threshold configurable in `speech_config.h`. A result below the threshold is reported as `UNKNOWN`, not forced to the highest-scoring keyword. Record top-1 and top-3 scores in test builds.

## 6. Demo vocabulary and UART protocol

The recovered WordCNN has 51 spelling classes and no dedicated unknown class. The selected acceptance prompts are:

```text
zero, one, two, three, four, five, six, seven, eight, nine,
yes, no, wifi, ip, delete, finish, cancel, capital, star, dollar,
banana (expected result: UNKNOWN by confidence rejection)
```

`banana` is deliberately outside the vocabulary and tests confidence-based rejection. Because the published model has no unknown class, the rejection threshold must be calibrated on a development corpus. A dedicated unknown class requires model retraining.

UART output shall be line-oriented UTF-8 and stable enough for automation:

```text
SPEECHAPP READY target=Board core=M55_HE rate=16000 channel=4
MODE mode=LIVE_CONTINUOUS source=MIC
VAD START sample=123456
VAD END samples=16000
RESULT label=seven confidence=0.934 latency_ms=87
RESULT label=UNKNOWN confidence=0.312 latency_ms=86
ERROR code=AUDIO_OVERFLOW count=1
TEST name=spk03_seven_02 expected=seven actual=seven confidence=0.917 PASS
SUMMARY total=120 pass=114 fail=6 accuracy=95.00
```

UART writes shall be buffered and shall never stall the audio ISR. Live mode may suppress VAD lines in `Release`.

## 7. SDS test assets

SDS 3.1.0 is the canonical record/replay path. Integrate an SDSIO-Client in the Board firmware and use SDSIO-Server on the host to control recording and playback. Prefer the SDS USB bulk interface on AppKit-E8 so SDS binary traffic is independent of the UART used for transcript text. Use serial, RTT, or network only as a documented fallback after throughput and coexistence testing. FVP uses the SDS FVP/VSI3 interface strictly for playback and the same control file; its `interface:` node is ignored by the simulator as defined by SDS. FVP shall never create or record SDS audio files.

Define two streams with metadata:

- `AudioIn`: mono PCM16, 16 kHz, 160-sample/10 ms records after board filtering and decimation; recorded from the microphone and consumed during playback.
- `SpeechResult`: timestamp/sample index, expected/actual class, confidence, latency, mode, overflow/discontinuity counters, and pass/fail fields for correlation with audio.

Each test recording shall be stored as mono PCM16 at 16 kHz with a manifest entry containing file, speaker ID, expected label, source/license, duration, peak level, and SHA-256. Use SDS filenames of the form `<stream-name>.<label>[.p].sds` and an `AudioIn.sds.yml` metadata file. Keep recordings out of generated C source.

Minimum useful corpus:

- 20 selected model words plus the `banana` unknown-word prompt listed above.
- At least 6 speakers, with a balance of voices/accents where available.
- At least 2 utterances per keyword from different speakers for the smoke set (40 positive clips).
- Preferred qualification set: 3 utterances per keyword from at least 3 different speakers (60 positive clips).
- At least 20 negative clips: silence, background noise, non-keywords, and ordinary connected speech.
- No speaker may appear in only one class; distribute speakers across the vocabulary to avoid a speaker/class confound.
- Normalize file format, not loudness. Preserve realistic level variation and document any gain operation.

`test/speech.sdsio.yml` shall configure the board interface, `workdir`, `metadir`, stream views, and ordered `play:` steps. Each playback step shall add or reference 250 ms of leading silence and 500 ms of trailing silence so VAD behavior is tested, not bypassed. On Board, use SDSIO-Server's standard controls: `R` starts recording, `P` starts playback, and `S` stops either operation. FVP accepts playback only. Automated FVP runs use `--playback "*" --exit-after-playback --log <file>` and shall not depend on interactive keystrokes.

Playback shall reset streaming/VAD state at each configured step, replay the selected labels, wait for exactly one final result per positive clip, and compare results with `expected.csv`. Recording and playback must also be exercised for a continuous multi-token stream to prove that the microphone/replay source stays open across successive recognitions.

Three synthetic speaker sessions are present under `test/recordings`; each contains all 20 selected words plus `banana`, with 500 ms pauses, and has both WAV and SDS forms. They are functional fixtures, not a substitute for consented human-speaker acceptance recordings.

## 8. Test plan and acceptance criteria

### Host tests

- Upstream-equivalent tests for `Argmax`, ties, stable softmax, VAD segmentation, pre-roll, trailing flush, and clip alignment.
- PCM conversion, mono extraction, ring wrap, overflow accounting, and SDS manifest parsing.
- Golden feature comparison against Moonshine desktop output within specified floating-point tolerance.
- Model metadata and class-table consistency checks.

### FVP tests

- Build `FVP.Debug` and replay the entire SDS smoke set.
- Verify one result per positive clip, `UNKNOWN` for negatives as appropriate, no stream overflows, and no arena or stack corruption.
- Compare model logits or top-3 output against the desktop TFLite reference for at least one clip per keyword.
- Preserve UART log, build diagnostics, memory report, and test summary as artifacts.

### Board replay tests

- Configure and program the M55_HE image with SETOOLS/CMSIS Debugger.
- Replay the same SDS smoke set through the board test input and compare results with FVP.
- Require bit-identical int8 classification where the feature input is identical; otherwise diagnose feature or runtime differences before relaxing tolerance.
- Run for at least 30 minutes with repeated clips and require zero unhandled audio overflows and zero hard faults.
- Run a continuous-capture stress test in which microphone capture remains started during every inference; require no vStream restart and no unreported discontinuity.

### Board live tests

- Verify microphone channel, signal level, and UART independently before enabling inference.
- Each of the 20 selected words and at least one unknown word shall be spoken by at least three people, three trials each, at approximately 0.3-1.0 m in a quiet room.
- Record a confusion matrix, rejection rate, median/P95 inference latency, and audio overflow count.
- Initial demo gate: at least 90% top-1 accuracy over the positive live trials, at least 90% rejection of negative trials, no crash or hang, and P95 final-result latency under 750 ms after VAD end. If upstream model accuracy makes this unattainable for particular voices, report the measured result rather than tuning on the evaluation set.

### Resource gates

- Link succeeds without region overflow for `FVP` and `Board` in Debug and Release.
- Measured arena plus margin fits the chosen HE SRAM region.
- Stack high-water marks retain at least 25% margin.
- Model and code fit the selected MRAM/external-flash execution layout.

## 9. Implementation sequence

1. Record exact pack/tool/compiler versions and the Moonshine commit; resolve every item in `Missing.md` that affects target identity or memory layout.
2. Copy the `cmsis-u55` template into `speech-app`, rename contexts, add `FVP` and `Board` target types, and select an active target so CDA has a project/target context.
3. Import `AUGD0028-Ensemble-E8-AI-ML-AppKit-AK-E8-User-Guide-v1.1.pdf` and `Alif_E8_HWRM_v1.0.pdf` with **CMSIS Developer Assistant: Import Document for Current Target**. Verify both with `list_target_docs` and a known `search_target_docs` query; never parse the PDFs directly in the firmware workflow.
4. With SW4 in `SE`, use SETOOLS from `C:\Alif` to configure the AppKit for M55_HE, USB V-Com UART, PDM microphone, required clocks, MRAM/external memory, on-board J-Link debug access, and NPU access. Export and archive the configuration inputs and generated artifacts.
5. Prove the unchanged TinyCNN ExecuTorch baseline on `FVP`, then build a UART-only application for both targets.
6. Bring up vStream AudioIn using `ARM::SDS@3.1.0`: SDS tone/ramp on FVP, then continuous live PDM on Board. Add the non-blocking PCM queue and prove that the microphone remains started during artificial worst-case inference delays.
7. Pin Moonshine, reproduce the upstream PyTorch models and reference logits, and add host unit tests plus the license inventory.
8. Export and quantize TinyVadCNN and SpellingCNN to `.pt2`; compare PyTorch/quantized outputs against upstream TFLite references on the same feature tensors.
9. Add both inputs to `mlops.model.input`, generate the ExecuTorch AI layer, inspect Vela/delegation reports, and require approved operator coverage.
10. Integrate C/C++ feature generation, ExecuTorch TinyVadCNN, and the segmenter; verify SDS speech boundaries, continuous multi-token operation, and desktop/FVP parity.
11. Integrate ExecuTorch SpellingCNN and its 51 labels; measure allocator, stack, NPU latency, and memory placement on FVP and Board.
12. Add the stable UART protocol, 20-keyword filter/acceptance table, confidence rejection, and test runner.
13. Collect/approve the multi-speaker corpus, generate SDS assets and manifests, and run desktop/FVP/board replay parity.
14. Execute live acceptance, capture results, tune only documented VAD/confidence parameters on a separate development set, and freeze a reproducible demo release.

Each step has a runnable exit condition and shall be committed separately. Do not combine board bring-up, audio capture, and model integration into the first hardware image.

## 10. ExecuTorch model-conversion gate

ExecuTorch/Ethos-U55 is the required deployment path, but it is a gated engineering task rather than a file-format rename. The gate passes only when both models can be recreated in PyTorch with the upstream weights, their pre-quantization logits match the Moonshine reference, PT2E quantization meets the agreed accuracy tolerance, Vela lowers the intended partitions for U55-128, and FVP output matches the quantized desktop reference. If either graph cannot meet those conditions, stop and document the offending operator or numerical mismatch; do not silently fall back to TFLM. A CPU ExecuTorch/CMSIS-NN context may remain as a diagnostic/reference target.

## 11. Deliverables

- `speech-app/SpeechApp.csolution.yml` with `FVP` and `Board` targets.
- Board/FVP layers, application sources, pinned Moonshine model/export integration, generated ExecuTorch AI layer, and license inventory.
- SETOOLS configuration recipe/artifacts for the exact connected board.
- SDS corpus manifest and replay assets or approved external artifact references.
- SDSIO-Client integration plus `speech.sdsio.yml` for SDSIO-Server-controlled recording and playback.
- Automated host, FVP, board-replay, and live-test instructions.
- UART logs, confusion matrix, latency/resource reports, and a reproducible release record.
- Updated `Missing.md`, empty of blocking items at release time or carrying explicit accepted deviations.

## 12. Upstream references

- Moonshine Micro overview: https://github.com/moonshine-ai/moonshine/blob/main/micro/README.md
- Moonshine Micro STT: https://github.com/moonshine-ai/moonshine/blob/main/micro/stt/README.md
- Moonshine Micro models: https://github.com/moonshine-ai/moonshine/blob/main/micro/models/README.md
- Moonshine Micro VAD: https://github.com/moonshine-ai/moonshine/blob/main/micro/vad/README.md
- Workspace baseline: `cmsis-u55/cmsis-u55.csolution.yml`

