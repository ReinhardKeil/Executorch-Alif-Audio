# Speech SDS tests

Use SDSIO-Server with `speech.sdsio.yml`. The firmware stream name is
`AudioIn`; samples are mono signed PCM16 at 16 kHz.

The 20 model commands, followed by one out-of-vocabulary prompt, are:

`zero one two three four five six seven eight nine yes no wifi ip delete finish
cancel capital star dollar banana`

The model has no unknown class. `banana` is expected to be rejected as
`UNKNOWN` when its top softmax confidence is below the configured threshold.

Record at least three independent speakers, one session per speaker containing
all 21 prompts in that order. The configured files are
`recordings/AudioIn.1.sds`, `AudioIn.2.sds`, and `AudioIn.3.sds`, with matching
WAV files. Keep the numeric-to-speaker mapping in `recordings/sessions.tsv`;
numeric labels avoid extension filename-discovery issues and personal data.

`speech.sdsio.yml` contains three playback steps, one natural session per
speaker. Each session uses 500 ms silence before and between utterances so VAD
can segment and reset. From this directory, start a Board test with:

```powershell
sdsio-server -c speech.sdsio.yml --playback "*" --exit-after-playback
```

The Board configuration uses USB bulk; UART/V-Com remains reserved for
transcript and diagnostic output. FVP uses the SDS VSI layer directly rather
than the USB SDSIO-Server interface.

Each inference is also recorded to the `ModelOut` value-list stream described
by `ModelOut.sds.yml`. During playback SDS names these associated output files
`recordings/ModelOut.<label>.p.sds`. Every 32-byte payload contains session,
utterance, captured sample count, predicted class, expected class, pass flag,
confidence, and winning logit. Class `-1` means confidence-rejected `UNKNOWN`;
the UART log supplies the corresponding human-readable word.

The `tts-model-vocab/` directory contains 63 synthetic ZipVoice source utterances
generated with Moonshine's synthesis tool. `recordings/` exposes only three
speaker-session WAV/SDS pairs:

- `speaker-a`: `zipvoice_american_female`
- `speaker-b`: `zipvoice_english_male`
- `speaker-c`: `zipvoice_indian_female`

`wav_to_sds.py` concatenates each speaker's 21 utterances at their natural
lengths with 500 ms pauses and writes ten-millisecond SDS records.
`recordings/sessions.tsv` preserves the voice and expected word order. These
fixtures are suitable for deterministic pipeline tests, but do not replace
real-speaker acceptance data.

## Current FVP baseline

The final FVP run completed all three playback steps and produced exactly 63
inferences and three 21-record ModelOut files. The pipeline result is PASS when
all configured input is consumed, every utterance is segmented and inferred,
the output records are written, and playback closes cleanly. Classification
comparison is reported separately: the recovered 51-class checkpoint currently
passes 38 of 63 expectations (60.3%) on this independent synthetic corpus.
Do not reinterpret the 25 classification failures as an SDS/FVP failure or
silently change the expected labels to match observed output.
