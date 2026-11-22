# Input report: MoonshineWordCNN

- input: `speech-app/model/moonshine/model.py`
- input type: `.py`
- execution: `CPU with Ethos-U acceleration`
- parameters: `{"calibration_samples": 63, "input_shape": [1, 1, 64, 128]}`
- optimization: `default`
- CMSIS-NN required: `no`
- methods: `forward`

## CPU fallback operators

- `quantized_decomposed::dequantize_per_tensor.out`
- `quantized_decomposed::quantize_per_tensor.out`

## Vela

- accelerator: `ethos-u55-128`
- system config: `RTSS_HE_SRAM_MRAM`
- memory mode: `Shared_Sram`

```text
Network summary for out
Accelerator configuration               Ethos_U55_128
System configuration                RTSS_HE_SRAM_MRAM
Memory mode                               Shared_Sram
Accelerator clock                                 160 MHz
Design peak SRAM bandwidth                       1.19 GB/s
Design peak Off-chip Flash bandwidth             0.36 GB/s

Total SRAM used                                172.27 KiB
Total Off-chip Flash used                     1034.25 KiB

CPU operators = 0 (0.0%)
NPU operators = 55 (100.0%)

Average SRAM bandwidth                           0.70 GB/s
Input   SRAM bandwidth                           1.64 MB/batch
Weight  SRAM bandwidth                           0.91 MB/batch
Output  SRAM bandwidth                           0.88 MB/batch
Total   SRAM bandwidth                           3.62 MB/batch
Total   SRAM bandwidth            per input      3.62 MB/inference (batch size 1)

Average Off-chip Flash bandwidth                 0.20 GB/s
Input   Off-chip Flash bandwidth                 0.00 MB/batch
Weight  Off-chip Flash bandwidth                 0.99 MB/batch
Output  Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth                 1.01 MB/batch
Total   Off-chip Flash bandwidth  per input      1.01 MB/inference (batch size 1)

Neural network macs                          35234434 MACs/batch
```
