// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Headless ExecuTorch runner for a Cortex-M or Ethos-U application. Loads the .pte
// embedded by the AI layer (MoonshineWordCNN_pte.h), runs one
// inference, and prints the output logits. Built entirely from the
// PyTorch::ExecuTorch pack's runtime + operator components; the board layer
// provides main(), stdout and the Ethos-U driver init and then calls app_main.
//
// EmbeddedModule (arm_embedded_module.hpp) manages program loading, method
// memory and execution, like ExecuTorch's Module class does on POSIX hosts.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>

#if defined(SPEECHAPP_FVP) || defined(SPEECHAPP_BOARD)
#include "cmsis_os2.h"
#endif

#if defined(SPEECHAPP_FVP) || defined(SPEECHAPP_BOARD)
#include "sds.h"
#endif

#if defined(SPEECHAPP_BOARD)
#include "cmsis_vstream.h"
#include CMSIS_target_header
extern "C" int32_t speechapp_pdm_configure(uint32_t mode, uint32_t channel,
                                             uint32_t iir_coefficient);
#endif

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "audio_config.h"
#include "feature_generation/feature_generation.h"
#include "mel_tables.h"
#include "MoonshineWordCNN_pte.h"

#define APP_STRINGIFY_IMPL(value) #value
#define APP_STRINGIFY(value) APP_STRINGIFY_IMPL(value)

#if defined(ETHOSU55)
#define APP_ETHOS_U_NAME "Ethos-U55"
#elif defined(ETHOSU65)
#define APP_ETHOS_U_NAME "Ethos-U65"
#elif defined(ETHOSU85)
#define APP_ETHOS_U_NAME "Ethos-U85"
#endif

#ifdef APP_ETHOS_U_NAME
#ifdef ETHOSU_MACS
#define APP_TARGET_CONFIG APP_ETHOS_U_NAME "-" APP_STRINGIFY(ETHOSU_MACS)
#else
#define APP_TARGET_CONFIG APP_ETHOS_U_NAME
#endif
#else
#define APP_TARGET_CONFIG "Cortex-M55"
#endif

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

namespace {

constexpr const char* kWordLabels[] = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "capital", "uppercase", "star", "dollar", "underscore",
    "exclamation", "percent", "dash", "delete", "finish", "cancel", "wifi",
    "ip", "yes", "no",
};
constexpr size_t kWordLabelCount = sizeof(kWordLabels) / sizeof(kWordLabels[0]);
constexpr const char* kExpectedPrompts[] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "yes", "no", "wifi", "ip", "delete", "finish", "cancel", "capital",
    "star", "dollar", "UNKNOWN",
};
constexpr int kExpectedClass[] = {
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 49, 50, 47, 48, 44, 45, 46, 36, 38, 39, -1,
};
constexpr size_t kExpectedPromptCount =
    sizeof(kExpectedPrompts) / sizeof(kExpectedPrompts[0]);
constexpr float kUnknownConfidenceThreshold = 0.50f;

// Pool sizes and placement are board-overridable: a board layer that cannot
// fit 8 MB of .bss in its default RAM sets APP_*_POOL_SIZE and, via
// APP_POOL_SECTION, the linker section its linker script routes to a larger
// (NPU-accessible) memory.
#ifndef APP_METHOD_POOL_SIZE
#define APP_METHOD_POOL_SIZE (4 * 1024 * 1024)
#endif
#ifndef APP_TEMP_POOL_SIZE
#define APP_TEMP_POOL_SIZE (4 * 1024 * 1024)  // Ethos-U scratch is drawn from here.
#endif
#ifdef APP_POOL_SECTION
#define APP_POOL_ATTRIBUTES __attribute__((section(APP_POOL_SECTION)))
#else
#define APP_POOL_ATTRIBUTES
#endif

constexpr size_t kMethodPoolSize = APP_METHOD_POOL_SIZE;
constexpr size_t kTempPoolSize = APP_TEMP_POOL_SIZE;

alignas(16) uint8_t g_method_pool[kMethodPoolSize] APP_POOL_ATTRIBUTES;
alignas(16) uint8_t g_temp_pool[kTempPoolSize] APP_POOL_ATTRIBUTES;

alignas(16) float g_log_mel[spelling::kNMels * spelling::kTargetFrames];
alignas(4) int16_t g_speech_clip[spelling::kClipNumSamples];

#if defined(SPEECHAPP_FVP)
struct ModelOutRecord {
  uint32_t session;
  uint32_t utterance;
  uint32_t captured_samples;
  int32_t predicted_class;
  int32_t expected_class;
  uint32_t pass;
  float confidence;
  float logit;
};
static_assert(sizeof(ModelOutRecord) == 32U, "ModelOut SDS metadata contract changed");

sdsId_t g_model_out = nullptr;

bool record_model_out(uint32_t timeslot, const ModelOutRecord& record) {
  if (g_model_out == nullptr) return false;
  int32_t status;
  do {
    status = sdsWrite(g_model_out, timeslot, &record, sizeof(record));
    if (status == SDS_NO_SPACE) osDelay(1U);
  } while (status == SDS_NO_SPACE);
  if (status != static_cast<int32_t>(sizeof(record))) {
    printf("SPEECH SDS write=FAIL stream=ModelOut status=%ld\n",
           static_cast<long>(status));
    return false;
  }
  return true;
}
#endif

spelling::LogMelParams make_frontend_params() {
  spelling::LogMelParams params;
  params.sample_rate = spelling::kSampleRate;
  params.n_fft = spelling::kNFft;
  params.hop_length = spelling::kHopLength;
  params.win_length = spelling::kWinLength;
  params.n_mels = spelling::kNMels;
  params.f_min = spelling::kFMin;
  params.f_max = spelling::kFMax;
  params.target_frames = spelling::kTargetFrames;
  params.precomputed_window = spelling::kMelWindow;
  params.precomputed_window_len = spelling::kMelTableNFft;
  params.precomputed_nz_off = spelling::kMelNzOff;
  params.precomputed_nz_idx = spelling::kMelNzIdx;
  params.precomputed_nz_val = spelling::kMelNzVal;
  params.precomputed_nz_total = spelling::kMelNzTotal;
  return params;
}

bool infer_clip(EmbeddedModule& module, spelling::LogMelSpectrogram& frontend,
                const int16_t* pcm, size_t samples, size_t captured_samples,
                uint32_t session, uint32_t utterance, uint32_t timeslot,
                uint32_t& correct) {
  frontend.Compute(pcm, samples, g_log_mel);

  std::array<SizesType, 4> sizes{1, 1, spelling::kNMels, spelling::kTargetFrames};
  std::array<DimOrderType, 4> dim_order{0, 1, 2, 3};
  TensorImpl input_impl(ScalarType::Float, sizes.size(), sizes.data(), g_log_mel,
                        dim_order.data());
  Tensor input(&input_impl);
  auto outputs = module.forward(input);
  if (!outputs.ok() || outputs->empty() || !outputs->front().isTensor()) {
    printf("SPEECH inference=FAIL session=%lu utterance=%lu err=%u\n",
           static_cast<unsigned long>(session),
           static_cast<unsigned long>(utterance),
           outputs.ok() ? 0U : static_cast<unsigned>(outputs.error()));
    return false;
  }

  Tensor result = outputs->front().toTensor();
  if (result.scalar_type() != ScalarType::Float ||
      result.numel() != static_cast<int64_t>(kWordLabelCount)) {
    printf("SPEECH inference=FAIL reason=output_contract classes=%ld\n",
           static_cast<long>(result.numel()));
    return false;
  }

  const float* logits = result.const_data_ptr<float>();
  int64_t best = 0;
  for (int64_t i = 1; i < result.numel(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  float denominator = 0.0f;
  for (int64_t i = 0; i < result.numel(); ++i) {
    denominator += std::exp(logits[i] - logits[best]);
  }
  const float confidence = 1.0f / denominator;
  const int predicted = confidence >= kUnknownConfidenceThreshold
                            ? static_cast<int>(best)
                            : -1;
#if defined(SPEECHAPP_FVP)
  const char* expected = utterance >= 1U && utterance <= kExpectedPromptCount
                             ? kExpectedPrompts[utterance - 1U]
                             : "UNEXPECTED";
  const bool pass = utterance <= kExpectedPromptCount &&
                    predicted == kExpectedClass[utterance - 1U];
#else
  const char* expected = "LIVE";
  const bool pass = true;
#endif
  if (pass) ++correct;
#if defined(SPEECHAPP_FVP)
  const ModelOutRecord record{
      session,
      utterance,
      static_cast<uint32_t>(captured_samples),
      predicted,
      utterance <= kExpectedPromptCount ? kExpectedClass[utterance - 1U] : -1,
      pass ? 1U : 0U,
      confidence,
      logits[best],
  };
  if (!record_model_out(timeslot, record)) return false;
#endif
  printf("SPEECH prediction=%s expected=%s class=%d confidence=%.4f logit=%.4f "
         "session=%lu utterance=%lu samples=%lu %s\n",
         predicted >= 0 ? kWordLabels[predicted] : "UNKNOWN", expected, predicted,
         static_cast<double>(confidence), static_cast<double>(logits[best]),
         static_cast<unsigned long>(session),
         static_cast<unsigned long>(utterance),
         static_cast<unsigned long>(captured_samples),
         pass ? "PASS" : "FAIL");
  return true;
}

constexpr uint32_t kAudioBlockBytes = 320;  // 10 ms, mono PCM16 at 16 kHz.
#if defined(SPEECHAPP_FVP)
constexpr uint32_t kSdsStreamBufferBytes = 8192;
alignas(4) uint8_t g_sds_stream_buffer[kSdsStreamBufferBytes];
alignas(4) uint8_t g_model_out_stream_buffer[2048];
#endif

#if defined(SPEECHAPP_FVP) || defined(SPEECHAPP_BOARD)
void sds_event_callback(sdsId_t, uint32_t event) {
  if ((event & SDS_EVENT_ERROR_IO) != 0U) {
    printf("SPEECH SDS event=io_error\n");
  }
  if ((event & SDS_EVENT_NO_SPACE) != 0U) {
    printf("SPEECH SDS event=no_space\n");
  }
}
#endif

class EnergySegmenter {
 public:
  void Reset() {
    active_ = false;
    samples_ = 0;
    silent_blocks_ = 0;
    utterances_ = 0;
  }

  bool Push(const int16_t* pcm, size_t samples) {
    uint64_t energy = 0;
    for (size_t i = 0; i < samples; ++i) {
      const int32_t value = pcm[i];
      energy += static_cast<uint64_t>(value * value);
    }
    const bool voice = samples != 0U && energy / samples >= kEnergyThresholdSquared;
    if (!active_) {
      if (!voice) return false;
      active_ = true;
      samples_ = 0;
      silent_blocks_ = 0;
      for (size_t i = 0; i < spelling::kClipNumSamples; ++i) {
        g_speech_clip[i] = 0;
      }
    }

    const size_t remaining = spelling::kClipNumSamples - samples_;
    const size_t copy = samples < remaining ? samples : remaining;
    for (size_t i = 0; i < copy; ++i) g_speech_clip[samples_ + i] = pcm[i];
    samples_ += copy;
    silent_blocks_ = voice ? 0U : silent_blocks_ + 1U;
    return silent_blocks_ >= kEndSilenceBlocks;
  }

  bool active() const { return active_; }
  size_t samples() const { return samples_; }
  uint32_t NextUtterance() { return ++utterances_; }
  uint32_t utterances() const { return utterances_; }
  void FinishSegment() {
    active_ = false;
    samples_ = 0;
    silent_blocks_ = 0;
  }

 private:
  // Synthetic and microphone silence has ample margin around this RMS=96
  // threshold. A neural VAD can replace this policy without changing the
  // PCM-to-log-mel front end.
  // Hardware measurements on AppKit-E8 PDM4 show an RMS noise floor below 8
  // and normal near-field speech around 40. Keep margin above noise without
  // discarding most of a spoken word.
#if defined(SPEECHAPP_BOARD)
  // PDM gain 0x80 raises the measured raw noise floor; retain margin while
  // still accepting normal near-field speech.
  static constexpr uint64_t kEnergyThresholdSquared = 64U * 64U;
#else
  static constexpr uint64_t kEnergyThresholdSquared = 96U * 96U;
#endif
  static constexpr uint32_t kEndSilenceBlocks = 20U;  // 200 ms at 10 ms/block.
  bool active_ = false;
  size_t samples_ = 0;
  uint32_t silent_blocks_ = 0;
  uint32_t utterances_ = 0;
};

#if defined(SPEECHAPP_FVP)
alignas(4) uint8_t g_audio_block[kAudioBlockBytes];

int run_sds_playback_tests(EmbeddedModule& module,
                           spelling::LogMelSpectrogram& frontend) {
  int32_t status = sdsInit(sds_event_callback);
  if (status != SDS_OK) {
    printf("SPEECH SDS init=FAIL status=%ld\n", static_cast<long>(status));
    return 1;
  }

  sdsId_t audio = nullptr;
  uint32_t sessions = 0;
  uint32_t blocks = 0;
  uint64_t bytes = 0;
  uint32_t inferences = 0;
  uint32_t correct = 0;
  uint32_t last_timeslot = 0;
  EnergySegmenter segmenter;
  sdsState = SDS_STATE_INACTIVE;
  printf("SPEECH SDS init=PASS stream=AudioIn\n");

  for (;;) {
    (void)sdsExchange();

    if ((sdsFlags & SDS_FLAG_TERMINATE) != 0U) {
      sdsState = SDS_STATE_TERMINATE;
    } else if ((sdsFlags & SDS_FLAG_RESET) != 0U) {
      sdsState = SDS_STATE_RESET;
    } else if ((sdsFlags & SDS_FLAG_ALIVE) == 0U) {
      sdsState = SDS_STATE_INACTIVE;
    }

    switch (sdsState) {
      case SDS_STATE_INACTIVE:
        if ((sdsFlags & SDS_FLAG_ALIVE) != 0U) {
          sdsState = SDS_STATE_CONNECTED;
        }
        break;

      case SDS_STATE_CONNECTED:
        if ((sdsFlags & SDS_FLAG_START) != 0U) {
          sdsState = SDS_STATE_START;
        }
        break;

      case SDS_STATE_START:
        if ((sdsFlags & SDS_FLAG_PLAYBACK) == 0U) {
          printf("SPEECH SDS mode=recording result=UNSUPPORTED\n");
          sdsState = SDS_STATE_STOP_DONE;
          break;
        }
        audio = sdsOpen("AudioIn", sdsModeRead, g_sds_stream_buffer,
                        sizeof(g_sds_stream_buffer));
        if (audio == nullptr) {
          printf("SPEECH SDS open=FAIL stream=AudioIn\n");
          sdsState = SDS_STATE_STOP_DONE;
        } else {
          g_model_out = sdsOpen("ModelOut", sdsModeWrite,
                                g_model_out_stream_buffer,
                                sizeof(g_model_out_stream_buffer));
          if (g_model_out == nullptr) {
            printf("SPEECH SDS open=FAIL stream=ModelOut\n");
            (void)sdsClose(audio);
            audio = nullptr;
            sdsState = SDS_STATE_STOP_DONE;
          } else {
            ++sessions;
            segmenter.Reset();
            printf("SPEECH SDS playback=START session=%lu\n",
                   static_cast<unsigned long>(sessions));
            sdsState = SDS_STATE_ACTIVE;
          }
        }
        break;

      case SDS_STATE_ACTIVE: {
        uint32_t timeslot = 0;
        status = sdsRead(audio, &timeslot, g_audio_block, sizeof(g_audio_block));
        if (status > 0) {
          last_timeslot = timeslot;
          ++blocks;
          bytes += static_cast<uint32_t>(status);
          const size_t samples = static_cast<size_t>(status) / sizeof(int16_t);
          if (segmenter.Push(reinterpret_cast<const int16_t*>(g_audio_block), samples)) {
            const uint32_t utterance = segmenter.NextUtterance();
            if (!infer_clip(module, frontend, g_speech_clip,
                            spelling::kClipNumSamples, segmenter.samples(),
                            sessions, utterance, timeslot, correct)) {
              return 1;
            }
            ++inferences;
            segmenter.FinishSegment();
          }
        } else if (status == SDS_NO_DATA) {
          osDelay(1U);
        } else if (status == SDS_EOS) {
          if (segmenter.active()) {
            const uint32_t utterance = segmenter.NextUtterance();
            if (!infer_clip(module, frontend, g_speech_clip,
                            spelling::kClipNumSamples, segmenter.samples(),
                            sessions, utterance, last_timeslot, correct)) {
              return 1;
            }
            ++inferences;
            segmenter.FinishSegment();
          }
          printf("SPEECH SDS playback=EOS session=%lu blocks=%lu bytes=%llu\n",
                 static_cast<unsigned long>(sessions),
                 static_cast<unsigned long>(blocks),
                 static_cast<unsigned long long>(bytes));
          sdsState = SDS_STATE_STOP_REQ;
        } else {
          printf("SPEECH SDS read=FAIL status=%ld\n", static_cast<long>(status));
          sdsState = SDS_STATE_STOP_REQ;
        }
        break;
      }

      case SDS_STATE_STOP_REQ:
        status = sdsClose(g_model_out);
        g_model_out = nullptr;
        if (status != SDS_OK) {
          printf("SPEECH SDS close=FAIL stream=ModelOut status=%ld\n",
                 static_cast<long>(status));
          (void)sdsUninit();
          return 1;
        }
        status = sdsClose(audio);
        audio = nullptr;
        if (status != SDS_OK) {
          printf("SPEECH SDS close=FAIL status=%ld\n", static_cast<long>(status));
          (void)sdsUninit();
          return 1;
        }
        printf("SPEECH SDS playback=STOP session=%lu\n",
               static_cast<unsigned long>(sessions));
        printf("SPEECH session=%lu utterances=%lu expected=21\n",
               static_cast<unsigned long>(sessions),
               static_cast<unsigned long>(segmenter.utterances()));
        if (segmenter.utterances() != kExpectedPromptCount) return 1;
        sdsState = SDS_STATE_STOP_DONE;
        break;

      case SDS_STATE_STOP_DONE:
        // Acknowledge completion to VSI3. Auto-playback will only request the
        // next configured step after the target has cleared START.
        sdsFlagsModify(0U, SDS_FLAG_START);
        sdsState = SDS_STATE_INACTIVE;
        break;

      case SDS_STATE_TERMINATE:
        (void)sdsUninit();
        printf("SPEECH SDS result=PASS sessions=%lu blocks=%lu bytes=%llu "
               "inferences=%lu correct=%lu total=%lu\n",
               static_cast<unsigned long>(sessions),
               static_cast<unsigned long>(blocks),
               static_cast<unsigned long long>(bytes),
               static_cast<unsigned long>(inferences),
               static_cast<unsigned long>(correct),
               static_cast<unsigned long>(3U * kExpectedPromptCount));
        return sessions == 3U && inferences == 3U * kExpectedPromptCount ? 0 : 1;

      case SDS_STATE_RESET:
      default:
        if (audio != nullptr) {
          (void)sdsClose(audio);
        }
        if (g_model_out != nullptr) {
          (void)sdsClose(g_model_out);
          g_model_out = nullptr;
        }
        (void)sdsUninit();
        printf("SPEECH SDS result=FAIL state=%lu\n",
               static_cast<unsigned long>(sdsState));
        return 1;
    }

    osDelay(1U);
  }
}
#endif

#if defined(SPEECHAPP_BOARD)
// SDS stream-open and host USB scheduling can stall the application thread for
// several hundred milliseconds. Keep 1.28 s of capture queued so PDM IRQ
// capture remains lossless across that control-plane latency.
constexpr uint32_t kMicBlockCount = 128U;
constexpr uint32_t kMicInputRate = 48000U;
constexpr uint32_t kMicOutputRate = 16000U;
constexpr uint32_t kMicDecimation = kMicInputRate / kMicOutputRate;
constexpr uint32_t kMicInputBlockBytes = kAudioBlockBytes * kMicDecimation;
constexpr uint32_t kMicWarmupBlocks = 100U;
constexpr uint32_t kBoardSdsBufferBytes = 8192U;
alignas(4) uint8_t g_mic_stream_buffer[kMicBlockCount * kMicInputBlockBytes];
alignas(4) int16_t g_mic_decimated_block[kAudioBlockBytes / sizeof(int16_t)];
alignas(4) uint8_t g_board_sds_buffer[kBoardSdsBufferBytes];
volatile uint32_t g_mic_events = 0U;
sdsId_t g_board_audio_stream = nullptr;
uint32_t g_board_sds_sessions = 0U;
uint32_t g_board_sds_blocks = 0U;
uint32_t g_board_sds_next_exchange = 0U;

// 63-tap Hamming-window low-pass, 7 kHz cutoff at 48 kHz, Q15 unity gain.
// It rejects content that would alias into the 16 kHz model input.
constexpr int16_t kDecimatorFir[] = {
    -4,   20,   32,    19,    -17,  -53,   -54,  0,     80,
    118,  56,   -88,   -207,  -174, 38,    291,  361,   117,
    -320, -607, -428,  211,   878,  966,   173,  -1126, -1946,
    -1253, 1301, 4983, 8244,  9546, 8244,  4983, 1301,  -1253,
    -1946, -1126, 173, 966,   878,  211,   -428, -607,  -320,
    117,  361,  291,   38,    -174, -207,  -88,  56,    118,
    80,   0,    -54,   -53,   -17,  19,    32,   20,    -4,
};
constexpr size_t kDecimatorTaps = sizeof(kDecimatorFir) / sizeof(kDecimatorFir[0]);
int16_t g_decimator_history[kDecimatorTaps - 1U] = {};
int16_t g_decimator_work[(kDecimatorTaps - 1U) +
                         (kMicInputBlockBytes / sizeof(int16_t))];

void decimate_48k_to_16k(const int16_t* input, int16_t* output) {
  constexpr size_t kHistory = kDecimatorTaps - 1U;
  constexpr size_t kInputSamples = kMicInputBlockBytes / sizeof(int16_t);
  constexpr size_t kOutputSamples = kAudioBlockBytes / sizeof(int16_t);
  std::memcpy(g_decimator_work, g_decimator_history,
              sizeof(g_decimator_history));
  std::memcpy(g_decimator_work + kHistory, input,
              kMicInputBlockBytes);

  for (size_t i = 0; i < kOutputSamples; ++i) {
    const size_t newest = kHistory + (i * kMicDecimation);
    int64_t accumulator = 0;
    for (size_t tap = 0; tap < kDecimatorTaps; ++tap) {
      accumulator += static_cast<int32_t>(kDecimatorFir[tap]) *
                     static_cast<int32_t>(g_decimator_work[newest - tap]);
    }
    accumulator = (accumulator + (1LL << 14)) >> 15;
    if (accumulator > 32767) accumulator = 32767;
    if (accumulator < -32768) accumulator = -32768;
    output[i] = static_cast<int16_t>(accumulator);
  }
  std::memcpy(g_decimator_history, input + kInputSamples - kHistory,
              sizeof(g_decimator_history));
}

void microphone_event_callback(uint32_t event) {
  g_mic_events |= event;
}

void normalize_live_utterance(int16_t* pcm, size_t samples) {
  if ((pcm == nullptr) || (samples == 0U)) return;

  uint64_t energy = 0U;
  uint32_t peak = 0U;
  for (size_t i = 0; i < samples; ++i) {
    const int32_t sample = pcm[i];
    const uint32_t magnitude = sample < 0
                                   ? static_cast<uint32_t>(-sample)
                                   : static_cast<uint32_t>(sample);
    if (magnitude > peak) peak = magnitude;
    energy += static_cast<uint64_t>(sample * sample);
  }
  const float input_rms = std::sqrt(static_cast<float>(energy) /
                                    static_cast<float>(samples));
  if ((input_rms < 1.0f) || (peak == 0U)) return;

  constexpr float kTargetRms = 4000.0f;
  constexpr float kPeakCeiling = 28000.0f;
  constexpr float kMaximumGain = 128.0f;
  float gain = kTargetRms / input_rms;
  const float peak_limited_gain = kPeakCeiling / static_cast<float>(peak);
  if (gain > peak_limited_gain) gain = peak_limited_gain;
  if (gain > kMaximumGain) gain = kMaximumGain;

  for (size_t i = 0; i < samples; ++i) {
    float scaled = static_cast<float>(pcm[i]) * gain;
    if (scaled > 32767.0f) scaled = 32767.0f;
    if (scaled < -32768.0f) scaled = -32768.0f;
    pcm[i] = static_cast<int16_t>(scaled);
  }
  printf("SPEECH frontend=AGC input_rms=%.1f input_peak=%lu gain=%.2f "
         "target_rms=4000\n",
         static_cast<double>(input_rms), static_cast<unsigned long>(peak),
         static_cast<double>(gain));
}

bool board_sds_init() {
  const int32_t status = sdsInit(sds_event_callback);
  if (status != SDS_OK) {
    printf("SPEECH SDS USB init=FAIL status=%ld\n", static_cast<long>(status));
    return false;
  }
  NVIC_SetPriority(USB_IRQ_IRQn, SDS_USB_IRQ_PRIORITY);
  sdsState = SDS_STATE_INACTIVE;
  g_board_sds_next_exchange = osKernelGetTickCount();
  printf("SPEECH SDS USB init=PASS stream=AudioIn mode=recording "
         "irq_priority=%u\n",
         static_cast<unsigned>(NVIC_GetPriority(USB_IRQ_IRQn)));
  return true;
}

bool board_sds_close_stream() {
  if (g_board_audio_stream == nullptr) return true;
  const int32_t status = sdsClose(g_board_audio_stream);
  if (status != SDS_OK) {
    printf("SPEECH SDS USB close=FAIL status=%ld\n", static_cast<long>(status));
    return false;
  }
  g_board_audio_stream = nullptr;
  printf("SPEECH SDS USB recording=STOP session=%lu blocks=%lu\n",
         static_cast<unsigned long>(g_board_sds_sessions),
         static_cast<unsigned long>(g_board_sds_blocks));
  return true;
}

bool board_sds_exchange_control() {
  const uint32_t now = osKernelGetTickCount();
  if (static_cast<int32_t>(now - g_board_sds_next_exchange) < 0) {
    return true;
  }
  g_board_sds_next_exchange = now + 100U;
  (void)sdsExchange();

  if ((sdsFlags & SDS_FLAG_RESET) != 0U) {
    (void)board_sds_close_stream();
    __NVIC_SystemReset();
  }
  if ((sdsFlags & SDS_FLAG_TERMINATE) != 0U) {
    (void)board_sds_close_stream();
    sdsFlagsModify(0U, SDS_FLAG_TERMINATE | SDS_FLAG_START);
    sdsState = SDS_STATE_INACTIVE;
  } else if ((sdsFlags & SDS_FLAG_ALIVE) == 0U) {
    (void)board_sds_close_stream();
    sdsState = SDS_STATE_INACTIVE;
  }

  switch (sdsState) {
    case SDS_STATE_INACTIVE:
      if ((sdsFlags & SDS_FLAG_ALIVE) != 0U) {
        sdsState = SDS_STATE_CONNECTED;
        printf("SPEECH SDS USB host=CONNECTED\n");
      }
      break;
    case SDS_STATE_CONNECTED:
      if ((sdsFlags & SDS_FLAG_START) != 0U) sdsState = SDS_STATE_START;
      break;
    case SDS_STATE_START:
      if ((sdsFlags & SDS_FLAG_PLAYBACK) != 0U) {
        printf("SPEECH SDS USB playback=UNSUPPORTED target=Board\n");
        sdsFlagsModify(0U, SDS_FLAG_START);
        sdsState = SDS_STATE_CONNECTED;
        break;
      }
      g_board_audio_stream = sdsOpen("AudioIn", sdsModeWrite,
                                     g_board_sds_buffer,
                                     sizeof(g_board_sds_buffer));
      if (g_board_audio_stream == nullptr) {
        printf("SPEECH SDS USB open=FAIL stream=AudioIn\n");
        sdsFlagsModify(0U, SDS_FLAG_START);
        sdsState = SDS_STATE_CONNECTED;
        return false;
      }
      ++g_board_sds_sessions;
      g_board_sds_blocks = 0U;
      sdsState = SDS_STATE_ACTIVE;
      printf("SPEECH SDS USB recording=START session=%lu stream=AudioIn\n",
             static_cast<unsigned long>(g_board_sds_sessions));
      break;
    case SDS_STATE_ACTIVE:
      if ((sdsFlags & SDS_FLAG_START) == 0U) {
        sdsState = SDS_STATE_STOP_REQ;
      }
      break;
    case SDS_STATE_STOP_REQ:
      if (!board_sds_close_stream()) return false;
      sdsFlagsModify(0U, SDS_FLAG_START);
      sdsState = SDS_STATE_CONNECTED;
      break;
    default:
      break;
  }
  return true;
}

bool board_sds_record_block(const void* block, uint32_t timeslot) {
  if ((sdsState != SDS_STATE_ACTIVE) || (g_board_audio_stream == nullptr)) {
    return true;
  }
  int32_t status;
  do {
    status = sdsWrite(g_board_audio_stream, timeslot, block, kAudioBlockBytes);
    if (status == SDS_NO_SPACE) {
      if (!board_sds_exchange_control()) return false;
      osDelay(1U);
    }
  } while ((status == SDS_NO_SPACE) &&
           ((sdsFlags & SDS_FLAG_START) != 0U));
  if (status != static_cast<int32_t>(kAudioBlockBytes)) {
    printf("SPEECH SDS USB write=FAIL status=%ld blocks=%lu\n",
           static_cast<long>(status),
           static_cast<unsigned long>(g_board_sds_blocks));
    return false;
  }
  ++g_board_sds_blocks;
  return true;
}

int run_continuous_microphone(EmbeddedModule& module,
                              spelling::LogMelSpectrogram& frontend) {
  vStreamDriver_t& microphone = Driver_vStreamAudioIn;
  if (!board_sds_init()) return 1;
  int32_t status = microphone.Initialize(microphone_event_callback);
  if (status != VSTREAM_OK) {
    printf("SPEECH microphone=FAIL stage=initialize status=%ld\n",
           static_cast<long>(status));
    return 1;
  }

  status = speechapp_pdm_configure(AUDIO_CFG_PDM_MODE,
                                   AUDIO_CFG_PDM_CHANNEL,
                                   AUDIO_CFG_PDM_IIR_COEF);
  if (status != 0) {
    printf("SPEECH microphone=FAIL stage=set_pdm_mode status=%ld\n",
           static_cast<long>(status));
    (void)microphone.Uninitialize();
    return 1;
  }

  status = microphone.SetBuf(g_mic_stream_buffer, sizeof(g_mic_stream_buffer),
                             kMicInputBlockBytes);
  if (status != VSTREAM_OK) {
    printf("SPEECH microphone=FAIL stage=set_buffer status=%ld\n",
           static_cast<long>(status));
    (void)microphone.Uninitialize();
    return 1;
  }

  status = microphone.Start(VSTREAM_MODE_CONTINUOUS);
  if (status != VSTREAM_OK) {
    printf("SPEECH microphone=FAIL stage=start status=%ld\n",
           static_cast<long>(status));
    (void)microphone.Uninitialize();
    return 1;
  }

  EnergySegmenter segmenter;
  segmenter.Reset();
  const uint32_t microphone_start_timeslot = osKernelGetTickCount();
  uint32_t correct = 0U;
  uint32_t blocks = 0U;
  uint32_t level_blocks = 0U;
  uint64_t level_energy = 0U;
  uint32_t level_peak = 0U;
  printf("SPEECH microphone=READY mode=continuous channel=PDM%s format=PCM16 "
         "source_rate=48000 rate=16000 decimation=3 block_ms=10 "
         "pdm_mode=%u pdm_clock_hz=%lu "
         "dc_block=on iir_coefficient=%u\n",
         APP_STRINGIFY(AUDIO_CFG_PDM_CHANNEL),
         static_cast<unsigned>(AUDIO_CFG_PDM_MODE),
         static_cast<unsigned long>(AUDIO_CFG_PDM_CLOCK_HZ),
         static_cast<unsigned>(AUDIO_CFG_PDM_IIR_COEF));
  fflush(stdout);

  for (;;) {
    if (!board_sds_exchange_control()) return 1;
    const uint32_t events = g_mic_events;
    g_mic_events &= ~events;
    if ((events & VSTREAM_EVENT_OVERFLOW) != 0U) {
      printf("SPEECH microphone=WARNING event=overflow blocks=%lu\n",
             static_cast<unsigned long>(blocks));
    }
    if ((events & VSTREAM_EVENT_EOS) != 0U) {
      const vStreamStatus_t stream_status = microphone.GetStatus();
      printf("SPEECH microphone=FAIL event=eos active=%u overflow=%u blocks=%lu\n",
             stream_status.active, stream_status.overflow,
             static_cast<unsigned long>(blocks));
      (void)microphone.Stop();
      (void)microphone.Uninitialize();
      return 1;
    }

    bool consumed = false;
    for (;;) {
      void* block = microphone.GetBlock();
      if (block == nullptr) break;
      consumed = true;
      ++blocks;
      const uint32_t capture_timeslot =
          microphone_start_timeslot + ((blocks - 1U) * 10U);
      decimate_48k_to_16k(static_cast<const int16_t*>(block),
                          g_mic_decimated_block);
      const int16_t* pcm = g_mic_decimated_block;
      for (size_t i = 0; i < kAudioBlockBytes / sizeof(int16_t); ++i) {
        const int32_t sample = pcm[i];
        const uint32_t magnitude = sample < 0
                                       ? static_cast<uint32_t>(-sample)
                                       : static_cast<uint32_t>(sample);
        if (magnitude > level_peak) level_peak = magnitude;
        level_energy += static_cast<uint64_t>(sample * sample);
      }
      ++level_blocks;
      if (level_blocks == 100U) {
        const double mean_square = static_cast<double>(level_energy) /
                                   (100.0 * kAudioBlockBytes / sizeof(int16_t));
        printf("SPEECH microphone=LEVEL blocks=%lu rms=%.1f peak=%lu\n",
               static_cast<unsigned long>(blocks), std::sqrt(mean_square),
               static_cast<unsigned long>(level_peak));
        level_blocks = 0U;
        level_energy = 0U;
        level_peak = 0U;
        fflush(stdout);
      }
      const bool utterance_ready =
          (blocks > kMicWarmupBlocks) &&
          segmenter.Push(pcm, kAudioBlockBytes / sizeof(int16_t));
      if (!board_sds_record_block(pcm, capture_timeslot)) return 1;
      status = microphone.ReleaseBlock();
      if (status != VSTREAM_OK) {
        printf("SPEECH microphone=FAIL stage=release status=%ld\n",
               static_cast<long>(status));
        return 1;
      }
      if (utterance_ready) {
        const uint32_t utterance = segmenter.NextUtterance();
        normalize_live_utterance(g_speech_clip, segmenter.samples());
        if (!infer_clip(module, frontend, g_speech_clip,
                        spelling::kClipNumSamples, segmenter.samples(), 1U,
                        utterance, blocks, correct)) {
          return 1;
        }
        segmenter.FinishSegment();
        fflush(stdout);
      }
    }
    if (!consumed) osDelay(1U);
  }
}
#endif

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();

  printf("SPEECH status=boot runtime=ExecuTorch target=%s model_bytes=%lu\n",
         APP_TARGET_CONFIG, MoonshineWordCNN_pte_size);

  EmbeddedModule module(
      MoonshineWordCNN_pte, MoonshineWordCNN_pte_size,
      std::make_unique<BufferDataLoader>(MoonshineWordCNN_pte, MoonshineWordCNN_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));
  spelling::LogMelSpectrogram frontend(make_frontend_params());

#if defined(SPEECHAPP_FVP)
  if (run_sds_playback_tests(module, frontend) != 0) {
    printf("SPEECH status=sds_playback result=FAIL\n");
    putchar('\x04');
    fflush(stdout);
    return 1;
  }
#elif defined(SPEECHAPP_BOARD)
  printf("SPEECH frontend=READY input=PCM16 rate=16000 mels=64 frames=128\n");
  if (run_continuous_microphone(module, frontend) != 0) {
    printf("SPEECH status=microphone result=FAIL\n");
    fflush(stdout);
    return 1;
  }
#else
  printf("SPEECH frontend=READY input=PCM16 rate=16000 mels=64 frames=128\n");
#endif

  printf("SPEECH status=inference_complete result=PASS\n");
  fflush(stdout);
#if defined(SPEECHAPP_FVP)
  putchar('\x04');  // EOT: ends the FVP run
  fflush(stdout);
#endif
  return 0;
}
