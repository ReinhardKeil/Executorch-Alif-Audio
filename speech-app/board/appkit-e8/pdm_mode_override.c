// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "Driver_PDM.h"

extern ARM_DRIVER_PDM Driver_PDM;

int32_t speechapp_pdm_configure(uint32_t mode, uint32_t channel,
                                uint32_t iir_coefficient) {
  /* 35-tap symmetric 6.5 kHz Hamming low-pass, unity gain, quantized to
   * the PDM peripheral's signed 11.10 coefficient format. */
  PDM_CH_CONFIG filter = {
      .ch_num = (uint8_t)channel,
      .ch_fir_coef = {0x7FF, 0x000, 0x001, 0x7FD, 0x005, 0x7FB,
                      0x002, 0x005, 0x7F1, 0x018, 0x7E6, 0x010,
                      0x00A, 0x7CD, 0x063, 0x76E, 0x0B4, 0x342},
      .ch_iir_coef = iir_coefficient,
  };
  int32_t status = Driver_PDM.Control(ARM_PDM_MODE, mode, 0U);
  if (status != ARM_DRIVER_OK) {
    return status;
  }
  status = Driver_PDM.Config(&filter);
  if (status != ARM_DRIVER_OK) {
    return status;
  }
  return Driver_PDM.Control(ARM_PDM_BYPASS_IIR_FILTER, 0U, 0U);
}
