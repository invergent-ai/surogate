// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "sol.h"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <chrono>

#include "utilities/dtype.h"
#include "kernels/kernels.h"        // for benchmarking matmul
#include "utils.h"

struct sPerfSpecs {
    const char* Chip = nullptr;
    int SMs = -1;    // Number of SMs
    int CoresPerSM;  // Number of cores per SM
    int TensorPerSM; // Number of tensor cores per SM
    int BoostClock;  // in MhZ

    float TF32_TFlops;        // TFlops in TF32
    float BF16_TFlops;        // in BF16
    float FP16_32_TFlops;     // in FP16 with FP32 accumulate
    float FP16_16_TFlops;     // in FP16 with FP16 accumulate
    float INT8_TFlops;        // in INT8
    float INT4_TFlops;        // in INT4
    float FP8_32_TFlops = -1; // in FP8 with FP32 accumulate
    float FP8_16_TFlops = -1; // in FP8 with FP16 accumulate
    float FP4_32_TFlops = -1; // in FP4 with FP32 accumulate
};

// source: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
sPerfSpecs RTX_3090_FE = {
    .Chip = "GA102", .SMs = 82, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1695, .TF32_TFlops = 35.6, .BF16_TFlops = 71, .FP16_32_TFlops = 71, .FP16_16_TFlops = 142, .INT8_TFlops = 284, .INT4_TFlops = 568};

sPerfSpecs RTX_3070_FE = {
    .Chip = "GA104", .SMs = 46, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1725, .TF32_TFlops = 20.3, .BF16_TFlops = 40.6, .FP16_32_TFlops = 40.6, .FP16_16_TFlops = 81.3, .INT8_TFlops = 162.6, .INT4_TFlops = 253.2};

// source: https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
// source: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf
sPerfSpecs B6000_MaxQ = {
    .Chip = "GB202", .SMs = 188, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2280, .TF32_TFlops = 219.5, .BF16_TFlops = 438.9, .FP16_32_TFlops = 438.9, .FP16_16_TFlops = 438.9, .INT8_TFlops = 877.9, .INT4_TFlops = -1, .FP8_32_TFlops = 877.9, .FP8_16_TFlops = 877.9, .FP4_32_TFlops = 1755.7};

sPerfSpecs B6000_WS = {
    .Chip = "GB202", .SMs = 188, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2617, .TF32_TFlops = 251.9, .BF16_TFlops = 503.8, .FP16_32_TFlops = 503.8, .FP16_16_TFlops = 503.8, .INT8_TFlops = 1007.6, .INT4_TFlops = -1, .FP8_32_TFlops = 1007.6, .FP8_16_TFlops = 1007.6, .FP4_32_TFlops = 2015.2};

sPerfSpecs A6000 = {
    .Chip = "GA102", .SMs = 84, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1800, .TF32_TFlops = 77.4, .BF16_TFlops = 154.8, .FP16_32_TFlops = 154.8, .FP16_16_TFlops = 154.8, .INT8_TFlops = 309.7, .INT4_TFlops = 619.4};

sPerfSpecs A6000_ADA = {
    .Chip = "AD102", .SMs = 142, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2505, .TF32_TFlops = 182.1, .BF16_TFlops = 364.2, .FP16_32_TFlops = 364.2, .FP16_16_TFlops = 364.2, .INT8_TFlops = 728.5, .INT4_TFlops = 1457.0, .FP8_32_TFlops = 728.5, .FP8_16_TFlops = 728.5};

// source: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
sPerfSpecs A100_SXM = {
    .Chip = "GA100", .SMs = 108, .CoresPerSM = 64, .TensorPerSM = 4, .BoostClock = 1410, .TF32_TFlops = 156, .BF16_TFlops = 312, .FP16_32_TFlops = 312, .FP16_16_TFlops = 312, .INT8_TFlops = 624, .INT4_TFlops = 1248};

// source: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
sPerfSpecs RTX_4090 = {
    .Chip = "AD102", .SMs = 128, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2520, .TF32_TFlops = 82.6, .BF16_TFlops = 165.2, .FP16_32_TFlops = 165.2, .FP16_16_TFlops = 330.3, .INT8_TFlops = 660.6, .INT4_TFlops = 1321.2, .FP8_32_TFlops = 330.3, .FP8_16_TFlops = 660.6};

sPerfSpecs RTX_4080 = {
    .Chip = "AD103", .SMs = 76, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2505, .TF32_TFlops = 48.7, .BF16_TFlops = 97.5, .FP16_32_TFlops = 97.5, .FP16_16_TFlops = 194.9, .INT8_TFlops = 389.9, .INT4_TFlops = 779.8, .FP8_32_TFlops = 194.9, .FP8_16_TFlops = 389.9};

sPerfSpecs RTX_3090_Ti = {
    .Chip = "GA102", .SMs = 84, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1860, .TF32_TFlops = 40, .BF16_TFlops = 80, .FP16_32_TFlops = 80, .FP16_16_TFlops = 160, .INT8_TFlops = 320, .INT4_TFlops = 640};

sPerfSpecs RTX_3080_Ti = {
    .Chip = "GA102", .SMs = 80, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1665, .TF32_TFlops = 34.1, .BF16_TFlops = 68.2, .FP16_32_TFlops = 68.2, .FP16_16_TFlops = 136.8, .INT8_TFlops = 272.8, .INT4_TFlops = 545.6};

sPerfSpecs L40 = {
    .Chip = "AD102",
    .SMs = 142,
    .CoresPerSM = 128,
    .TensorPerSM = 4,
    .BoostClock = 2490,
    .TF32_TFlops = 90.5,
    .BF16_TFlops = 181,
    .FP16_32_TFlops = 181,
    .FP16_16_TFlops = 362, /* extrapolated */
    .INT8_TFlops = 362,
    .INT4_TFlops = 728,
    .FP8_32_TFlops = 362,
    .FP8_16_TFlops = 728 /* extrapolated */
};

// Note: I don't believe these numbers. Running a gigantic matmul (32k x 32k), I can get only
// about 230 TFlop/s on our machine.
sPerfSpecs L40S = {
    .Chip = "AD102",
    .SMs = 142,
    .CoresPerSM = 128,
    .TensorPerSM = 4,
    .BoostClock = 2520,
    .TF32_TFlops = 183,
    .BF16_TFlops = 362.05,
    .FP16_32_TFlops = 362.05,
    .FP16_16_TFlops = 362.05,
    .INT8_TFlops = 733,
    .INT4_TFlops = 733,
    .FP8_32_TFlops = 733,
    .FP8_16_TFlops = 728
};

sPerfSpecs A40 = {
    .Chip = "GA102",
    .SMs = 84,
    .CoresPerSM = 128,
    .TensorPerSM = 4,
    .BoostClock = 1740,
    .TF32_TFlops = 74.8,
    .BF16_TFlops = 149.7,
    .FP16_32_TFlops = 149.7,
    .FP16_16_TFlops = 299.3, /* extrapolated */
    .INT8_TFlops = 299.3,
    .INT4_TFlops = 598.7};

sPerfSpecs L4 = {
    .Chip = "AD104", .SMs = 58, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2040, .TF32_TFlops = 60, .BF16_TFlops = 121, .FP16_32_TFlops = 121, .FP16_16_TFlops = 242, /* extrapolated */
    .INT8_TFlops = 242,
    .INT4_TFlops = 484,
    .FP8_32_TFlops = 242,
    .FP8_16_TFlops = 484 /* extrapolated */
};

// source: https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
sPerfSpecs H100_SXM = {
    .Chip = "GH100", .SMs = 132, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1830, .TF32_TFlops = 494.7, .BF16_TFlops = 989.4, .FP16_32_TFlops = 989.4, .FP16_16_TFlops = 989.4, .INT8_TFlops = 1978.9, .INT4_TFlops = -1, .FP8_32_TFlops = 1978.9, .FP8_16_TFlops = 1978.9};

sPerfSpecs H100_PCI = {
    .Chip = "GH100", .SMs = 114, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1620, .TF32_TFlops = 378, .BF16_TFlops = 756, .FP16_32_TFlops = 756, .FP16_16_TFlops = 756, .INT8_TFlops = 1513, .INT4_TFlops = -1, .FP8_32_TFlops = 1513, .FP8_16_TFlops = 1513};

// source: https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200
sPerfSpecs H200_SXM = {
    .Chip = "GH100", .SMs = 132, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1980, .TF32_TFlops = 989, .BF16_TFlops = 1979, .FP16_32_TFlops = 1979, .FP16_16_TFlops = 1979, .INT8_TFlops = 3958, .INT4_TFlops = -1, .FP8_32_TFlops = 3958, .FP8_16_TFlops = 3958};

// source: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
sPerfSpecs RTX_5090 = {
    .Chip = "GB202", .SMs = 170, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2407, .TF32_TFlops = 104.8, .BF16_TFlops = 209.5, .FP16_32_TFlops = 209.5, .FP16_16_TFlops = 419, .INT8_TFlops = 838, .INT4_TFlops = -1, .FP8_32_TFlops = 419, .FP8_16_TFlops = 838, .FP4_32_TFlops = 1676};

sPerfSpecs RTX_5080 = {
    .Chip = "GB203", .SMs = 84, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2617, .TF32_TFlops = 56.3, .BF16_TFlops = 112.6, .FP16_32_TFlops = 112.6, .FP16_16_TFlops = 225.1, .INT8_TFlops = 450.2, .INT4_TFlops = -1, .FP8_32_TFlops = 225.1, .FP8_16_TFlops = 450.2, .FP4_32_TFlops = 900.4};

sPerfSpecs RTX_5070_Ti = {
    .Chip = "GB203", .SMs = 70, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2452, .TF32_TFlops = 43.9, .BF16_TFlops = 87.9, .FP16_32_TFlops = 87.9, .FP16_16_TFlops = 175.8, .INT8_TFlops = 351.5, .INT4_TFlops = -1, .FP8_32_TFlops = 175.8, .FP8_16_TFlops = 351.5, .FP4_32_TFlops = 703};

sPerfSpecs RTX_5070 = {
    .Chip = "GB205", .SMs = 48, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2512, .TF32_TFlops = 30.9, .BF16_TFlops = 61.7, .FP16_32_TFlops = 61.7, .FP16_16_TFlops = 123.5, .INT8_TFlops = 246.9, .INT4_TFlops = -1, .FP8_32_TFlops = 123.5, .FP8_16_TFlops = 246.9, .FP4_32_TFlops = 493.9};

sPerfSpecs RTX_4070_Ti = {
    .Chip = "AD104", .SMs = 60, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2610, .TF32_TFlops = 40.1, .BF16_TFlops = 80.2, .FP16_32_TFlops = 80.2, .FP16_16_TFlops = 160.4, .INT8_TFlops = 320.7, .INT4_TFlops = 641.4 /*e*/, .FP8_32_TFlops = 160.4, .FP8_16_TFlops = 320.7};

sPerfSpecs RTX_4070 = {
    .Chip = "AD104", .SMs = 46, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2475, .TF32_TFlops = 29.1, .BF16_TFlops = 58.3, .FP16_32_TFlops = 58.3, .FP16_16_TFlops = 116.6, .INT8_TFlops = 233.2, .INT4_TFlops = 466.4 /*e*/, .FP8_32_TFlops = 116.6, .FP8_16_TFlops = 233.2};

sPerfSpecs RTX_4070_Laptop_GPU = {
    .Chip = "AD106", .SMs = 36, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1695, .TF32_TFlops = 20.0, .BF16_TFlops = 40.1, .FP16_32_TFlops = 40.1, .FP16_16_TFlops = 80.2, .INT8_TFlops = 160.4, .INT4_TFlops = 320.7, .FP8_32_TFlops = 80.2, .FP8_16_TFlops = 160.4};

sPerfSpecs RTX_3080 = {
    .Chip = "GA102", .SMs = 68, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1710, .TF32_TFlops = 29.8, .BF16_TFlops = 59.5, .FP16_32_TFlops = 59.5, .FP16_16_TFlops = 119.1, .INT8_TFlops = 238.1, .INT4_TFlops = 476.2, /* extrapolated */
};

sPerfSpecs RTX_3070_Ti = {
    .Chip = "GA104", .SMs = 48, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1770, .TF32_TFlops = 21.7, .BF16_TFlops = 43.5, .FP16_32_TFlops = 43.5, .FP16_16_TFlops = 87, .INT8_TFlops = 87, .INT4_TFlops = 174, /* extrapolated */
};

sPerfSpecs RTX_3070 = {
    .Chip = "GA104", .SMs = 46, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1725, .TF32_TFlops = 20.3, .BF16_TFlops = 40.6, .FP16_32_TFlops = 40.6, .FP16_16_TFlops = 81.3, .INT8_TFlops = 81.3, .INT4_TFlops = 162.6 /* extrapolated */
};

// source: https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief
// + https://www.techpowerup.com/gpu-specs/b200-sxm-192-gb.c4210
sPerfSpecs B200_HGX = {
    .Chip = "GB100", .SMs = 264, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 1837, .TF32_TFlops = 1100, .BF16_TFlops = 2200, .FP16_32_TFlops = 2200, .FP16_16_TFlops = 2200, .INT8_TFlops = 4500, .INT4_TFlops = -1, .FP8_32_TFlops = 4500, .FP8_16_TFlops = 4500, .FP4_32_TFlops = 9000};

// source: https://www.nvidia.com/en-us/data-center/b300/
sPerfSpecs B300_SXM = {
    .Chip = "GB100", .SMs = 264, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2100, .TF32_TFlops = 1250, .BF16_TFlops = 2500, .FP16_32_TFlops = 2500, .FP16_16_TFlops = 2500, .INT8_TFlops = 5000, .INT4_TFlops = -1, .FP8_32_TFlops = 5000, .FP8_16_TFlops = 5000, .FP4_32_TFlops = 10000};

// These are mostly guesswork at this point!
// the spec sheet claims 1 pFLOP fp4+sparsity; assume this is mostly like a 5090;
// TODO get better estimates for these values
sPerfSpecs GB10 = {
    .Chip = "GB10", .SMs = 48, .CoresPerSM = 128, .TensorPerSM = 4, .BoostClock = 2418, .TF32_TFlops = 29.71, .BF16_TFlops = 59.42, .FP16_32_TFlops = 59.42, .FP16_16_TFlops = 118.84, .INT8_TFlops = 118.84, .INT4_TFlops = -1, .FP8_32_TFlops = 118.84, .FP8_16_TFlops = 237.68, .FP4_32_TFlops = 475.36
};


sPerfSpecs interpolate(const sPerfSpecs& src, int sms, int clock) {
    float scale_factor = static_cast<float>(sms) / src.SMs * static_cast<float>(clock) / src.BoostClock;
    return sPerfSpecs{
        .Chip = src.Chip,
        .SMs = sms,
        .CoresPerSM = src.CoresPerSM,
        .TensorPerSM = src.TensorPerSM,
        .BoostClock = clock,
        .TF32_TFlops = src.TF32_TFlops * scale_factor,
        .BF16_TFlops = src.BF16_TFlops * scale_factor,
        .FP16_32_TFlops = src.FP16_32_TFlops * scale_factor,
        .FP16_16_TFlops = src.FP16_16_TFlops * scale_factor,
        .INT8_TFlops = src.INT8_TFlops * scale_factor,
        .INT4_TFlops = src.INT4_TFlops * scale_factor,
        .FP8_32_TFlops = src.FP8_32_TFlops * scale_factor,
        .FP8_16_TFlops = src.FP8_16_TFlops * scale_factor,
        .FP4_32_TFlops = src.FP4_32_TFlops * scale_factor};
}

std::unordered_map<std::string_view, sPerfSpecs> create_device_map() {
    std::unordered_map<std::string_view, sPerfSpecs> device_map;
    // list of know GPU names
    device_map["NVIDIA A100-PCIE-40GB"] = interpolate(A100_SXM, 108, 1410);
    device_map["NVIDIA A100-PCIE-80GB"] = interpolate(A100_SXM, 108, 1410);
    device_map["NVIDIA A100-SXM4-40GB"] = interpolate(A100_SXM, 108, 1410);
    device_map["NVIDIA A100-SXM4-80GB"] = interpolate(A100_SXM, 108, 1410);
    device_map["NVIDIA RTX A2000"] = interpolate(A6000, 26, 1200);
    device_map["NVIDIA RTX A4000"] = interpolate(A6000, 48, 1560);
    device_map["NVIDIA RTX A4500"] = interpolate(A6000, 56, 1650);
    device_map["NVIDIA RTX A5000"] = interpolate(A6000, 64, 1695);
    device_map["NVIDIA RTX A5500"] = interpolate(A6000, 80, 1770);
    device_map["NVIDIA RTX A6000"] = A6000;

    device_map["NVIDIA GeForce RTX 3090 Ti"] = RTX_3090_Ti;
    device_map["NVIDIA GeForce RTX 3090"] = RTX_3090_FE;
    device_map["NVIDIA GeForce RTX 3080 Ti"] = RTX_3080_Ti;
    device_map["NVIDIA GeForce RTX 3080"] = RTX_3080;
    device_map["NVIDIA GeForce RTX 3070 Ti"] = RTX_3070_Ti;
    device_map["NVIDIA GeForce RTX 3070"] = RTX_3070;
    device_map["NVIDIA GeForce RTX 3060 Ti"] = interpolate(RTX_3070, 38, 1665);
    device_map["NVIDIA GeForce RTX 3060"] = interpolate(RTX_3070, 28, 1777);

    device_map["NVIDIA RTX 2000 Ada Generation"] = interpolate(A6000_ADA, 22, 2130);
    device_map["NVIDIA RTX 4000 Ada Generation"] = interpolate(A6000_ADA, 48, 2175);
    device_map["NVIDIA RTX 4500 Ada Generation"] = interpolate(A6000_ADA, 56, 2580);
    device_map["NVIDIA RTX 5000 Ada Generation"] = interpolate(A6000_ADA, 100, 2550);
    device_map["NVIDIA RTX 5880 Ada Generation"] = interpolate(A6000_ADA, 110, 2460);
    device_map["NVIDIA RTX 6000 Ada Generation"] = A6000_ADA;

    device_map["NVIDIA A40"] = A40;
    device_map["NVIDIA L40"] = L40;
    device_map["NVIDIA L40S"] = L40S;
    device_map["NVIDIA L4"] = L4;

    device_map["NVIDIA GeForce RTX 4090"] = RTX_4090;
    device_map["NVIDIA GeForce RTX 4080 SUPER"] = interpolate(RTX_4090, 80, 2550);
    device_map["NVIDIA GeForce RTX 4080"] = RTX_4080;
    device_map["NVIDIA GeForce RTX 4070 Ti SUPER"] = interpolate(RTX_4070_Ti, 66, 2610);
    device_map["NVIDIA GeForce RTX 4070 Ti"] = RTX_4070_Ti;
    device_map["NVIDIA GeForce RTX 4070 SUPER"] = interpolate(RTX_4070, 56, 2475);
    device_map["NVIDIA GeForce RTX 4070 Laptop GPU"] = RTX_4070_Laptop_GPU;
    device_map["NVIDIA GeForce RTX 4070"] = RTX_4070;
    device_map["NVIDIA GeForce RTX 4060 Ti"] = interpolate(RTX_4070, 34, 2535);
    device_map["NVIDIA GeForce RTX 4060"] = interpolate(RTX_4070, 24, 2460);

    device_map["NVIDIA H100 PCIe"] = H100_PCI;
    device_map["NVIDIA H100 80GB HBM3"] = H100_SXM;
    device_map["NVIDIA H200"] = H200_SXM;
    device_map["NVIDIA B200"] = B200_HGX;
    device_map["NVIDIA B300 SXM6 AC"] = B300_SXM;

    device_map["NVIDIA GeForce RTX 5090"] = RTX_5090;
    device_map["NVIDIA GeForce RTX 5080"] = RTX_5080;
    device_map["NVIDIA GeForce RTX 5070 Ti"] = RTX_5070_Ti;
    device_map["NVIDIA GeForce RTX 5070"] = RTX_5070;
    device_map["NVIDIA GeForce RTX 5060 Ti"] = interpolate(RTX_5070, 36, 2572);
    device_map["NVIDIA GeForce RTX 5060"] = interpolate(RTX_5070, 30, 2497);

    device_map["NVIDIA RTX PRO 6000 Blackwell Server Edition"] = B6000_WS;
    device_map["NVIDIA GB10"] = GB10;
    return device_map;
}

sPerfSpecs get_device_perf(std::string_view device) {
    static std::unordered_map<std::string_view, sPerfSpecs> device_map = create_device_map();
    if (auto it = device_map.find(device); it != device_map.end()) {
        return it->second;
    } else {
        fprintf(stderr, "WARNING: unknown device %s\n", device.data());
        return sPerfSpecs{};
    }
}

// Cached measured peak rates in TFLOP/s; -1 means not yet measured
static float g_measured_bf16_peak_tflops = -1.0f;
static float g_measured_fp8_peak_tflops = -1.0f;

float get_peak_rate(const sPerfSpecs& spec, ETensorDType dtype) {
    switch (dtype) {
        case ETensorDType::FP32:
            return spec.TF32_TFlops;
        case ETensorDType::BF16:
            // Use measured value if available, otherwise fall back to spec sheet
            if (g_measured_bf16_peak_tflops > 0.0f) {
                return g_measured_bf16_peak_tflops;
            }
            return spec.BF16_TFlops;
        case ETensorDType::FP16:
            return spec.FP16_32_TFlops; // TODO ambiguous accumulator
        case ETensorDType::INT32:
        case ETensorDType::INT8:
            return spec.INT8_TFlops;
        case ETensorDType::FP8_E4M3:
        case ETensorDType::FP8_E5M2:
            // Use measured FP8 value if available, otherwise fall back to spec sheet
            if (g_measured_fp8_peak_tflops > 0.0f) {
                return g_measured_fp8_peak_tflops;
            }
            return spec.FP8_32_TFlops;
        case ETensorDType::BYTE:
        default:
            // Speed-of-light estimation is best-effort; never crash training for an unsupported dtype.
            return -1.0f;
    }
}

std::int64_t time_for_op_ns(const sPerfSpecs& spec, ETensorDType dtype, std::int64_t count) {
    // tera = 10^12; nano = 10^-9
    count /= 1000;
    float peak = get_peak_rate(spec, dtype);
    if (peak <= 0.0f) {
        return -1;
    }
    return static_cast<std::int64_t>(count / peak);
}

long estimate_speed_of_light(const char* device, const std::vector<std::pair<ETensorDType, long>>& ops) {
    sPerfSpecs spec = get_device_perf(device);
    if (!spec.Chip)
        return -1; // ¯\_(ツ)_/¯
    std::int64_t nanoseconds = 0;
    for (auto [op, count] : ops) {
        std::int64_t dt = time_for_op_ns(spec, op, count);
        if (dt < 0) return -1;
        nanoseconds += dt;
    }
    return nanoseconds;
}

float get_peak_rate(const char* device, ETensorDType dtype) {
    sPerfSpecs spec = get_device_perf(device);
    if (!spec.Chip)
        return -1; // ¯\_(ツ)_/¯
    return get_peak_rate(spec, dtype);
}

std::vector<std::pair<ETensorDType, long>> get_transformer_ops(long non_embedding_params, ETensorDType non_embedding_dtype, long embedding_params, ETensorDType embedding_dtype, long d_att, long n_layers, long ctx) {
    std::vector<std::pair<ETensorDType, long>> ops;
    ops.emplace_back(non_embedding_dtype, 6l * non_embedding_params);
    ops.emplace_back(embedding_dtype, 6l * embedding_params);
    ops.emplace_back(embedding_dtype, 6l * n_layers * d_att * ctx);
    return ops;
}

cublasLtHandle_t create_cublaslt_handle();
void destroy_cublaslt_handle(cublasLtHandle_t handle) noexcept;

double measure_real_peak() {
    constexpr int N = 32768;  // Use larger matrix for more accurate peak measurement
    constexpr size_t REQUIRED_MEMORY = 3 * 2ull * N * N + 32 * 1024 * 1024;  // 3 matrices + workspace

    // Check if we have enough free memory before allocating
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem < REQUIRED_MEMORY + 256 * 1024 * 1024) {  // Need 256MB headroom
        // Not enough memory for benchmark, skip and return 0 (will use spec sheet values)
        return 0.0;
    }

    nv_bfloat16* a;
    nv_bfloat16* b;
    nv_bfloat16* c;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), 2ll * N * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), 2ll * N * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c), 2ll * N * N));
    CUDA_CHECK(cudaMemset(a, 0b00010101, 2ll * N * N));
    CUDA_CHECK(cudaMemset(b, 0b00010101, 2ll * N * N));

    cublasLtHandle_t handle = create_cublaslt_handle();
    std::byte* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace), 32*1024*1024));


    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    // warmup
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    int trip_count = 0;
    while (true) {
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        CUDA_CHECK(cudaDeviceSynchronize());
        if(dt > 500) {
            break;
        }
        ++trip_count;
        matmul(c, a, b, nullptr, nullptr, nullptr, handle, workspace, 32 * 1024 * 1024,
               N, N, N, EMMTranspose::TN, false, nullptr);
    }

    // now, actual measurement
    CUDA_CHECK(cudaEventRecord(start_event));
    for(int i = 0; i < trip_count; ++i) {
        matmul(c, a, b, nullptr, nullptr, nullptr, handle, workspace, 32 * 1024 * 1024,
               N, N, N, EMMTranspose::TN, false, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms_total;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start_event, stop_event));

    std::int64_t ops_total = 2ll * N * N * N * trip_count;

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(workspace));
    destroy_cublaslt_handle(handle);

    double ops_per_sec = ops_total / ms_total * 1000;

    // Cache the measured peak in TFLOP/s for use by get_peak_rate()
    g_measured_bf16_peak_tflops = static_cast<float>(ops_per_sec / 1e12);

    return ops_per_sec;
}

double measure_real_peak_fp8() {
    constexpr int N = 32768;  // Use larger matrix for more accurate peak measurement
    // FP8: 2 × 1-byte matrices + 1 × 4-byte output matrix + workspace
    constexpr size_t REQUIRED_MEMORY = 2 * 1ull * N * N + 4ull * N * N + 32 * 1024 * 1024;

    // Check if we have enough free memory before allocating
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem < REQUIRED_MEMORY + 256 * 1024 * 1024) {  // Need 256MB headroom
        // Not enough memory for benchmark, skip and return 0 (will use spec sheet values)
        return 0.0;
    }

    __nv_fp8_e4m3* a;
    __nv_fp8_e4m3* b;
    float* c;
    float* scale_a;
    float* scale_b;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), 1ll * N * N));  // FP8 is 1 byte
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), 1ll * N * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c), 4ll * N * N));  // FP32 output
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&scale_a), sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&scale_b), sizeof(float)));
    CUDA_CHECK(cudaMemset(a, 0b00010101, 1ll * N * N));
    CUDA_CHECK(cudaMemset(b, 0b00010101, 1ll * N * N));
    float one = 1.0f;
    CUDA_CHECK(cudaMemcpy(scale_a, &one, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scale_b, &one, sizeof(float), cudaMemcpyHostToDevice));

    cublasLtHandle_t handle = create_cublaslt_handle();
    std::byte* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace), 32*1024*1024));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    // warmup
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    int trip_count = 0;
    while (true) {
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        CUDA_CHECK(cudaDeviceSynchronize());
        if(dt > 500) {
            break;
        }
        ++trip_count;
        matmul(c, a, b, static_cast<const float*>(nullptr), scale_a, scale_b, handle, workspace, 32 * 1024 * 1024,
               N, N, N, EMMTranspose::TN, false, nullptr);
    }

    // now, actual measurement
    CUDA_CHECK(cudaEventRecord(start_event));
    for(int i = 0; i < trip_count; ++i) {
        matmul(c, a, b, static_cast<const float*>(nullptr), scale_a, scale_b, handle, workspace, 32 * 1024 * 1024,
               N, N, N, EMMTranspose::TN, false, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms_total;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start_event, stop_event));

    std::int64_t ops_total = 2ll * N * N * N * trip_count;

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(scale_a));
    CUDA_CHECK(cudaFree(scale_b));
    CUDA_CHECK(cudaFree(workspace));
    destroy_cublaslt_handle(handle);

    double ops_per_sec = ops_total / ms_total * 1000;

    // Cache the measured peak in TFLOP/s for use by get_peak_rate()
    g_measured_fp8_peak_tflops = static_cast<float>(ops_per_sec / 1e12);

    return ops_per_sec;
}
