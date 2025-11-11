# Post-Training Quantization (PTW)

## Concepts

Quantization is a method used to make machine learning models more efficient by converting their internal parameters (weights and activations) from high-precision formats (such as 32-bit floating point) to lower-precision formats (such as 8-bit integer or 4-bit floating point).

By using fewer bits to represent data, quantized models take up less memory, can run faster, and may use less power during inference.

A quantization format consists of the precision format, the block format, the scaling factor and the calibration method.

### Precision Format
The precision format specifies how many bits are used to represent each quantized value. Integer formats use a sign bit and mantissa bits, while floating-point formats use a sign bit, exponent bits, and mantissa bits. This determines the range and accuracy of the quantized values.

### Block Format
The block format determines how a tensor is split into smaller parts (blocks) for quantization, specifically for sharing scaling factors. The main types are:

- **Per-tensor quantization**: The entire tensor is treated as one block, using a single scaling factor.
- **Per-channel quantization**: Each channel (e.g., output channel in a neural network layer) is quantized separately, each with its own scaling factor.
- **Per-block quantization**: The tensor is divided into fixed-size blocks (often along the channel dimension), and each block gets its own scaling factor.

For very low-bit quantization (like 4-bit), using smaller blocks (per-block quantization) helps maintain model accuracy.

Weight and activation may share different precision and block formats. For example, in GPTQ and AWQ, the weight is quantized to 4-bit while activation stays in high precision. Weight-only quantization is helpful for bandwidth-constrained scenarios, while weight and activation quantization can reduce both bandwidth and computation cost.

### Scaling Factor
The scaling factor is a number (usually a floating-point value) that determines how the original high-precision values are converted to lower-precision values during quantization. It is used to compress the range of the original data so it fits within the range allowed by the quantized format. All values within the same block share the same scaling factor. The scaling factor is determined during calibration, a process that analyzes the data to find the best value for minimizing accuracy loss.

### Calibration Method
The calibration method determines how scaling factors are chosen and, in some cases, how weights are adjusted to minimize accuracy loss after quantization. The simplest method, max calibration, sets the scaling factor based on the largest value in the tensor and just rounds weights to the nearest quantized value. More advanced methods, like [Entropy Calibration](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/EntropyCalibrator.html), [SmoothQuant](https://arxiv.org/abs/2211.10438), [AWQ](https://arxiv.org/abs/2306.00978), and [SVDQuant](https://arxiv.org/pdf/2411.05007), use statistical analysis or optimization to find scaling factors that better preserve model accuracy.

## Supported Formats
The following quantization formats are supported for post-training quantization:
- **FP8 (W8A8)** (NVIDIA Hopper and Blackwell GPUs): quantizes the weights and activations to FP8 
- **AWQ**: a weight-only quantization method, commonly used with INT4 weights and FP16 activations (INT4 W4A16)
- **GPTQ**: a weight-only quantization method, commonly used with INT4 weights and FP16 activations or INT8 weights and INT8 activations  (INT4 W4A16, INT8 W8A8)
- **NVFP4** (Nvidia Blackwell GPUs): quantizes the weights and activations to FP4, optimized for Blackwell architecture

## Best Practices
For small-batch inference (batch size ≤ 4), the main bottleneck is memory bandwidth, as loading weights from GPU memory is the limiting factor. In this case, using weight-only quantization methods like INT4 AWQ or INT4-FP8 AWQ improves performance by reducing memory usage.

For large-batch inference (batch size ≥ 16), both memory bandwidth and computation become important. Here, quantizing both weights and activations and using lower-precision computation kernels is recommended. The best quantization method may depend on the specific model.

FP8 quantization is generally preferred due to minimal accuracy loss and good performance. If FP8 is not sufficient, try INT4-FP8 AWQ. For older GPUs (Ampere or earlier), use INT4 AWQ or INT8 SQ.

Depending on your specific use case, you may prioritize different trade-offs between accuracy, inference speed, and calibration time. The table below highlights these trade-offs to help you select the most suitable quantization method for your needs:

| Quantization Method           | Inference Speedup (Small Batch) | Inference Speedup (Large Batch) | Accuracy | Details                                                                                                                                                                                                                                           |
|-------------------------------|---------------------------------|---------------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FP8                           | Average                         | Average                         | High     | - FP8 per-tensor weight & activation quantization with min-max calibration. <br/> - Compresses FP16/BF16 model to 50% of original size <br/>- Calibration time: minutes                                                                           |
| INT8 SmoothQuant              | Average                         | Average                         | Average  | - 8-bit integer quantization with a variant of SmoothQuant calibration. <br/> - Per-channel weight quantization, per-tensor activation quantization. <br/> - Compresses FP16/BF16 model to 50% of original size <br/> - Calibration time: minutes |
| INT4 Weights only AWQ (W4A16) | High                            | Low                             | High     | - 4-bit integer group-wise/block-wise weight only quantization with AWQ calibration. <br/> - Compresses FP16/BF16 model to 25% of original size.<br/> - Calibration time: tens of minutes                                                         |
| INT4-FP8 AWQ (W4A8)           | High                            | Average                         | High     | - 4-bit integer group-wise/block-wise weight quantization, FP8 per-tensor activation quantization & AWQ calibration.<br/> - Compresses FP16/BF16 model to 25% of original size.- Calibration time: tens of minutes                                |