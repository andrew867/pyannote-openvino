# Phase 3 – OpenVINO IR Conversion

## SPEC
- Convert ONNX segmentation and embedding models to OpenVINO IR (XML/BIN) for CPU and DLA/GPU if available.
- Document precision choices (FP32, FP16) and device-specific settings. Explain how to specify target device when running the conversion.
- Provide scripts that wrap OpenVINO Model Optimizer (or `optimum-intel` CLI) to convert and store IR files in `models/ov/`.
- Acceptance: `convert_to_ov.py` produces IR models, the script can be invoked with device overrides, and documentation lists required OpenVINO version commands.

## TESTS
- Run conversion script to produce IR for both models in FP32 and verify the files exist and loadable via `openvino.runtime.Core.read_model`.
- Run simple inference on IR models via OpenVINO runtime to confirm they produce outputs and shapes match ONNX references.
- Benchmark script outputs latency for torch CPU, OV CPU, OV GPU (if available) and reports in logs.

## PLAN
1. Review ONNX files from Phase 2 and determine conversion options (input shapes, dynamic axes, precision).
2. Write `convert_to_ov.py` using `ov.utils` (Model Optimizer) or `optimum.intel.openvino` conversion APIs to generate IRs; support CLI arguments for device and precision.
3. Develop helper script `scripts/phase3/validate_ov.py` that loads IR via OpenVINO runtime, runs dummy inference, and prints output shapes.
4. Create benchmarking script to compare torch CPU vs OV CPU vs OV GPU using simple repeated runs and report average latencies.
5. Document precision tradeoffs and any special flags needed to avoid unsupported ops.

## CODE
- `convert_to_ov.py`: handles CLI, converts ONNX to OV IR for segmentation and embedding, uses target precision, and stores outputs under `models/ov/`.
- Benchmark/resolution script to measure latency.
