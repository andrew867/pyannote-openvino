# Phase 4 – Pipeline Integration

## SPEC
- Implement `OVSegmentationModel` and `OVEmbeddingModel` that wrap OpenVINO runtime models and expose the same interface as pyannote.audio's torch-based models.
- Create `OVSpeakerDiarization` pipeline class mirroring `pyannote.audio.pipeline.Pipeline`, supporting `.from_pretrained`, `.to`, and `__call__` with audio files.
- Support device selection strings (`CPU`, `GPU`, `AUTO`, `GPU.0`). Ensure model initialization loads appropriate OpenVINO device and precision.
- Document configuration options (model paths, tokenizer/feature pipeline, device) and fallback to ONNX when needed.

## TESTS
- Instantiate `OVSpeakerDiarization` with sample OV segmentation and embedding models and run inference on dummy audio to ensure pipeline runs without errors.
- Compare diarization output format (timeline segments with speaker IDs) against reference pipeline on a fixed clip to ensure compatibility.
- Verify `.to(device)` or equivalent correctly reconfigures device for runtime.

## PLAN
1. Examine pyannote.audio's `Model` and `Pipeline` base classes to understand required methods (e.g., `forward`, `__call__`).
2. Implement wrappers that load OV model file via `openvino.runtime.Core`. Provide inference methods returning expected tensor shapes.
3. Build pipeline class using pyannote `Pipeline` API or replicating necessary logic, replacing torch model calls with OV wrappers.
4. Ensure device selection and precision (FP32/FP16) propagate to runtime configuration and conversions.
5. Add documentation in README or pipeline module describing usage and drop-in replacement scenario.

## CODE
- `pyannote_openvino/ov_model.py`: wrappers for segmentation and embedding models.
- `pyannote_openvino/pipeline.py`: pipeline class building segmentation + embedding + clustering (reuse CPU clustering) to produce diarization output.
