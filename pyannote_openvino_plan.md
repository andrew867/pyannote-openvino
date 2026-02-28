# pyannote-openvino
## OpenVINO acceleration for pyannote.audio speaker diarization pipelines

### Goal
Convert the pyannote speaker diarization 3.1 pipeline to run inference on Intel GPUs via OpenVINO, preserving output fidelity and the existing pyannote API surface so it's a drop-in replacement.

---

## Background and scope

pyannote/speaker-diarization-3.1 is a multi-model pipeline:

1. **Segmentation model** - `pyannote/segmentation-3.0` - a SincNet+LSTM transformer that produces frame-level speaker activity and change point scores. This is the heaviest inference component.
2. **Embedding model** - `pyannote/wespeaker-voxceleb-resnet34-LM` - a ResNet34 that produces 256-dim speaker embeddings per segment.
3. **PLDA scoring** - `pyannote/speaker-diarization-community-1` - classical PLDA (not a neural net), runs on CPU, not a conversion target.
4. **Clustering** - agglomerative hierarchical clustering on embeddings, CPU-bound, not a conversion target.

Conversion targets are (1) and (2) only. These account for the majority of inference time.

---

## Phases

### Phase 1 - Environment and model audit

**Goal:** Understand the exact model architectures, input/output shapes, and confirm exportability before writing any conversion code.

Deliverables:
- Documented input/output tensor shapes for segmentation and embedding models
- Confirmed ONNX export works for both models with sample inputs
- List of any dynamic axes, control flow, or ops that need special handling
- Environment spec: Python version, torch version, openvino version, optimum-intel version

### Phase 2 - ONNX export

**Goal:** Export both models to ONNX as an intermediate step before OpenVINO IR conversion.

Deliverables:
- `export_segmentation.py` - exports segmentation model to ONNX with correct dynamic axes for variable length audio
- `export_embedding.py` - exports embedding model to ONNX
- Validation script that runs both ONNX models against the original torch models on sample inputs and checks output numerical agreement within tolerance
- Documented known export issues and workarounds

### Phase 3 - OpenVINO IR conversion

**Goal:** Convert ONNX models to OpenVINO IR format and validate on Intel GPU.

Deliverables:
- `convert_to_ov.py` - converts both ONNX models to OV IR using `mo` or `optimum-cli`
- Benchmark script comparing latency: torch CPU vs OV CPU vs OV GPU for both models
- Numerical validation confirming OV outputs match ONNX outputs within tolerance
- Document any precision issues (FP16 vs FP32) and recommended settings per device

### Phase 4 - Pipeline integration

**Goal:** Wrap the OV models in pyannote-compatible interfaces so the existing pipeline can use them without modification.

Deliverables:
- `OVSegmentationModel` class - wraps OV IR segmentation model, implements pyannote Model interface
- `OVEmbeddingModel` class - wraps OV IR embedding model, implements pyannote Model interface
- `OVSpeakerDiarization` pipeline class - drop-in replacement for `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")` that uses OV models internally
- Support for device selection: `CPU`, `GPU`, `GPU.0`, `AUTO`

### Phase 5 - Testing and validation

**Goal:** Confirm the accelerated pipeline produces correct diarization output on real audio.

Deliverables:
- Test suite with sample audio clips covering: single speaker, two speakers, three speakers, overlapping speech, silence, noise
- DER (diarization error rate) comparison between original pipeline and OV pipeline on test clips - must be within acceptable tolerance (target: <5% relative degradation)
- Regression tests for edge cases: very short clips, very long clips, clips with no speech
- CI-ready test runner

### Phase 6 - Packaging and documentation

**Goal:** Make it usable by anyone with an Intel GPU.

Deliverables:
- `pyannote_openvino` Python package with clean public API
- `README.md` with install instructions, usage examples, supported devices, benchmark results
- `requirements.txt` and optional `pyproject.toml`
- Example script matching the pyannote quickstart but using OV backend
- MIT LICENSE file
- GitHub Actions workflow for basic CI on CPU (GPU CI is optional/self-hosted)

---

## API design target

```python
# existing pyannote usage
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token="hf_...")
diarization = pipeline("audio.wav")

# target drop-in replacement
from pyannote_openvino import OVSpeakerDiarization
pipeline = OVSpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", token="hf_...", device="GPU")
diarization = pipeline("audio.wav")
```

Same output format. Same RTTM-compatible result. Faster on Intel hardware.

---

## Known risks and open questions

- Dynamic sequence lengths in the segmentation model may require careful ONNX export axis handling
- Embedding model ResNet34 should export cleanly but needs validation
- FP16 on Intel iGPU may introduce enough numerical drift to affect clustering - needs measurement
- pyannote internal APIs are not stable - pin to a specific version
- PLDA and clustering remain on CPU and will still be the bottleneck for very long audio - out of scope for this project but worth noting

---

## Codex prompt instructions

For each phase above, generate:
1. **SPEC** - detailed specification of inputs, outputs, interfaces, and acceptance criteria
2. **TESTS** - test cases that prove the spec is met, written before implementation
3. **PLAN** - step by step implementation plan with dependencies called out
4. **CODE** - implementation

Work one phase at a time. Do not start Phase N+1 until Phase N tests pass.
