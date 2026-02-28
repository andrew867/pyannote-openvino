# Phase 5 – Testing and Validation

## SPEC
- Build a lightweight pytest suite that exercises the OV pipeline stack using synthetic waveforms, avoiding large audio downloads and torchcodec audio decoding by feeding preloaded tensors.
- Ensure the suite confirms the loaded IR models produce a `pyannote.audio.pipelines.speaker_diarization.DiarizeOutput` whose `.speaker_diarization` is a `pyannote.core.Annotation`.
- Make the tests fast (<15s) so they can run inside CI and preview pipelines without significant resource cost.

## TESTS
- `python -m pytest` executes `tests/test_ov_pipeline.py`, verifying the pipeline accepts `{"waveform": tensor, "sample_rate": ...}` inputs, runs the OV models, and returns a well-formed annotation.
- Tests must pass on every platform targeted by the CI (Linux + Windows on GitHub, Linux for GitLab).

## PLAN
1. Create `tests/test_ov_pipeline.py` that loads `OVSpeakerDiarization.from_pretrained(device="CPU")`, feeds 1 second of random noise, and asserts the return value contains a `speaker_diarization` annotation.
2. Update `pyproject.toml` to register a `[project.optional-dependencies]` entry like `test = ["pytest>=8.0"]` so CI can install the suite alongside the runtime extras.
3. Mention the pytest command and any required extras in `README.md` so contributors know how to run the tests locally (e.g., `python -m pip install -e .[stt,test]` followed by `python -m pytest`).
4. Run `python -m pytest` locally, capturing the results for documentation and verifying the suite finishes quickly.

## CODE
- `tests/test_ov_pipeline.py`: single pytest module that checks the OV pipeline returns a valid diarization annotation when given a random waveform.
