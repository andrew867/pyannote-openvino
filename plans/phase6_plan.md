# Phase 6 – Packaging and Documentation

## SPEC
- Package project as `pyannote_openvino` with clean public API, entry points, and dependency listing.
- Provide README covering installation, usage, device selection, benchmarks, and troubleshooting.
- Include `requirements.txt`, optional `pyproject.toml`, MIT license, and example quickstart script.
- Set up GitHub Actions workflow for CPU-based tests/deploy.

## TESTS
- Verify package can be installed locally via `pip install -e .` and imports `pyannote_openvino` with OV pipeline classes.
- Run README example script to ensure usage instructions are valid after packaging (calls pipeline, prints diarization timeline).
- Ensure GitHub Actions workflow passes (at least by running simulated commands locally matching steps) or provide log of manual steps.

## PLAN
1. Define `setup.cfg`/`pyproject.toml` or `setup.py` describing package metadata, dependencies, entry points for OV pipeline classes.
2. Write README detailing installation (with optional GPU support), usage example (OV pipeline drop-in), benchmarking results summary, and troubleshooting tips.
3. Provide example scripts referencing pipeline classes and showing audio inference results.
4. Create MIT LICENSE file and `requirements.txt` reflecting minimal dependencies.
5. Add GitHub Actions workflow file (maybe `.github/workflows/ci.yml`) running tests on CPU (e.g., pyro tests) and packaging checks.
6. Document `OVSpeakerDiarization` usage in README and include e.g., CLI instructions for running end-to-end test.

## CODE
- Package stub modules under `pyannote_openvino/__init__.py`. Expose `OVSpeakerDiarization` class.
- `README.md` with detail sections.
- `.github/workflows/ci.yml` with at least install/test steps.
- `scripts/e2e_test.py` that loads pipeline, runs on sample audio, and prints DER (for final step).
