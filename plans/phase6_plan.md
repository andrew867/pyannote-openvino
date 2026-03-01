# Phase 6 – Packaging and Documentation

## SPEC
- Deliver a packaged `pyannote-openvino` distribution (`pyproject.toml` + `README.md` + MIT license) that can be installed via `pip`/`pip install -e .`.
- Document the CI/CD setup (GitHub + GitLab) so maintainers know how tests, release builds, and artifacts are produced.
- Provide release pipelines that build the wheel/tarball on tags and publish the artifacts (GitHub release + GitLab artifacts).

## TESTS
- `python -m build` succeeds after the tests and checkout, confirming the project metadata is valid.
- GitHub Actions `ci.yml` passes on every push/PR, and the `release.yml` job runs on tags, builds the package, and uploads the artifacts.
- GitLab CI runs the same `pytest` command and builds the package on tag pushes, making the `dist/` output available as an artifact.

## PLAN
1. Ensure the README sections describe how to install the package, run the tests, and invoke the new `docs/transcribe_v4.py` CLI.
2. Add `.github/workflows/ci.yml` to run `pytest` on pushes/PRs and `.github/workflows/release.yml` to rebuild/test on tags, create a GitHub release, and upload the dist files.
3. Add `.gitlab-ci.yml` with `test` and `release` stages that mirror the GitHub flow, publishing `dist/` artifacts on tags.
4. Update `.gitignore` so build artifacts (`dist/`, `build/`, `*.egg-info`) stay out of git.
5. Document in README how to trigger the pipelines and how releases are produced from tags.
6. Run `python -m build` locally after tests pass to verify packaging.

## CODE
- `.github/workflows/ci.yml` – installs `[stt,test]` extras and runs `python -m pytest`.
- `.github/workflows/release.yml` – runs on tag pushes, installs `[stt,test,build]`, runs tests, builds the package, and uses `softprops/action-gh-release` to publish dist files.
- `.gitlab-ci.yml` – Linux pipeline with `test` and `release` jobs, producing artifacts for `dist/`.
