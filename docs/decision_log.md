# Decision Log

Engineering and tooling decisions for this pipeline, in the order made. Data decisions
are recorded in `notebooks/eda.ipynb`, not duplicated here.

## 1. Migrated to uv's project workflow, PyTorch pinned to CPU
**Context.** PyPI's default torch wheel is a CUDA build, inconsistent with a
GPU-free, reproducible-anywhere environment. Pinning it via a manual install
step complicates documentation and reproduction of the project, and `requirements.txt`
directives didn't work reliably under uv.

**Decision.** Moved dependencies into `pyproject.toml`, with torch pinned to
a dedicated CPU-only index via uv's source-pinning mechanism. `uv.lock` is
the committed, resolved lockfile. `requirements.txt` was removed.

**Consequence.** Setup is one command (`uv sync`). A transitive dependency
also pulled in a broken old `pyyaml` version during this migration, fixed
with a direct dependency and a forced override.
