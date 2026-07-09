# synthetic-health-data-pipeline
End-to-end synthetic health data pipeline: generation (CART &amp; CTGAN), and evaluation across fidelity, utility, and disclosure risk.

## Status

Environment setup in progress: Python3.11 environment complete. R environment pending.

## Setup

### Python (3.11, CPU-only, via uv)

```bash
uv venv --python 3.11
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Plain venv/pip works identically if uv is unavailable:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The environment is CPU-only by design: no GPU, CUDA, or vendor-specific
backend is required, keeping the pipeline reproducible on any machine.
PyTorch is pinned to the CPU build via the extra index declared in
`requirements.txt`.
