# Contributing Guide

Thanks for your interest in improving Sketch2Image.

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Coding Standards

- Use Python 3.10+.
- Keep scripts configurable through CLI flags (avoid hardcoded paths).
- Add concise docstrings to non-trivial functions.
- Preserve deterministic behavior where applicable (seeded randomness).

## Local Validation

Run a syntax check before opening a PR:

```bash
python -m py_compile script/process_dataset.py
```

## Pull Request Checklist

- Clear summary of what changed and why.
- Updated README/docs when behavior changes.
- No large binary/model files committed.
- Dataset outputs and local artifacts are excluded by `.gitignore`.

## Issue Ideas Welcome

Useful contributions include:

- Better evaluation metrics and experiment tracking.
- Cleaner inference scripts and model packaging.
- Benchmarks across multiple domains (faces, rooms, products).
