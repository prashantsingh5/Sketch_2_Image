"""Environment doctor for Sketch2Image.

Checks Python version, key folders, and optional dependencies.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 10)
CORE_MODULES = ["numpy", "cv2", "PIL"]
OPTIONAL_MODULES = ["torch", "torchvision", "torchmetrics", "lpips", "streamlit", "skimage"]


def check_python() -> tuple[bool, str]:
    ok = sys.version_info >= REQUIRED_PYTHON
    return ok, f"Python {sys.version.split()[0]} (required >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]})"


def check_modules(modules: list[str]) -> tuple[list[str], list[str]]:
    present, missing = [], []
    for name in modules:
        try:
            importlib.import_module(name)
            present.append(name)
        except Exception:
            missing.append(name)
    return present, missing


def check_paths() -> list[str]:
    checks = [
        Path("README.md"),
        Path("script/process_dataset.py"),
        Path("script/evaluate_metrics.py"),
        Path("demo/app.py"),
    ]
    missing = [str(p) for p in checks if not p.exists()]
    return missing


def main() -> None:
    print("Sketch2Image Doctor")
    print("-" * 24)

    py_ok, py_msg = check_python()
    print(f"[{'ok' if py_ok else 'fail'}] {py_msg}")

    missing_paths = check_paths()
    if missing_paths:
        print("[fail] Missing files:")
        for item in missing_paths:
            print(f"  - {item}")
    else:
        print("[ok] Core project files present")

    core_present, core_missing = check_modules(CORE_MODULES)
    print(f"[ok] Core modules present: {', '.join(core_present) if core_present else 'none'}")
    if core_missing:
        print(f"[warn] Core modules missing: {', '.join(core_missing)}")

    opt_present, opt_missing = check_modules(OPTIONAL_MODULES)
    print(f"[ok] Optional modules present: {', '.join(opt_present) if opt_present else 'none'}")
    if opt_missing:
        print(f"[info] Optional modules missing: {', '.join(opt_missing)}")

    if py_ok and not missing_paths and not core_missing:
        print("\nResult: environment looks healthy for standard usage.")
    else:
        print("\nResult: setup needs fixes. See missing items above.")


if __name__ == "__main__":
    main()
