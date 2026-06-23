"""
Phase 0 environment check.

Verifies that the core ML dependencies are installed and reports which
compute device is available (CUDA GPU, Apple Silicon MPS, or CPU fallback).

Run after `pip install -r requirements.txt`:
    python environment_check.py
"""
import importlib
import platform
import sys

REQUIRED = [
    "torch",
    "numpy",
    "scipy",
    "sklearn",
    "mne",
    "pyedflib",
    "mlflow",
    "matplotlib",
]


def check_imports() -> dict[str, str]:
    results = {}
    for mod_name in REQUIRED:
        try:
            mod = importlib.import_module(mod_name)
            results[mod_name] = getattr(mod, "__version__", "unknown version")
        except ImportError as exc:
            results[mod_name] = f"MISSING ({exc})"
    return results


def check_device() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed - cannot check device"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA available: {name}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple Silicon MPS available (no CUDA GPU)"
    return "No GPU detected - CPU only (fine for pipeline dev, too slow for full diffusion training)"


def main() -> int:
    print(f"Python: {platform.python_version()} ({platform.system()} {platform.machine()})")
    print()

    print("Dependency check:")
    results = check_imports()
    ok = True
    for name, status in results.items():
        marker = "OK" if "MISSING" not in status else "FAIL"
        if marker == "FAIL":
            ok = False
        print(f"  [{marker:4}] {name:12} {status}")

    print()
    print("Compute device:")
    print(f"  {check_device()}")

    print()
    if ok:
        print("All required packages are importable. Ready for Phase 0 work.")
        return 0
    else:
        print("Some packages are missing. Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
