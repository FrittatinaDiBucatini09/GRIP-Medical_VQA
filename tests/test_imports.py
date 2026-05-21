#!/usr/bin/env python3
"""
Module import validation for the medclip_routing preprocessing stage.

Verifies that the routing module and all its dependencies can be imported
correctly inside the Docker environment.

Run from the project root:
    docker run --rm ... python tests/test_imports.py
"""
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "preprocessing" / "medclip_routing" / "src"
sys.path.insert(0, str(SRC_PATH))

print("=" * 70)
print("TEST: medclip_routing — Module & Dependency Validation")
print("=" * 70)

results = {}

try:
    import utils
    results["utils"] = "PASS"
    print("✅ utils imported successfully")
except Exception as e:
    results["utils"] = f"FAIL: {e}"
    print(f"❌ utils import failed: {e}")

try:
    from utils import (
        load_scispacy,
        load_gemma,
        load_biomed_clip,
        evaluate_query_quality,
        expand_query,
        generate_cam_bbox,
        CAMWrapper,
        reshape_transform,
    )
    results["utils functions"] = "PASS"
    print("✅ All utils functions available")
except Exception as e:
    results["utils functions"] = f"FAIL: {e}"
    print(f"❌ Utils functions check failed: {e}")

try:
    import main_routing
    results["main_routing"] = "PASS"
    print("✅ main_routing imported successfully")
except Exception as e:
    results["main_routing"] = f"FAIL: {e}"
    print(f"❌ main_routing import failed: {e}")

deps_ok = True
dep_results = []
for dep in ["torch", "numpy", "pandas", "PIL", "cv2", "spacy", "transformers", "open_clip"]:
    try:
        __import__(dep)
        dep_results.append(f"  ✅ {dep}")
    except ImportError as e:
        dep_results.append(f"  ❌ {dep}: {e}")
        deps_ok = False

results["dependencies"] = "PASS" if deps_ok else "FAIL"
print(f"\n📦 Dependencies:")
for res in dep_results:
    print(res)

print("\n" + "=" * 70)
print("SUMMARY:")
for test, result in results.items():
    print(f"  {test}: {result}")
print("=" * 70)

if any("FAIL" in v for v in results.values()):
    sys.exit(1)
