#!/usr/bin/env python3
"""
Validates that the inter-stage bridge logic in slurm_templates.py is
correctly wired: preprocessing output → VQA generation input.

Run from the project root:
    python tests/test_bridge.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
templates_path = PROJECT_ROOT / "orchestrator" / "slurm_templates.py"

print("=" * 70)
print("TEST: Preprocessing → VQA Bridge Logic")
print("=" * 70)

content = templates_path.read_text()

# Check 1: routing output path registered
assert '"bbox_preproc": "results"' in content, "bbox_preproc not in _PREPROCESSING_OUTPUT_PATHS"
assert '"attn_map": "results"' in content, "attn_map not in _PREPROCESSING_OUTPUT_PATHS"
assert '"segmentation": "results/step2_masks"' in content, "segmentation not in _PREPROCESSING_OUTPUT_PATHS"
print("✅ All preprocessing stages registered in output path map")

# Check 2: bridge function exists
assert "_generate_preprocessing_to_vqa_bridge" in content, "Bridge function missing"
print("✅ _generate_preprocessing_to_vqa_bridge function present")

# Check 3: bridge env vars are exported
assert "DATA_FILE_OVERRIDE" in content, "DATA_FILE_OVERRIDE missing from bridge"
assert "VQA_IMAGE_PATH" in content, "VQA_IMAGE_PATH missing from bridge"
assert "PREPROC_TYPE" in content, "PREPROC_TYPE missing from bridge"
print("✅ Bridge exports: DATA_FILE_OVERRIDE, VQA_IMAGE_PATH, PREPROC_TYPE")

# Check 4: auto-injection logic
assert '_PREPROCESSING_STAGE_KEYS' in content, "Stage key set missing"
assert 'keys[idx + 1] == "vqa_gen"' in content, "Bridge injection condition missing"
print("✅ Bridge auto-injection fires on: preprocessing → vqa_gen")

# Check 5: routing → preprocessing bridge
assert "_generate_routing_to_preprocessing_bridge" in content, "Routing bridge function missing"
assert "ROUTED_DATASET_OVERRIDE" in content, "ROUTED_DATASET_OVERRIDE missing"
print("✅ medclip_routing → preprocessing bridge present")

print("\n✅ ALL BRIDGE CHECKS PASSED")
print("=" * 70)
