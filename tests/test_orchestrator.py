#!/usr/bin/env python3
"""
Headless orchestrator dry-run test.
Programmatically builds a multi-stage pipeline (BBox → AttnMap → Seg → VQA)
and verifies that the meta-job script is generated without errors.

Run from the project root:
    python tests/test_orchestrator.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "orchestrator"))

from orchestrator import STAGE_REGISTRY, StageConfig, run_pipeline


def main():
    print("[TEST] Starting headless orchestrator dry-run...")

    # Select stages by registry key
    key_order = {s.key: i for i, s in enumerate(STAGE_REGISTRY)}
    target_keys = ["bbox_preproc", "attn_map", "segmentation", "vqa_gen"]
    selected_stages = [STAGE_REGISTRY[key_order[k]] for k in target_keys]

    print(f"[TEST] Stages: {[s.name for s in selected_stages]}")

    configs = [
        StageConfig(config_file="configs/gemex/exp_01_vqa.conf"),   # bbox_preproc
        StageConfig(config_file="configs/gemex/exp_01_vqa.conf"),   # attn_map
        StageConfig(env_overrides={"TARGET_MODE": "all", "LIMIT": "50"}),  # segmentation
        StageConfig(config_file="configs/generation/hard_coded_gen.conf"),  # vqa_gen
    ]

    dataset = "gemex_VQA_mimic_mapped.csv"
    print(f"[TEST] Dataset: {dataset}")
    print("[TEST] Running in dry-run mode (no SLURM submission)...")

    run_pipeline(
        stages=selected_stages,
        stage_configs=configs,
        dataset_override=dataset,
        dry_run=True,
    )

    print("[TEST] Dry-run complete. Check orchestrator_runs/ for the generated meta-job script.")


if __name__ == "__main__":
    main()
