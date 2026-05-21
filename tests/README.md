# Tests

This directory contains the test suite for the GRIP Medical VQA pipeline.
Tests are organised by the component they exercise and can be run without
GPU access (heavy model dependencies are mocked where necessary).

## Test Overview

| File | Component | Requires GPU? |
|------|-----------|:---:|
| `test_orchestrator.py` | Orchestrator dry-run — pipeline config & meta-job generation | No |
| `test_bridge.py` | Inter-stage bridge logic in `slurm_templates.py` | No |
| `test_imports.py` | medclip_routing module & dependency validation | No* |
| `test_judge_parsing.py` | LLM Judge response parser (unit tests) | No |
| `test_end_to_end_judge.py` | LLM Judge pipeline — end-to-end with mocked vLLM | No |
| `test_error_handling.py` | VQA pipeline error & edge-case handling | No |
| `test_integration_pipeline.py` | VQA generation pipeline integration tests | No |

\* `test_imports.py` is designed to be run **inside the Docker container** where
all Python dependencies (`torch`, `spacy`, `open_clip`, etc.) are installed.

## Running the Tests

### Lightweight tests (no Docker required)

```bash
# From the project root
python tests/test_bridge.py
python tests/test_orchestrator.py        # generates a dry-run meta-job script
python tests/test_judge_parsing.py
python tests/test_end_to_end_judge.py
python tests/test_error_handling.py
python tests/test_integration_pipeline.py
```

### Docker environment test (medclip_routing)

```bash
cd preprocessing/medclip_routing
export CUDA_VISIBLE_DEVICES=0
docker run --rm \
    -v $PWD:/workspace \
    med_routing_project:latest \
    python /workspace/../../tests/test_imports.py
```

### Full orchestrator dry-run

Verifies that a BBox → AttnMap → Segmentation → VQA pipeline can be
configured programmatically and that the SLURM meta-job script is generated
correctly (no cluster submission occurs):

```bash
python tests/test_orchestrator.py
```

The generated script is written to `orchestrator_runs/run_<timestamp>/meta_job.sh`.
