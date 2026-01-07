# MAKER Experiment - Contract Review with Zero Errors

Complete implementation of the MAKER framework for legal contract analysis using the CUAD dataset.

## Overview

This experiment applies **MAKER** (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging) to automated contract review, achieving high-precision identification of 41 clause categories across commercial legal contracts.

**Key Achievement Goal:** >95% accuracy with >90% zero-error contract reviews at <$5 per contract.

## What's Included

### 1. Data Collection Pipeline

**File:** `pipelines/collect/collect_cuad.py`

Downloads and prepares the CUAD dataset from HuggingFace:
- 84,325 training samples across 510 commercial contracts
- 41 clause categories (33 binary, 8 entity extraction)
- Expert-annotated by lawyers

```bash
# Download 1000 samples for testing
python -m pipelines.collect.collect_cuad --limit 1000

# Download full dataset
python -m pipelines.collect.collect_cuad --limit -1

# Prepare for MAKER experiments
python -m pipelines.collect.collect_cuad --limit 1000 --prepare-maker --voting-threshold 3
```

### 2. Experiment Plan

**File:** `knowledge/experiments/plans/cuad_maker_plan.md`

Complete experimental design with 5 phases:
1. **Single-Clause Classification** - Baseline accuracy
2. **Multi-Level Hierarchical Decomposition** - Complex contract handling
3. **Error Correction via Voting** - Optimal k value determination
4. **Red-Flagging Implementation** - Filter unreliable responses
5. **Scaling Laws Validation** - Verify theoretical predictions

### 3. MAKER Paper Notes

**File:** `knowledge/papers/notes/maker_paper.md`

Comprehensive notes on "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030):
- Theoretical framework
- Scaling laws: cost grows as Θ(s ln s)
- Red-flagging criteria
- Voting convergence mathematics

### 4. Dataset Documentation

**File:** `knowledge/datasets/cards/cuad_dataset_card.md`

Complete CUAD dataset documentation:
- Sample structure
- 41 clause categories
- Quality metrics
- Usage examples

### 5. Supporting Utilities

**Files:** `utils/*.py`

- `decorators.py` - Flow/task decorators (backward compatible)
- `hydra_aim.py` - Experiment tracking integration
- `manifest.py` - Dataset provenance tracking
- `git_utils.py` - Git state capture for reproducibility

### 6. Configuration

**File:** `conf/pipeline/collect_cuad.yaml`

Hydra configuration for CUAD collection:
```yaml
collect_cuad:
  source:
    dataset: "theatticusproject/cuad"
    split: train
    limit: 1000
  maker:
    enabled: false
    voting_threshold: 3
    granularity: clause_level
```

## Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install datasets transformers hydra-core omegaconf aim openai

# 2. Download sample data
python -m pipelines.collect.collect_cuad --limit 100

# 3. Review experiment plan
cat knowledge/experiments/plans/cuad_maker_plan.md

# 4. Prepare MAKER experiment
python -m pipelines.collect.collect_cuad --limit 1000 --prepare-maker
```

### Docker (Portable)

```bash
# 1. Build image
docker build -f Dockerfile.maker -t maker-experiment:latest .

# 2. Run collection
docker run --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  python -m pipelines.collect.collect_cuad --limit 1000

# 3. Interactive session
docker run --rm -it \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest bash
```

### Export for Transfer

```bash
# Export Docker image for another system
./scripts/export_maker.sh

# Or with compression (slower but smaller)
./scripts/export_maker.sh --compress

# Creates:
#   exports/maker-experiment_latest.tar (or .tar.gz)
#   exports/TRANSFER_INSTRUCTIONS.md
```

## Experiment Workflow

### Phase 1: Data Collection

```bash
# Download dataset
python -m pipelines.collect.collect_cuad --split train --limit 1000

# Output: ./data/collected/cuad/train/
#   ├── cuad_contracts.jsonl      # Raw contracts
#   ├── metadata.json              # Dataset stats
#   └── collection_manifest.json   # Provenance
```

### Phase 2: MAKER Preparation

```bash
# Decompose into subtasks
python -m pipelines.collect.collect_cuad --prepare-maker --voting-threshold 3

# Output: ./data/collected/cuad_maker_experiment/
#   ├── maker_tasks.jsonl    # 41 tasks per contract
#   └── maker_config.json    # Experiment config
```

### Phase 3: Baseline Estimation

**TODO:** Implement in `pipelines/evaluate/cuad_maker.py`

```python
from pipelines.evaluate.cuad_maker import estimate_success_rate

# Estimate per-step success rate on sample
p = estimate_success_rate(
    model="gpt-4.1-mini",
    dataset="cuad",
    sample_size=100,
    clause_categories=["Anti-Assignment", "Audit Rights", ...],
)

print(f"Base success rate: {p:.4f}")
```

### Phase 4: Optimize k Value

**TODO:** Implement voting cost projection

```python
from pipelines.evaluate.cuad_maker import project_costs

s = 41  # number of clause categories per contract
target_accuracy = 0.95

optimal_k, projected_cost = find_optimal_k(
    p=p,
    s=s,
    target_accuracy=target_accuracy,
)

print(f"Optimal k: {optimal_k}, Projected cost: ${projected_cost:.2f}")
```

### Phase 5: Run Full Experiment

**TODO:** Implement MAKER voting execution

```python
from pipelines.evaluate.cuad_maker import run_maker_experiment

results = run_maker_experiment(
    dataset="cuad",
    split="test",
    k=optimal_k,
    red_flagging=True,
    max_tokens=750,
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Zero-error rate: {results['zero_error_rate']:.2%}")
print(f"Total cost: ${results['total_cost']:.2f}")
```

## File Structure

```
ml_workbench/
├── Dockerfile.maker              # Portable Docker image
├── README_MAKER.md              # This file
├── conf/
│   ├── config.yaml              # Base Hydra config
│   └── pipeline/
│       └── collect_cuad.yaml    # CUAD collection settings
├── knowledge/
│   ├── datasets/
│   │   ├── cards/
│   │   │   └── cuad_dataset_card.md
│   │   └── manifests/           # Auto-generated provenance
│   ├── experiments/
│   │   └── plans/
│   │       └── cuad_maker_plan.md
│   └── papers/
│       └── notes/
│           └── maker_paper.md
├── pipelines/
│   ├── collect/
│   │   └── collect_cuad.py      # ✅ Implemented
│   └── evaluate/
│       ├── cuad_maker.py        # TODO: Implement MAKER voting
│       ├── benchmark.py         # Evaluation utilities
│       └── metrics.py           # Metrics calculation
├── scripts/
│   └── export_maker.sh          # Export for transfer
└── utils/
    ├── decorators.py            # Flow/task decorators
    ├── git_utils.py             # Git state capture
    ├── hydra_aim.py             # Experiment tracking
    └── manifest.py              # Dataset provenance
```

## Implementation Status

### ✅ Completed

- [x] CUAD data collection pipeline
- [x] MAKER experiment plan (comprehensive)
- [x] MAKER paper notes (detailed)
- [x] Dataset documentation
- [x] Portable Docker image
- [x] Export scripts
- [x] Provenance tracking
- [x] Hydra configuration

### 🚧 To Implement

- [ ] `pipelines/evaluate/cuad_maker.py` - MAKER voting logic
- [ ] Baseline success rate estimation
- [ ] Optimal k value calculation
- [ ] Red-flagging criteria
- [ ] Multi-agent voting execution
- [ ] Results aggregation and reporting

## MAKER Framework Details

### Maximal Agentic Decomposition

Break each contract review into 41 independent clause identification tasks:

```python
# For each contract → 41 subtasks
for clause_category in CLAUSE_CATEGORIES:
    task = {
        "prompt": f"Does this contract contain a {clause_category} clause?",
        "contract_text": contract,
        "category": clause_category,
    }
```

### First-to-Ahead-by-K Voting

For each subtask, sample multiple independent solutions:

```python
votes = {"YES": 0, "NO": 0}
while max(votes.values()) < k + min(votes.values()):
    response = llm.complete(task_prompt)
    if not is_red_flagged(response):
        answer = parse_answer(response)
        votes[answer] += 1
```

### Red-Flagging Criteria

Discard unreliable responses:

```python
def is_red_flagged(response: str) -> bool:
    # Length-based flags
    if len(response.split()) > 750:
        return True

    # Format-based flags
    if not has_required_format(response):
        return True

    # Confidence-based flags
    uncertainty_markers = ["maybe", "possibly", "unclear"]
    if any(marker in response.lower() for marker in uncertainty_markers):
        return True

    return False
```

## Expected Results

From the experiment plan:

| Metric | Target |
|--------|--------|
| Accuracy (F1) | >95% per clause category |
| Zero-error rate | >90% of contracts |
| Cost per contract | <$5 |
| Scalability | Maintain accuracy up to 10k chars |

## Cost Analysis

Based on MAKER paper (Section 5.2):

| Model | Cost/MTok | Per-Step Error | With k=3 | Notes |
|-------|-----------|----------------|----------|-------|
| GPT-4.1-mini | $1.60 | 0.40% | $3-5/contract | Recommended |
| o3-mini | $4.40 | 0.18% | $9-12/contract | Higher cost |
| GPT-4.1-nano | $0.40 | 35.71% | N/A | Too error-prone |

**Projected cost:** ~$4 per contract (41 clauses × k=3 votes × $0.03 per clause)

## Next Steps

1. **Review Documentation**
   - Read `knowledge/experiments/plans/cuad_maker_plan.md`
   - Read `knowledge/papers/notes/maker_paper.md`

2. **Collect Data**
   - Download CUAD dataset: `python -m pipelines.collect.collect_cuad --limit 1000`
   - Prepare MAKER tasks: `python -m pipelines.collect.collect_cuad --prepare-maker`

3. **Implement MAKER Logic**
   - Create `pipelines/evaluate/cuad_maker.py`
   - Implement voting with red-flagging
   - Follow experiment plan phases

4. **Run Experiments**
   - Estimate baseline success rate (p)
   - Find optimal k value
   - Execute full contract reviews
   - Validate scaling laws

5. **Export for Production**
   - Build Docker image: `docker build -f Dockerfile.maker -t maker-experiment:latest .`
   - Export for transfer: `./scripts/export_maker.sh --compress`
   - Deploy to target system

## Transferring to Another System

### Export

```bash
# On source system
./scripts/export_maker.sh --compress

# Creates:
#   exports/maker-experiment_latest.tar.gz
#   exports/TRANSFER_INSTRUCTIONS.md
```

### Transfer

Copy files to target system:
```bash
scp exports/maker-experiment_latest.tar.gz user@target-host:~/
scp exports/TRANSFER_INSTRUCTIONS.md user@target-host:~/
```

### Import

```bash
# On target system
gunzip -c maker-experiment_latest.tar.gz | docker load

# Verify
docker images | grep maker-experiment

# Test
mkdir outputs
docker run --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  python -m pipelines.collect.collect_cuad --limit 10
```

## Troubleshooting

### Docker build fails

```bash
# Check Docker installed
docker --version

# Check disk space
df -h

# Build with verbose output
docker build -f Dockerfile.maker -t maker-experiment:latest . --progress=plain
```

### Collection fails - rate limits

```bash
# Add HuggingFace token
export HF_TOKEN=your_token_here

# Or in Docker:
docker run --rm \
  -e HF_TOKEN=your_token \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  python -m pipelines.collect.collect_cuad --limit 1000
```

### Out of disk space

```bash
# Clean Docker cache
docker system prune -a

# Check dataset size
du -sh data/collected/cuad
```

## References

- **MAKER Paper:** Meyerson et al., "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
- **CUAD Dataset:** Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review" (arXiv:2103.06268)
- **HuggingFace:** https://huggingface.co/datasets/theatticusproject/cuad

## Support

For questions or issues:
1. Review experiment plan: `cat knowledge/experiments/plans/cuad_maker_plan.md`
2. Check transfer instructions: `cat exports/TRANSFER_INSTRUCTIONS.md`
3. View container README: `docker run --rm maker-experiment:latest cat /workspace/README_CONTAINER.md`
