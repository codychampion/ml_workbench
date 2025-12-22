---
type: dataset
name: "CUAD"
fiftyone_name: "cuad_contracts"
fiftyone_url: "http://localhost:5151/datasets/cuad_contracts"

# Source
source: huggingface
source_url: "https://huggingface.co/datasets/theatticusproject/cuad"
license: "CC-BY-4.0"
citation: "Hendrycks et al., 2021"

# Storage
s3_bucket: "mlops-data"
s3_path: "datasets/cuad"
local_path: "data/collected/cuad"
size_gb: 0.005
num_files: 84325

# Data Statistics
total_samples: 84325
train_samples: 84325
val_samples: 0
test_samples: 0
num_classes: 41
class_distribution: {}

# Data Type
modality: text
task_type: legal-contract-analysis
format: jsonl
resolution: "variable-length-text"

# Quality
has_labels: true
label_quality: high
annotation_tool: "expert-lawyers"
quality_checks_passed: ["expert-annotated", "peer-reviewed"]

# Preprocessing
preprocessing_applied: []
augmentations: []
normalization: null

# Versioning
version: "1.0"
dvc_tracked: false
created_date: 2025-12-22
last_modified: 2025-12-22

# Connections
used_in_experiments: ["[[cuad_maker_plan]]"]
derived_from: ""
creates_dataset: []
related_papers: ["[[maker_paper]]"]
part_of_project: "contract-review-ai"

tags:
  - dataset
  - legal
  - contracts
  - nlp
  - text-classification
  - entity-extraction
---

# Dataset: CUAD (Contract Understanding Atticus Dataset)

## Overview

The Contract Understanding Atticus Dataset (CUAD) v1 is a corpus of more than 13,000 labels across 510 commercial legal contracts designed to support NLP research in legal document analysis. Created by The Atticus Project, it enables automated identification of 41 important clause categories that lawyers typically review during corporate transactions.

**Key Features:**
- 84,325 training samples
- 41 clause categories (33 binary, 8 entity extraction)
- Expert-annotated by lawyers
- Text length: 0-6,970 characters
- CC-BY-4.0 license (commercial use allowed)

## Sample Data

The dataset contains real commercial contracts with annotations for clauses like:
- Anti-Assignment
- Audit Rights
- Cap On Liability
- Non-Compete
- Termination clauses
- IP ownership and licensing
- And 35 more categories...

## Collection Methodology

- **Source:** The Atticus Project
- **Annotation:** Expert lawyers
- **Domain:** Commercial legal contracts
- **Format:** SQuAD 2.0-style JSON + CSV + Excel
- **Quality:** Peer-reviewed, expert-annotated

## Label Schema

### Binary Classification Categories (33)

| Category | Type | Description |
|----------|------|-------------|
| Anti-Assignment | Binary | Restrictions on contract assignment |
| Audit Rights | Binary | Right to audit counterparty |
| Cap On Liability | Binary | Limitation on liability damages |
| Non-Compete | Binary | Non-competition restrictions |
| ... | ... | (29 more binary categories) |

### Entity Extraction Categories (8)

| Category | Type | Format |
|----------|------|--------|
| Document Name | Entity | Text string |
| Effective Date | Entity | mm/dd/yyyy |
| Expiration Date | Entity | mm/dd/yyyy |
| Notice Period | Entity | Duration |
| ... | ... | (4 more entity categories) |

## Data Quality

### Quality Checks
- [x] No corrupted files
- [x] Labels verified by legal experts
- [x] Class balance documented
- [x] No duplicates
- [ ] Train/val/test split (single train split provided)

### Known Issues
- Imbalanced class distribution (some clauses are rare)
- Single domain (commercial contracts only)
- English language only
- US-centric legal terminology

## Usage

### Loading with HuggingFace
```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("theatticusproject/cuad")

# Load specific split
train_data = load_dataset("theatticusproject/cuad", split="train")
```

### Loading with This Repo
```bash
# Download 1000 samples for testing
python -m pipelines.collect.collect_cuad --split train --limit 1000

# Download full dataset
python -m pipelines.collect.collect_cuad --split train --limit -1

# Prepare for MAKER experiment
python -m pipelines.collect.collect_cuad --prepare-maker
```

### Loading with Hydra
```yaml
pipeline: collect_cuad
collect_cuad:
  source:
    dataset: "theatticusproject/cuad"
    split: train
    limit: 1000
```

## Preprocessing Pipeline

No preprocessing applied by default. The text is stored as-is from the original contracts.

For MAKER experiments, the dataset is decomposed into individual clause identification tasks:
```python
# Each contract → 41 clause identification subtasks
# Each subtask = (contract_text, clause_category, label)
```

## Augmentations Used

None. Legal text should not be augmented as it may change legal meaning.

## Related Datasets
- [[ContractNLI]] - Natural language inference for contracts
- [[LEDGAR]] - Legal provision classification
- [[MultiLegalPile]] - Large-scale legal text corpus

## Experiments Using This Dataset
- [[cuad_maker_plan]] - MAKER-based contract review with zero errors

## Related Papers
- [[maker_paper]] - "Solving a Million-Step LLM Task with Zero Errors"
- Original CUAD Paper: Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"

## Notes

### Strengths
- Expert-annotated by lawyers (high quality)
- Covers 41 important clause categories
- Real commercial contracts
- Suitable for both binary classification and entity extraction
- Permissive license (CC-BY-4.0)

### Limitations
- Single domain (commercial contracts)
- No train/val/test splits provided
- Class imbalance (some clauses are rare)
- English only
- May not generalize to other legal systems

### Ideal Use Cases
- Training contract analysis models
- Benchmarking legal NLP systems
- Testing multi-step reasoning systems ([[maker_paper]])
- Clause extraction and classification research

## Citation

```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}
```

## Download Instructions

See [[cuad_maker_plan#Setup]] for complete setup instructions.

**Quick Start:**
```bash
python -m pipelines.collect.collect_cuad --limit 100
```
