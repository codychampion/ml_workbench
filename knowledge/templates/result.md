---
type: result
name: "{{result_name}}"
date: {{date}}

# Evaluation Info
experiment: "[[{{experiment}}]]"
model: "[[{{model}}]]"
model_version: "{{version}}"
dataset: "[[{{dataset}}]]"
split: test  # train, val, test

# Code Traceability
git:
  commit: "{{commit_hash}}"
  commit_short: "{{commit_short}}"
  branch: "{{branch}}"
  commit_url: "{{repo_url}}/commit/{{commit_hash}}"
  commit_message: "{{commit_message}}"
  evaluation_script: "{{eval_script_path}}"
  # Trace back to training code
  training_commit: "{{training_commit}}"
  training_commit_url: "{{repo_url}}/commit/{{training_commit}}"

# Metrics
metrics:
  accuracy:
  precision:
  recall:
  f1_score:
  auc_roc:
  loss:
  mAP:
  custom: {}

# Comparison
baseline_model: ""
baseline_metrics: {}
improvement_pct:

# Statistical Significance
num_runs: 1
std_dev: {}
confidence_interval: 95
p_value:

# Compute
evaluation_time_seconds:
samples_per_second:

# Artifacts
confusion_matrix_path: ""
roc_curve_path: ""
predictions_path: ""
report_path: ""

# Validation
great_expectations_suite: ""
validation_passed: true
data_quality_score:

# Connections
compared_with: []
reported_in_paper: []
part_of_project: ""

tags:
  - result
  - {{split}}
---

# Results: {{result_name}}

## Summary
<!-- High-level summary of results -->


## Evaluation Setup

### Model
- Model: [[{{model}}]]
- Version: {{version}}
- Checkpoint: `{{checkpoint_path}}`

### Dataset
- Dataset: [[{{dataset}}]]
- Split: {{split}}
- Samples: {{num_samples}}

### Evaluation Config
```yaml
# Evaluation configuration
```

## Metrics

### Primary Metrics
| Metric | Score | Baseline | Δ |
|--------|-------|----------|---|
| Accuracy | | | |
| F1 Score | | | |
| Precision | | | |
| Recall | | | |

### Per-Class Performance
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
|       |           |        |     |         |

## Visualizations

### Confusion Matrix
<!-- Embed or link to confusion matrix -->


### ROC Curve
<!-- Embed or link to ROC curve -->


### Error Analysis
<!-- Sample misclassifications or errors -->


## Analysis

### What Went Well
-

### Failure Cases
-

### Edge Cases
-

## Comparison with Baselines

| Model | Accuracy | F1 | Notes |
|-------|----------|-----|-------|
| This  |          |     |       |
| Baseline |       |     |       |

## Statistical Analysis
<!-- If multiple runs, report statistics -->


## Conclusions
<!-- What do these results tell us? -->


## Recommendations
<!-- What should be done next based on these results? -->
- [ ]

## Great Expectations Validation
<!-- Data quality validation results -->


## Code Traceability

**Evaluation Code:** [`{{commit_short}}`]({{repo_url}}/commit/{{commit_hash}})
**Training Code:** [`{{training_commit_short}}`]({{repo_url}}/commit/{{training_commit}})

### Reproduce Evaluation
```bash
# Checkout evaluation code
git checkout {{commit_hash}}

# Run evaluation
python -m pipelines.evaluate.evaluate \
  model={{model}} \
  dataset={{dataset}}
```

## Related
- Experiment: [[{{experiment}}]]
- Model: [[{{model}}]]

