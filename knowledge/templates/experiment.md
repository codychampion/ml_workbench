---
type: experiment
name: "{{experiment_name}}"
aim_run_hash: "{{hash}}"
aim_url: "http://localhost:43800/runs/{{hash}}"

# Experiment Info
date: {{date}}
status: running  # planned, running, completed, failed, cancelled
hypothesis: "{{hypothesis}}"

# Code Traceability
git:
  commit: "{{commit_hash}}"
  commit_short: "{{commit_short}}"
  branch: "{{branch}}"
  commit_url: "{{repo_url}}/commit/{{commit_hash}}"
  commit_message: "{{commit_message}}"
  author: "{{commit_author}}"
  dirty: false  # true if uncommitted changes
  diff_from_main: "{{repo_url}}/compare/main...{{commit_hash}}"
  pr_number: null
  pr_url: ""
code_files:
  training_script: "{{script_path}}"
  config_file: "{{config_path}}"
  modified_files: []  # Files changed in this commit

# Configuration
model:
  name: "{{model_name}}"
  version: "{{version}}"
  checkpoint: "{{checkpoint_path}}"
hyperparameters:
  learning_rate:
  batch_size:
  epochs:
  optimizer:
  scheduler:

# Data
dataset: "[[{{dataset_name}}]]"
train_samples:
val_samples:
test_samples:
preprocessing: []

# Compute
device: cpu  # cpu, cuda, mps
gpu_model:
training_time:
memory_peak_gb:

# Results (filled after completion)
final_metrics:
  loss:
  accuracy:
  f1:
best_epoch:
early_stopped: false

# Connections
based_on_paper: []
based_on_experiment: []
produces_model: ""
evaluated_in: []
part_of_project: ""

tags:
  - experiment
  - {{status}}
---

# Experiment: {{experiment_name}}

## Hypothesis
<!-- What are you testing? What do you expect to happen? -->


## Motivation
<!-- Why run this experiment? What question does it answer? -->
- Based on [[]]

## Setup

### Model Configuration
```yaml
# Paste Hydra config or model config here
```

### Training Command
```bash
# Command used to run this experiment
python -m pipelines.train.train_lora \
  model={{model}} \
  train.epochs={{epochs}}
```

## Results

### Metrics Over Time
<!-- Link to AIM charts or paste key metrics -->

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
|       |            |          |         |

### Final Performance
<!-- Summary of final results -->


### Visualizations
<!-- Screenshots or links to generated visualizations -->


## Analysis

### What Worked
-

### What Didn't Work
-

### Surprises
-

## Conclusions
<!-- What did you learn? Does this support/reject the hypothesis? -->


## Next Steps
- [ ]

## Artifacts
- Model checkpoint: `{{checkpoint_path}}`
- Logs: `{{log_path}}`
- Config: `{{config_path}}`

## Code Changes
<!-- Git commit and code state when experiment was run -->

**Commit:** [`{{commit_short}}`]({{repo_url}}/commit/{{commit_hash}}) on branch `{{branch}}`
**Message:** {{commit_message}}
**Author:** {{commit_author}}

### Modified Files
<!-- Files changed in the commit that ran this experiment -->
```
{{modified_files}}
```

### Key Code Snippets
<!-- Important code that produced these results -->
```python
# Paste relevant training code here
```

### Reproducibility
```bash
# Checkout exact code state
git checkout {{commit_hash}}

# Or view diff from main
git diff main...{{commit_hash}}
```

## Related
- [[]]

