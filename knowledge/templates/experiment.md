---
type: experiment
name: "{{experiment_name}}"
aim_run_hash: "{{hash}}"
aim_url: "http://localhost:43800/runs/{{hash}}"

# Experiment Info
date: {{date}}
status: running  # planned, running, completed, failed, cancelled
hypothesis: "{{hypothesis}}"

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

## Related
- [[]]

