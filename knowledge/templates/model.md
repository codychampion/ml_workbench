---
type: model
name: "{{model_name}}"
version: "{{version}}"
registry_url: "http://localhost:43800/models/{{model_name}}"

# Model Info
architecture: "{{architecture}}"  # resnet50, vit-base, llama-7b, etc.
base_model: "{{base_model}}"      # Pretrained model used
task: "{{task}}"                   # classification, generation, etc.
framework: pytorch  # pytorch, tensorflow, jax

# Code Traceability
git:
  # Commit that trained this model
  training_commit: "{{training_commit}}"
  training_commit_short: "{{training_commit_short}}"
  training_branch: "{{training_branch}}"
  training_commit_url: "{{repo_url}}/commit/{{training_commit}}"
  training_commit_message: "{{training_commit_message}}"
  # Training script and config
  training_script: "{{training_script_path}}"
  training_config: "{{training_config_path}}"
  # Model definition code
  model_definition_file: "{{model_def_path}}"
  model_definition_line: null
  # PR that introduced this model
  pr_number: null
  pr_url: ""
  # Diff showing all changes for this model version
  changes_url: "{{repo_url}}/compare/{{prev_version_commit}}...{{training_commit}}"

# Size & Performance
parameters_millions:
size_mb:
inference_time_ms:
memory_mb:
flops:

# Training
trained_from_experiment: "[[{{experiment}}]]"
training_dataset: "[[{{dataset}}]]"
training_epochs:
final_loss:
final_accuracy:

# Storage
s3_bucket: "mlops-models"
s3_path: "models/{{model_name}}/{{version}}"
local_path: "models/{{model_name}}"
checkpoint_format: pytorch  # pytorch, safetensors, onnx, etc.

# Deployment
stage: development  # development, staging, production, archived
deployed_to: []
inference_endpoint: ""
api_version: ""

# Versioning
created_date: {{date}}
promoted_date:
deprecated_date:
parent_version: ""
child_versions: []

# Connections
based_on_paper: []
evaluated_in: []
used_by_projects: []
fine_tuned_from: ""
fine_tuned_to: []

tags:
  - model
  - {{architecture}}
  - {{stage}}
---

# Model: {{model_name}} v{{version}}

## Overview
<!-- Brief description of the model -->


## Architecture
<!-- Describe the model architecture -->

```
Model Architecture Diagram or Summary
```

### Key Components
-

### Modifications from Base
-

## Training Details

### Dataset
- [[{{dataset}}]]

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | |
| Batch Size | |
| Epochs | |
| Optimizer | |

### Training Curve
<!-- Link to training curves or paste metrics -->


## Performance

### Benchmarks
| Dataset | Metric | Score | Baseline |
|---------|--------|-------|----------|
|         |        |       |          |

### Comparison with Other Versions
| Version | Accuracy | F1 | Notes |
|---------|----------|-----|-------|
|         |          |     |       |

## Usage

### Loading the Model
```python
from utils import load_model

model_path = load_model("{{model_name}}", version="{{version}}")
model = torch.load(model_path)
```

### Inference Example
```python
# Example inference code
```

### ComfyUI Integration
<!-- If applicable, describe ComfyUI workflow -->


## Limitations
-

## Known Issues
-

## Changelog
### v{{version}}
-

## Code Traceability

### Training Code
**Commit:** [`{{training_commit_short}}`]({{repo_url}}/commit/{{training_commit}}) on `{{training_branch}}`
**Message:** {{training_commit_message}}

### Key Files
| File | Purpose |
|------|---------|
| [`{{training_script_path}}`]({{repo_url}}/blob/{{training_commit}}/{{training_script_path}}) | Training script |
| [`{{training_config_path}}`]({{repo_url}}/blob/{{training_commit}}/{{training_config_path}}) | Hydra config |
| [`{{model_def_path}}`]({{repo_url}}/blob/{{training_commit}}/{{model_def_path}}) | Model definition |

### Reproduce Training
```bash
# Checkout exact code state
git checkout {{training_commit}}

# Run training
python -m pipelines.train.train_lora \
  model={{model_name}} \
  +experiment={{experiment_preset}}
```

### Version Diff
[View changes from previous version]({{repo_url}}/compare/{{prev_version_commit}}...{{training_commit}})

## Related
- Paper: [[]]
- Experiment: [[]]
- Results: [[]]

