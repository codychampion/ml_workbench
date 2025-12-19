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

## Related
- Paper: [[]]
- Experiment: [[]]
- Results: [[]]

