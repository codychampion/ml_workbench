---
type: dataset
name: "{{dataset_name}}"
fiftyone_name: "{{fiftyone_dataset_name}}"
fiftyone_url: "http://localhost:5151/datasets/{{fiftyone_dataset_name}}"

# Source
source: custom  # custom, huggingface, kaggle, s3, url
source_url: "{{url}}"
license: "{{license}}"
citation: "{{citation}}"

# Storage
s3_bucket: "mlops-data"
s3_path: "datasets/{{dataset_name}}"
local_path: "data/{{dataset_name}}"
size_gb:
num_files:

# Data Statistics
total_samples:
train_samples:
val_samples:
test_samples:
num_classes:
class_distribution: {}

# Data Type
modality: image  # image, text, audio, video, tabular, multimodal
task_type: classification  # classification, detection, segmentation, captioning, etc.
format: jpg  # jpg, png, json, csv, parquet, etc.
resolution: "{{width}}x{{height}}"

# Quality
has_labels: true
label_quality: high  # low, medium, high, verified
annotation_tool: ""  # label-studio, cvat, manual, etc.
quality_checks_passed: []

# Preprocessing
preprocessing_applied: []
augmentations: []
normalization:

# Versioning
version: "1.0"
dvc_tracked: false
created_date: {{date}}
last_modified: {{date}}

# Connections
used_in_experiments: []
derived_from: ""
creates_dataset: []
related_papers: []
part_of_project: ""

tags:
  - dataset
  - {{modality}}
  - {{task_type}}
---

# Dataset: {{dataset_name}}

## Overview
<!-- Brief description of the dataset -->


## Sample Images/Data
<!-- Add sample images or data previews -->


## Collection Methodology
<!-- How was the data collected? -->


## Label Schema
<!-- Describe the labeling scheme -->

| Class | Description | Count | % |
|-------|-------------|-------|---|
|       |             |       |   |

## Data Quality

### Quality Checks
- [ ] No corrupted files
- [ ] Labels verified
- [ ] Class balance acceptable
- [ ] No duplicates
- [ ] Train/val/test split is proper

### Known Issues
-

## Usage

### Loading with FiftyOne
```python
import fiftyone as fo
dataset = fo.load_dataset("{{fiftyone_dataset_name}}")
print(dataset)
```

### Loading with Hydra
```yaml
dataset:
  name: {{dataset_name}}
  path: ${paths.data}/{{dataset_name}}
```

## Preprocessing Pipeline
<!-- Describe any preprocessing applied -->
```python
# Preprocessing code
```

## Augmentations Used
<!-- List augmentations and their parameters -->


## Related Datasets
- [[]]

## Experiments Using This Dataset
- [[]]

## Notes
<!-- Additional observations about the data -->

