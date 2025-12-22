---
type: model-registry
model_id: "{{model_id}}"
version: "{{version}}"
image_digest: "{{digest}}"
created: "{{date}}"
stage: development
tags: []
---

# Model: {{model_id}}

## Registry Info
- **Image:** localhost:5000/{{model_id}}:{{version}}
- **Digest:** {{digest}}
- **Run:** [[experiments/runs/{{run_id}}]]

## Metrics
| Metric | Value |
|--------|-------|
| | |

## Usage
```bash
docker pull localhost:5000/{{model_id}}:{{version}}
docker run --rm localhost:5000/{{model_id}}:{{version}} --input /data
```

## Changelog
- {{date}}: Initial version
