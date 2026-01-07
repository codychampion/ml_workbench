#!/bin/bash
# =========================================================================
# MAKER Experiment Export Script
# =========================================================================
# Builds a portable Docker image and exports it as a .tar file for
# transfer to another system.
#
# Usage:
#   ./scripts/export_maker.sh
#   ./scripts/export_maker.sh --compress  # Create .tar.gz instead

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

IMAGE_NAME="maker-experiment"
TAG="latest"
OUTPUT_DIR="./exports"
COMPRESS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --compress)
            COMPRESS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}MAKER Experiment Export${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Docker image
echo -e "${YELLOW}[1/3] Building Docker image...${NC}"
docker build -f Dockerfile.maker -t ${IMAGE_NAME}:${TAG} .

# Export image
if [ "$COMPRESS" = true ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${IMAGE_NAME}_${TAG}.tar.gz"
    echo -e "${YELLOW}[2/3] Exporting and compressing image...${NC}"
    docker save ${IMAGE_NAME}:${TAG} | gzip > "$OUTPUT_FILE"
else
    OUTPUT_FILE="${OUTPUT_DIR}/${IMAGE_NAME}_${TAG}.tar"
    echo -e "${YELLOW}[2/3] Exporting image...${NC}"
    docker save ${IMAGE_NAME}:${TAG} -o "$OUTPUT_FILE"
fi

# Get file size
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

# Create transfer instructions
INSTRUCTIONS_FILE="${OUTPUT_DIR}/TRANSFER_INSTRUCTIONS.md"
cat > "$INSTRUCTIONS_FILE" << 'EOF'
# MAKER Experiment - Transfer Instructions

## Files to Transfer

Transfer these files to the target system:

1. `maker-experiment_latest.tar` or `maker-experiment_latest.tar.gz` - Docker image
2. This instructions file (optional)

## On the Target System

### 1. Load Docker Image

```bash
# If .tar file:
docker load -i maker-experiment_latest.tar

# If .tar.gz file:
gunzip -c maker-experiment_latest.tar.gz | docker load
```

### 2. Verify Image Loaded

```bash
docker images | grep maker-experiment
# Should show: maker-experiment    latest    [IMAGE_ID]    [SIZE]
```

### 3. Run Basic Test

```bash
# Create output directory
mkdir -p outputs

# Test run - download 10 samples
docker run --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  python -m pipelines.collect.collect_cuad --limit 10

# Verify output
ls -la outputs/data/collected/cuad/train/
```

### 4. View Documentation

```bash
# View experiment plan
docker run --rm maker-experiment:latest \
  cat knowledge/experiments/plans/cuad_maker_plan.md | less

# View MAKER paper notes
docker run --rm maker-experiment:latest \
  cat knowledge/papers/notes/maker_paper.md | less

# View dataset card
docker run --rm maker-experiment:latest \
  cat knowledge/datasets/cards/cuad_dataset_card.md | less
```

### 5. Interactive Development

```bash
# Start interactive session
docker run --rm -it \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  bash

# Inside container:
python -m pipelines.collect.collect_cuad --limit 100
python -m pipelines.collect.collect_cuad --limit 100 --prepare-maker
cat knowledge/experiments/plans/cuad_maker_plan.md
```

## Common Use Cases

### Download Full CUAD Dataset

```bash
docker run --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  python -m pipelines.collect.collect_cuad --split train --limit -1
```

### Prepare MAKER Experiment

```bash
# Download data and prepare for MAKER voting
docker run --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  python -m pipelines.collect.collect_cuad \
    --limit 1000 \
    --prepare-maker \
    --voting-threshold 3
```

### With GPU Support

```bash
# Requires nvidia-docker2
docker run --gpus all --rm \
  -v $(pwd)/outputs:/workspace/outputs \
  maker-experiment:latest \
  python -m pipelines.collect.collect_cuad --limit 1000
```

### Mount Custom Code

```bash
# For development - mount local code
mkdir -p my_maker_code
docker run --rm -it \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/my_maker_code:/workspace/custom \
  maker-experiment:latest \
  bash
```

## Output Structure

After running collection:

```
outputs/
├── data/
│   └── collected/
│       └── cuad/
│           ├── train/
│           │   ├── cuad_contracts.jsonl       # Downloaded contracts
│           │   ├── metadata.json              # Dataset metadata
│           │   └── collection_manifest.json   # Provenance tracking
│           └── cuad_maker_experiment/
│               ├── maker_tasks.jsonl          # Decomposed tasks for MAKER
│               └── maker_config.json          # MAKER experiment config
└── aim/  # Optional: AIM experiment tracking logs
```

## Troubleshooting

**Issue:** Docker load fails
```bash
# Verify file integrity
sha256sum maker-experiment_latest.tar

# Try extracting manually
tar -xf maker-experiment_latest.tar
```

**Issue:** Permission denied on outputs
```bash
# Fix permissions
sudo chown -R $USER:$USER outputs/
```

**Issue:** Out of disk space
```bash
# Check Docker disk usage
docker system df

# Clean up if needed
docker system prune -a
```

## Next Steps

1. **Review Experiment Plan:** Read `/workspace/knowledge/experiments/plans/cuad_maker_plan.md`
2. **Collect Data:** Download CUAD dataset with desired sample size
3. **Implement MAKER:** Follow the plan to implement MAKER voting logic
4. **Run Experiments:** Execute experiments with different k values

## Support

For questions or issues:
- MAKER Paper: https://arxiv.org/abs/2511.09030
- CUAD Dataset: https://huggingface.co/datasets/theatticusproject/cuad
EOF

echo -e "${YELLOW}[3/3] Creating transfer package...${NC}"
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Export Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "Image exported to: ${YELLOW}$OUTPUT_FILE${NC}"
echo -e "File size: ${YELLOW}$FILE_SIZE${NC}"
echo -e "Instructions: ${YELLOW}$INSTRUCTIONS_FILE${NC}"
echo ""
echo -e "${YELLOW}Transfer these files to the target system:${NC}"
echo -e "  1. $OUTPUT_FILE"
echo -e "  2. $INSTRUCTIONS_FILE"
echo ""
echo -e "${YELLOW}On target system, run:${NC}"
if [ "$COMPRESS" = true ]; then
    echo -e "  gunzip -c maker-experiment_latest.tar.gz | docker load"
else
    echo -e "  docker load -i maker-experiment_latest.tar"
fi
echo ""
