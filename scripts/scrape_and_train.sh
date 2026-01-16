#!/bin/bash
# End-to-end pipeline: Scrape Reddit data and train LoRA (all in Docker)
set -e

# Configuration
SUBREDDIT="${1:-fo4}"
CONCEPT="${2:-fallout}"
LIMIT="${3:-100}"
EPOCHS="${4:-5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/scraped/${CONCEPT}_${TIMESTAMP}"

echo "=========================================="
echo "🤖 SCRAPE + TRAIN PIPELINE (Docker)"
echo "=========================================="
echo "Subreddit: r/${SUBREDDIT}"
echo "Concept: ${CONCEPT}"
echo "Limit: ${LIMIT} images"
echo "Epochs: ${EPOCHS}"
echo "=========================================="

# Step 1: Scrape images from Reddit (in Docker)
echo ""
echo "[1/2] 📥 Scraping images from r/${SUBREDDIT}..."
# Use relative path - Docker mounts . to /workspace
docker compose --profile pipeline run --rm -e MSYS_NO_PATHCONV=1 collect \
    python pipelines/collect/collect_reddit.py \
    --subreddit "${SUBREDDIT}" \
    --limit "${LIMIT}" \
    --output "${OUTPUT_DIR}"

# Wait for filesystem sync
sleep 2

# Count images (on host)
if [ -d "${OUTPUT_DIR}" ]; then
    IMAGE_COUNT=$(find "${OUTPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" -o -name "*.webp" \) 2>/dev/null | wc -l || echo "0")
else
    IMAGE_COUNT=0
fi

echo "✓ Collected ${IMAGE_COUNT} images in ${OUTPUT_DIR}"

if [ "${IMAGE_COUNT}" -lt 10 ]; then
    echo "❌ Error: Need at least 10 images for training"
    echo "   Attempted to scrape from: r/${SUBREDDIT}"
    echo "   Output directory: ${OUTPUT_DIR}"
    ls -la "${OUTPUT_DIR}" 2>/dev/null || echo "   Directory doesn't exist yet"
    exit 1
fi

# Step 2: Train LoRA (in Docker) - REAL TRAINING
echo ""
echo "[2/2] 🚀 Training LoRA (REAL diffusion training with Wan 2.2)..."
# Use relative path - Docker mounts . to /workspace
docker compose --profile pipeline run --rm -e MSYS_NO_PATHCONV=1 train \
    python pipelines/train/train_video_lora_real.py \
    --dataset "${OUTPUT_DIR}" \
    --concept "${CONCEPT}" \
    --model "./models/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" \
    --epochs "${EPOCHS}" \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --lora-rank 8 \
    --lora-alpha 16

echo ""
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo "Scraped images: ${OUTPUT_DIR}"
echo "Trained LoRA: ./outputs/lora/${CONCEPT}/"
echo ""

