#!/bin/bash
# Automated ML Workbench Setup Test
# Run this to quickly check if basic setup works
# Results saved to test_results.log

set -e

LOG_FILE="test_results.log"
echo "ML Workbench Setup Test - $(date)" > $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "" >> $LOG_FILE

# Helper functions
log_test() {
    echo "Testing: $1" | tee -a $LOG_FILE
}

log_pass() {
    echo "✓ PASS: $1" | tee -a $LOG_FILE
}

log_fail() {
    echo "✗ FAIL: $1" | tee -a $LOG_FILE
}

log_info() {
    echo "  $1" | tee -a $LOG_FILE
}

# Test 1: Docker
log_test "Docker installation"
if docker --version >> $LOG_FILE 2>&1; then
    log_pass "Docker is installed"
else
    log_fail "Docker not found"
    exit 1
fi

# Test 2: Docker Compose
log_test "Docker Compose"
if docker compose version >> $LOG_FILE 2>&1; then
    log_pass "Docker Compose is installed"
else
    log_fail "Docker Compose not found"
    exit 1
fi

# Test 3: NVIDIA GPU
log_test "NVIDIA GPU"
if nvidia-smi >> $LOG_FILE 2>&1; then
    log_pass "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $LOG_FILE
else
    log_fail "NVIDIA GPU not detected (may not be critical if using CPU)"
fi

# Test 4: NVIDIA Docker Runtime
log_test "NVIDIA Docker Runtime"
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >> $LOG_FILE 2>&1; then
    log_pass "NVIDIA Docker runtime works"
else
    log_fail "NVIDIA Docker runtime not working"
fi

# Test 5: MinIO Start
log_test "Starting MinIO"
if docker compose up -d >> $LOG_FILE 2>&1; then
    sleep 5
    log_pass "MinIO started"
else
    log_fail "Failed to start MinIO"
fi

# Test 6: MinIO Health
log_test "MinIO health check"
if docker compose ps | grep -q "healthy"; then
    log_pass "MinIO is healthy"
else
    log_fail "MinIO not healthy yet (may need more time)"
fi

# Test 7: Build Collect Pipeline
log_test "Building collect pipeline"
if docker compose build collect >> $LOG_FILE 2>&1; then
    log_pass "Collect pipeline built successfully"
else
    log_fail "Failed to build collect pipeline"
fi

# Test 8: Adversarial Pipeline Files
log_test "Adversarial pipeline files"
if [ -f "pipelines/adversarial/generate_adv_patch.py" ]; then
    log_pass "Adversarial pipeline exists"
    log_info "Files: $(ls pipelines/adversarial/*.py | wc -l) Python files"
else
    log_fail "Adversarial pipeline missing"
fi

# Test 9: Required directories
log_test "Required directories"
DIRS=("data" "models" "outputs" "pipelines")
ALL_EXIST=true
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        log_info "✓ $dir/"
    else
        log_info "✗ $dir/ MISSING"
        ALL_EXIST=false
    fi
done
if $ALL_EXIST; then
    log_pass "All required directories exist"
else
    log_fail "Some directories missing"
fi

# Summary
echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Test Summary" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
PASS_COUNT=$(grep -c "✓ PASS" $LOG_FILE || true)
FAIL_COUNT=$(grep -c "✗ FAIL" $LOG_FILE || true)
echo "Passed: $PASS_COUNT" | tee -a $LOG_FILE
echo "Failed: $FAIL_COUNT" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Full results saved to: $LOG_FILE" | tee -a $LOG_FILE

if [ $FAIL_COUNT -eq 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "🎉 All tests passed! Ready to use." | tee -a $LOG_FILE
    exit 0
else
    echo "" | tee -a $LOG_FILE
    echo "⚠️  Some tests failed. Check $LOG_FILE for details." | tee -a $LOG_FILE
    exit 1
fi
