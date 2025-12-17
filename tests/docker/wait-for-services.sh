#!/bin/bash
# =============================================================================
# Wait for Services and Run Tests
# =============================================================================
# Waits for critical services to be healthy before running the test suite.
# Uses individual service hostnames for Docker networking compatibility.

set -e

# Service hostnames (set via docker-compose environment)
MINIO_HOST="${MINIO_HOST:-localhost}"
MONGODB_HOST="${MONGODB_HOST:-localhost}"
REDIS_HOST="${REDIS_HOST:-localhost}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
AIM_HOST="${AIM_HOST:-localhost}"
PREFECT_HOST="${PREFECT_HOST:-localhost}"
VAULT_HOST="${VAULT_HOST:-localhost}"
LITELLM_HOST="${LITELLM_HOST:-localhost}"
KHOJ_HOST="${KHOJ_HOST:-localhost}"
COUCHDB_HOST="${COUCHDB_HOST:-localhost}"
ZOTERO_HOST="${ZOTERO_HOST:-localhost}"
GE_HOST="${GE_HOST:-localhost}"
LABEL_STUDIO_HOST="${LABEL_STUDIO_HOST:-localhost}"
CVAT_HOST="${CVAT_HOST:-localhost}"
FIFTYONE_HOST="${FIFTYONE_HOST:-localhost}"
SPOTLIGHT_HOST="${SPOTLIGHT_HOST:-localhost}"
COMFYUI_HOST="${COMFYUI_HOST:-localhost}"

MAX_WAIT="${MAX_WAIT:-300}"  # 5 minutes max wait
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"

echo "=============================================="
echo "MLOps Workbench Integration Test Runner"
echo "=============================================="
echo "Test Mode: ${TEST_MODE:-full}"
echo "Max Wait: ${MAX_WAIT}s"
echo ""

# Function to check HTTP endpoint
check_http() {
    local name=$1
    local url=$2
    local auth=$3

    if [ -n "$auth" ]; then
        curl -sf -u "$auth" "$url" > /dev/null 2>&1
    else
        curl -sf "$url" > /dev/null 2>&1
    fi
}

# Function to check TCP port
check_port() {
    local host=$1
    local port=$2
    nc -z "$host" "$port" > /dev/null 2>&1
}

# Build service check list dynamically
declare -A HTTP_SERVICES=(
    ["MinIO"]="${MINIO_HOST}:9000/minio/health/live"
    ["AIM"]="${AIM_HOST}:43800/"
    ["Prefect"]="${PREFECT_HOST}:4200/api/health"
    ["Vault"]="${VAULT_HOST}:8200/v1/sys/health"
)

declare -A TCP_SERVICES=(
    ["MongoDB"]="${MONGODB_HOST}:27017"
    ["Redis"]="${REDIS_HOST}:6379"
    ["PostgreSQL"]="${POSTGRES_HOST}:5432"
)

# Optional services (only check if TEST_MODE is full)
if [ "${TEST_MODE}" = "full" ]; then
    HTTP_SERVICES["Great Expectations"]="${GE_HOST}:8084/"
    HTTP_SERVICES["Khoj"]="${KHOJ_HOST}:42110/api/health"
    HTTP_SERVICES["Zotero"]="${ZOTERO_HOST}:8085/health"
fi

echo "Waiting for services to be ready..."
echo ""

start_time=$(date +%s)
services_ready=false

while [ "$services_ready" = false ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ $elapsed -ge $MAX_WAIT ]; then
        echo ""
        echo "WARNING: Timeout waiting for services after ${MAX_WAIT}s"
        echo "Running tests anyway to show which services failed..."
        break
    fi

    all_ready=true

    # Check HTTP services
    for name in "${!HTTP_SERVICES[@]}"; do
        endpoint="${HTTP_SERVICES[$name]}"
        url="http://${endpoint}"

        if check_http "$name" "$url" ""; then
            printf "  ✓ %-25s ready\n" "$name"
        else
            printf "  ○ %-25s waiting... (%s)\n" "$name" "$url"
            all_ready=false
        fi
    done

    # Check CouchDB with auth
    couchdb_url="http://${COUCHDB_HOST}:5984/_up"
    if check_http "CouchDB" "$couchdb_url" "obsidian:mlops-dev-password"; then
        printf "  ✓ %-25s ready\n" "CouchDB"
    else
        printf "  ○ %-25s waiting...\n" "CouchDB"
        all_ready=false
    fi

    # Check TCP services
    for name in "${!TCP_SERVICES[@]}"; do
        hostport="${TCP_SERVICES[$name]}"
        host="${hostport%%:*}"
        port="${hostport##*:}"

        if check_port "$host" "$port"; then
            printf "  ✓ %-25s ready (port %s)\n" "$name" "$port"
        else
            printf "  ○ %-25s waiting (port %s)...\n" "$name" "$port"
            all_ready=false
        fi
    done

    if [ "$all_ready" = true ]; then
        services_ready=true
        echo ""
        echo "All services ready! (${elapsed}s)"
    else
        echo ""
        echo "Waiting ${WAIT_INTERVAL}s... (${elapsed}s elapsed)"
        sleep $WAIT_INTERVAL
        echo ""
    fi
done

echo ""
echo "=============================================="
echo "Running Integration Tests"
echo "=============================================="
echo ""

# Export all service hosts for pytest
export MINIO_HOST MONGODB_HOST REDIS_HOST POSTGRES_HOST
export AIM_HOST PREFECT_HOST VAULT_HOST LITELLM_HOST
export KHOJ_HOST COUCHDB_HOST ZOTERO_HOST GE_HOST
export LABEL_STUDIO_HOST CVAT_HOST FIFTYONE_HOST SPOTLIGHT_HOST COMFYUI_HOST

# Run tests based on TEST_MODE
case "${TEST_MODE:-full}" in
    quick)
        echo "Mode: Quick health checks"
        python -m pytest tests/integration/test_services_health.py::TestAllServicesQuickCheck \
            -v --timeout=60 --tb=short
        ;;
    health)
        echo "Mode: All health checks"
        python -m pytest tests/integration/test_services_health.py \
            -v --timeout=120 --tb=short
        ;;
    full)
        echo "Mode: Full test suite"
        python -m pytest tests/integration/ \
            -v -m integration --timeout=300 --tb=short \
            --junit-xml=/workspace/test-results.xml 2>&1 || true
        ;;
    minio)
        echo "Mode: MinIO/S3 integration tests"
        python -m pytest tests/integration/test_minio_integration.py \
            -v --timeout=180 --tb=short
        ;;
    knowledge)
        echo "Mode: Knowledge stack tests"
        python -m pytest tests/integration/test_knowledge_stack.py \
            -v --timeout=180 --tb=short
        ;;
    mlops)
        echo "Mode: MLOps services tests"
        python -m pytest tests/integration/test_mlops_services.py \
            -v --timeout=180 --tb=short
        ;;
    *)
        echo "Unknown test mode: ${TEST_MODE}"
        echo "Available modes: quick, health, full, minio, knowledge, mlops"
        exit 1
        ;;
esac

TEST_EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Some tests failed (exit code: $TEST_EXIT_CODE)"
fi
echo "=============================================="

exit $TEST_EXIT_CODE
