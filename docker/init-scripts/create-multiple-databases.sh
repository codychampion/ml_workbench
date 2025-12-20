#!/bin/bash
# =============================================================================
# PostgreSQL Multiple Database Initialization
# =============================================================================
# Creates multiple databases for different services:
#   - khoj: Khoj AI assistant (with pgvector)
#   - litellm: LiteLLM API gateway
#
# This script is automatically run on first PostgreSQL container startup.
# =============================================================================

set -e
set -u

function create_database() {
    local database=$1
    echo "Creating database: $database"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        SELECT 'CREATE DATABASE $database'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL
}

# Create databases from POSTGRES_MULTIPLE_DATABASES environment variable
if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
    echo "Multiple database creation requested: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        create_database $db
    done
    echo "Multiple databases created successfully"
fi

# Create extensions in default database
echo "Creating extensions in default database..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_trgm";
    CREATE EXTENSION IF NOT EXISTS "vector";
EOSQL

# Create vector extension in khoj database (required for embeddings)
echo "Creating vector extension in khoj database..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "khoj" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS "vector";
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EOSQL

echo "Database initialization complete!"
