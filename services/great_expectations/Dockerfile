# Great Expectations Data Validation Service
FROM python:3.11-slim

WORKDIR /workspace

# Install Great Expectations and dependencies
RUN pip install --no-cache-dir \
    great_expectations>=0.18.0 \
    boto3>=1.34.0 \
    s3fs>=2024.2.0 \
    pandas>=2.0.0 \
    pyarrow>=15.0.0 \
    flask>=3.0.0 \
    flask-cors>=4.0.0 \
    pyyaml>=6.0

# Create directories
RUN mkdir -p /workspace/great_expectations

# Copy service files
COPY services/great_expectations/server.py /workspace/server.py
COPY services/great_expectations/init_gx.py /workspace/init_gx.py

EXPOSE 8084

CMD ["python", "/workspace/server.py"]
