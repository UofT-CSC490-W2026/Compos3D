FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Install build tools for pyarrow/pandas if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (optional, but faster) or just use pip
COPY pyproject.toml .
# No lock file for now to keep it simple, or copy uv.lock if available

# Install dependencies
RUN pip install --no-cache-dir \
    "boto3>=1.42.49" \
    "great-expectations>=1.12.3" \
    "pandas>=2.3.3" \
    "pyarrow>=23.0.0" \
    "pydantic>=2.12.5" \
    "pydantic-settings>=2.12.0" \
    "pyyaml>=6.0.3" \
    "rich>=14.3.2" \
    "s3fs>=2026.2.0" \
    "typer>=0.23.1"

# Copy source code
COPY src/ /app/src/
COPY config/ /app/config/

# Install the package in editable mode or just add to pythonpath
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command (can be overridden by Batch)
ENTRYPOINT ["python", "-m", "compos3d_dp.cli"]
CMD ["--help"]
