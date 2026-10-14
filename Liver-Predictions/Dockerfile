# Dockerfile for Referral-Budgeted, Sex-Audited Selective Classification
# Reproducible Research Environment

FROM python:3.11.15-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt requirements-lock.txt ./

# Install exact pinned dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-lock.txt

# Copy complete repository
COPY . .

# Default command: run pytest and paper verification
CMD ["make", "test"]
