#!/usr/bin/env docker build . --tag=ucbm-fia-tumorclassify-group-3 --file

# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create input data directory
RUN mkdir -p data

# Create results directory
RUN mkdir -p results

# Set the default command (can be overridden in docker-compose.yml)
CMD ["python", "-m", "src.main"]

