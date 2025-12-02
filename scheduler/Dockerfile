# ===== Option A: Generic (smaller) Python base (may need to build torch from wheel) =====
FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps needed for some Python packages and for stable builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy scheduler code (your file should be at scheduler/cassia_demo.py)
COPY cassia_demo.py /app/cassia_demo.py

# Ensure non-GUI backend is used by matplotlib by setting MPLCONFIGDIR (optional)
ENV MPLCONFIGDIR=/tmp/matplotlib-cache

# Default working dir and command
WORKDIR /app
CMD ["python", "cassia_demo.py"]

# ===== Notes =====
# - This image requires the Docker socket to be mounted at runtime to control the daemon:
#   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock --network cassia-net cassia-scheduler
# - If you prefer a base image with torch preinstalled (recommended for faster builds), use Option B below.
