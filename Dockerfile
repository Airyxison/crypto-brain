FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# System deps (sqlite3 for ticks.db access)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install before copying code so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Data and checkpoints live outside the image (mount at runtime)
VOLUME ["/data", "/checkpoints"]

# Default: show help. Override with the actual train command at runtime.
CMD ["python", "train.py", "--help"]
