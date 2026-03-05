# Use lightweight Python base image
FROM python:3.11-slim

# Set Python to run in unbuffered mode (better for logging)
ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install Jupyter for notebook support
RUN pip install jupyter notebook ipywidgets

# Create workspace directory
WORKDIR /workspace

# Default command (keeps container running in interactive mode)
CMD ["sleep", "infinity"]
