FROM ghcr.io/jim60105/whisperx:no_model AS base

# Set UID as build argument with default value
ARG UID=1001
USER root

# Create a directory for the wheels
WORKDIR /wheels

# ===== Intermediate build stage for packages =====
FROM python:3.11-slim AS builder

# Download the wheel files
RUN pip install --target=/tmp/wheels fastapi uvicorn python-multipart srt webvtt-py

# ===== Final stage =====
FROM base

# Copy wheels and install
COPY --from=builder /tmp/wheels /wheels
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends unzip && \
    cp -r /wheels/* /venv/lib/python3.11/site-packages/ && \
    apt-get purge -y unzip && apt-get autoremove -y && \
    rm -rf /wheels

# Copy the API code
COPY app.py /app/app.py
RUN chmod 775 /app/app.py

# Expose the API port
EXPOSE 8000

# Set the environment and entrypoint
ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="/venv/lib/python3.11/site-packages"

ENV XDG_CACHE_HOME=/root/.cache
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

WORKDIR /app
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 
