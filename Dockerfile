# Layout-Optima — Dockerfile
# Uses Python 3.11 to avoid the audioop/pydub removal issue in Python 3.13.
# Runs as non-root user (uid=1000) required by Hugging Face Spaces.

FROM python:3.11-slim

# System deps: ffmpeg needed by pydub (Gradio dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user expected by HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python dependencies as the non-root user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]