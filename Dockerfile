FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CONTAINER_NAME=object-detection-onnx
ENV PATH="/opt/venv/bin:$PATH"

# Install Python, venv, and basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv

COPY requirements.txt .

# Install requirements inside isolated virtual environment
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime onnxruntime-gpu || true \
    && pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.18.0 \
    && patchelf --clear-execstack /opt/venv/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so || true

COPY . .
RUN mkdir -p image_detection/crop image_detection/frame && chmod +x run.sh

# Directly run python using virtual env (PATH is already set to /opt/venv/bin)
CMD ["python", "-u", "object-face-detection.py"]