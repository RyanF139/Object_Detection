FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CONTAINER_NAME=object-detection-onnx

# Install Python and basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Map python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python || true

COPY requirements.txt .

# Install requirements (cleaner install using system-level CUDA/cuDNN)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime onnxruntime-gpu numpy || true \
    && rm -rf /usr/local/lib/python3.10/dist-packages/numpy* \
    && rm -rf /usr/lib/python3/dist-packages/numpy* \
    && pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.18.0 \
    && pip install --no-cache-dir --ignore-installed --force-reinstall numpy==1.26.4 \
    && patchelf --clear-execstack /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so || true \
    && patchelf --clear-execstack /usr/lib/python3/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so || true

COPY . .
RUN mkdir -p image_detection/crop image_detection/frame && chmod +x run.sh

# Directly run python (NVIDIA runtime exposes CUDA naturally)
CMD ["python", "-u", "object-face-detection.py"]