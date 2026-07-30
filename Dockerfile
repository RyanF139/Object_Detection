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

# Install numpy FIRST to lock the version and prevent any other package from pulling numpy 2.x
RUN pip install --no-cache-dir "numpy<2"

# Install PyTorch CUDA 11.8 explicitly
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt \
    && patchelf --clear-execstack /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so || true \
    && patchelf --clear-execstack /usr/lib/python3/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so || true

COPY . .
RUN mkdir -p image_detection/crop image_detection/frame && chmod +x run.sh

# Directly run python (NVIDIA runtime exposes CUDA naturally)
CMD ["python", "-u", "object-face-detection.py"]