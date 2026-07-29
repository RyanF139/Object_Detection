FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CONTAINER_NAME=object-detection-onnx

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    ffmpeg \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime onnxruntime-gpu || true \
    && pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.19.0 \
    && pip install --no-cache-dir --force-reinstall numpy==1.26.4 \
    && patchelf --clear-execstack /usr/local/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so

COPY . .
RUN mkdir -p image_detection/crop image_detection/frame && chmod +x run.sh

CMD ["./run.sh"]