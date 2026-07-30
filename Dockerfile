# ==========================================
# STAGE 1: Builder (Kompilasi OpenCV CUDA)
# ==========================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    cmake gcc g++ git \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "numpy<2"

WORKDIR /build

RUN git clone --depth 1 -b 4.10.0 https://github.com/opencv/opencv.git && \
    git clone --depth 1 -b 4.10.0 https://github.com/opencv/opencv_contrib.git

RUN git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && make install

WORKDIR /build/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/build/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_CUVID=ON \
    -D WITH_NVCUVID=ON \
    -D CUDA_ARCH_BIN="6.1;7.5;8.0;8.6;8.9" \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

RUN mkdir -p /opencv_lib && \
    cp /usr/local/lib/python3.10/dist-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so /opencv_lib/cv2.so

# ==========================================
# STAGE 2: Runtime
# ==========================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CONTAINER_NAME=object-detection-onnx
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    patchelf \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python || true

# Copy library OpenCV (Python module) dari builder
COPY --from=builder /opencv_lib/cv2.so /usr/local/lib/python3.10/dist-packages/cv2.so
COPY --from=builder /usr/local/lib/libopencv_* /usr/local/lib/
RUN ldconfig

COPY requirements.txt .
RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt \
    && patchelf --clear-execstack /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so || true

COPY . .
RUN mkdir -p image_detection/crop image_detection/frame && chmod +x run.sh

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

CMD ["python", "-u", "object-face-detection.py"]