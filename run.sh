#!/bin/bash

# Ambil direktori site-packages utama (indeks ke-0)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Set LD_LIBRARY_PATH secara eksplisit ke masing-masing folder lib NVIDIA
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH

echo "[LAUNCHER] LD_LIBRARY_PATH configured: $LD_LIBRARY_PATH"

# Jalankan aplikasi Python utama, filter log error H264 ke log_h264.txt (maks 1000 baris)
exec python -u object-face-detection.py 2> >(awk '
{
    if ($0 ~ /\[h264 @ |illegal POC type|error while decoding MB|cabac decode/) {
        print $0 >> "log_h264.txt"
        count++
        if (count >= 100) {
            system("tail -n 1000 log_h264.txt > log_h264_tmp.txt && mv log_h264_tmp.txt log_h264.txt")
            count = 0
        }
    } else {
        print $0 > "/dev/stderr"
    }
    fflush("/dev/stderr")
}')
