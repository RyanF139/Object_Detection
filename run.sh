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

# Redirect semua output (stdout & stderr) ke awk untuk difilter
exec > >(awk '
{
    # Tangkap pesan error H264 (tanpa tanda kurung siku awal untuk menghindari gagal match karena kode warna ANSI)
    if ($0 ~ /h264 @ |NULL @ |illegal POC type|error while decoding MB|cabac decode/) {
        lines[count % 1000] = $0
        count++
        if (count % 100 == 0) {
            file = "log_h264.txt"
            printf "" > file
            len = (count > 1000) ? 1000 : count
            start = (count > 1000) ? (count % 1000) : 0
            for (i = 0; i < len; i++) {
                print lines[(start + i) % 1000] >> file
            }
            close(file)
        }
    } else {
        print $0
    }
    fflush()
}') 2>&1


# Jalankan aplikasi Python utama
exec python -u object-face-detection.py

