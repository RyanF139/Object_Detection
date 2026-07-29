#!/bin/bash

# Cari direktori library nvidia secara bersih dan spesifik
NVIDIA_LIBS=$(python -c "
import site, os
paths = []
for sp in site.getsitepackages():
    if 'site-packages' in sp:
        nv_dir = os.path.join(sp, 'nvidia')
        if os.path.exists(nv_dir):
            # Hanya ambil folder 'lib' tingkat pertama di bawah sub-package (e.g. nvidia/cudnn/lib)
            for sub in os.listdir(nv_dir):
                lib_path = os.path.join(nv_dir, sub, 'lib')
                if os.path.isdir(lib_path):
                    paths.append(lib_path)
print(':'.join(paths))
")

if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
    echo "[LAUNCHER] LD_LIBRARY_PATH diset bersih ke: $LD_LIBRARY_PATH"
else
    echo "[LAUNCHER] Peringatan: Library pip nvidia tidak ditemukan."
fi

# Jalankan aplikasi Python utama
exec python -u object-face-detection.py
