#!/bin/bash

# Cari direktori library nvidia di python site-packages secara otomatis
NVIDIA_LIBS=$(python -c "
import site, os
paths = []
for sp in site.getsitepackages():
    if 'site-packages' in sp:
        nv_dir = os.path.join(sp, 'nvidia')
        if os.path.exists(nv_dir):
            for root, dirs, files in os.walk(nv_dir):
                if 'lib' in root:
                    paths.append(root)
print(':'.join(paths))
")

if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
    echo "[LAUNCHER] LD_LIBRARY_PATH berhasil diset ke: $LD_LIBRARY_PATH"
else
    echo "[LAUNCHER] Peringatan: Library pip nvidia tidak ditemukan."
fi

# Jalankan aplikasi Python utama
exec python -u object-face-detection.py
