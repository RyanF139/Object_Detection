# Panduan Revert: Logika Deteksi Kendaraan (ROI & Line)

Jika Anda ingin mengembalikan logika deteksi kendaraan ke kondisi semula — di mana **kendaraan wajib berada di dalam ROI terlebih dahulu sebelum garis penyeberangan (line crossing) dihitung** — ikuti panduan berikut.

---

## 1. `object-face-detection.py`
Buka berkas [object-face-detection.py](file:///c:/Users/TUF/Documents/python/app/object-detection-onnx/object-face-detection.py#L1269-L1285) dan ubah blok kode di dalam fungsi deteksi kendaraan menjadi:

```python
        cx, cy = bbox_center(x1, y1, x2, y2)

        if not self.vehicle_enabled or not self.line_enabled:
            return False
        if not self.is_inside_roi(cx, cy, roi_scaled):
            return False

        track_data = self.tracker.tracks.get(obj_id)
        if track_data is None:
            return False
        track_data.setdefault("last_cross_dir", None)

        direction = check_line_cross(track_data["history"], line_scaled, self.line_in_dir)
        if direction is None or direction == track_data["last_cross_dir"]:
            return False

        track_data["last_cross_dir"] = direction
        print(f"[VEHICLE] ID={obj_id} {cls_name} dir={direction} | cam={self.cid}")
```

---

## 2. `main.py`
Buka berkas [main.py](file:///c:/Users/TUF/Documents/python/app/object-detection-onnx/main.py#L786-L811) dan sesuaikan urutan kondisi `is_inside_roi` sebelum `check_line_cross`:

```python
        if cls_name in VEHICLE_CLASSES:
            if not self.vehicle_enabled:
                return False
            if not self.line_enabled:
                return False
            if not self.is_inside_roi(cx, cy, roi_scaled):
                return False

            track_data = self.tracker.tracks.get(obj_id)
            if track_data is None:
                return False

            direction = check_line_cross(
                track_data["history"], line_scaled, self.line_in_dir
            )
            if direction is None:
                return False

            direction_out = direction
            print(f"[VEHICLE] ID={obj_id} {cls_name} dir={direction_out} | cam={self.cid}")
```

---

## 3. `main2.py`
Buka berkas [main2.py](file:///c:/Users/TUF/Documents/python/app/object-detection-onnx/main2.py#L750-L781) dan kembalikan ke urutan berikut:

```python
            if not self.line_enabled:
                return False

            if not self.is_inside_roi(cx, cy, roi_scaled):
                return False

            track_data = self.tracker.tracks.get(obj_id)
            if track_data is None:
                return False

            track_data.setdefault("last_cross_dir", None)

            direction = check_line_cross(
                track_data["history"], line_scaled, self.line_in_dir
            )

            if direction is None:
                return False

            if direction == track_data["last_cross_dir"]:
                return False

            track_data["last_cross_dir"] = direction
            direction_out = direction
            print(f"[VEHICLE] ID={obj_id} {cls_name} dir={direction_out} | cam={self.cid}")
```

---

## 4. `object-detection-v3.py`
Buka berkas [object-detection-v3.py](file:///c:/Users/TUF/Documents/python/app/object-detection-onnx/object-detection-v3.py#L882-L908) dan kembalikan ke urutan berikut:

```python
        if cls_name in VEHICLE_CLASSES:
            if not self.vehicle_enabled:
                return False
            if not self.line_enabled:
                return False
            if not self.is_inside_roi(cx, cy, roi_scaled):
                return False

            track_data = self.tracker.tracks.get(obj_id)
            if track_data is None:
                return False

            track_data.setdefault("last_cross_dir", None)

            direction = check_line_cross(
                track_data["history"], line_scaled, self.line_in_dir
            )

            if direction is None:
                return False
            if direction == track_data["last_cross_dir"]:
                return False

            track_data["last_cross_dir"] = direction
            direction_out = direction
            print(f"[VEHICLE] ID={obj_id} {cls_name} dir={direction_out} | cam={self.cid}")
```

---

## Perintah Verifikasi Syntax
Setelah melakukan pengembalian kode di atas, jalankan perintah ini di terminal untuk memastikan tidak ada kesalahan sintaksis:
```bash
python -m py_compile object-face-detection.py main.py main2.py object-detection-v3.py
```
