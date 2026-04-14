import cv2
import time
import os
import math
import requests
import numpy as np
import onnxruntime as ort
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue, Empty

# ================= PERFORMANCE =================
cv2.setNumThreads(1)
cv2.setUseOptimized(True)

load_dotenv()

# ================= ENV =================

SERVICE_ID   = os.getenv("SERVICE_ID", "capture_100")
MODEL_PATH   = os.getenv("MODEL_PATH", "yolov8s.onnx")

ENDPOINT_URL            = os.getenv("CCTV_ENDPOINT", "http://localhost:8000/api/webhook/cctv/")
CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 10))

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- Detection thresholds ---
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.30))
IOU_THRESHOLD  = float(os.getenv("IOU_THRESHOLD",  0.40))

# --- Save / storage ---
SAVE_FOLDER    = os.getenv("SAVE_FOLDER", "image_detection")
SAVE_INTERVAL  = float(os.getenv("SAVE_INTERVAL", 10))
MAX_IMAGES     = int(os.getenv("MAX_IMAGES", 1000))
CROP_PADDING   = float(os.getenv("CROP_PADDING", 0.20))

# --- Save / send resolution ---
SAVE_MAX_WIDTH = int(os.getenv("SAVE_MAX_WIDTH", 0))

# --- Resize untuk inferensi & display ---
TARGET_MAX_WIDTH = int(os.getenv("RESIZE_WIDTH", 640))
IMG_SIZE         = int(os.getenv("IMG_SIZE", 320))

# --- FPS / idle ---
FRAME_FPS    = int(os.getenv("FRAME_FPS", 12))
IDLE_FPS     = int(os.getenv("IDLE_FPS",  3))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 10))

# --- Tracker ---
TRACK_MAX_DIST = int(os.getenv("TRACK_MAX_DIST", 80))
TRACK_MAX_MISS = int(os.getenv("TRACK_MAX_MISS", 20))
TRACK_MOVE_THR = int(os.getenv("TRACK_MOVE_THR", 40))

# --- Display ---
ENABLE_VIEW    = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH  = int(os.getenv("DISPLAY_WIDTH",  1200))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT",  800))

# --- Debug ---
DEBUG_MODE      = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_VIDEO_DIR = os.getenv("DEBUG_VIDEO_DIR", "./sample_videos")

# ================= OPTIMASI: ONNX THREAD CONFIG =================
# Jumlah thread ONNX untuk shared session
# Rekomendasi: 4-8 untuk server 28 core
# Jangan terlalu besar karena akan bersaing dengan camera threads
ONNX_INTRA_THREADS = int(os.getenv("ONNX_INTRA_THREADS", 4))
ONNX_INTER_THREADS = int(os.getenv("ONNX_INTER_THREADS", 2))

# ================= CLASS CONFIG =================

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

DEFAULT_ROI_POLYGON = np.array([
    [0, 0],
    [640, 0],
    [640, 360],
    [0, 360]
], dtype=np.int32)

WIDTH_LINE = 1

DEFAULT_LINE            = [[0, 320], [640, 320]]
DEFAULT_LINE_IN_DIR     = "A"
DEFAULT_LINE_ENABLED    = True
DEFAULT_VEHICLE_ENABLED = os.getenv("VEHICLE_ENABLED", "true").lower() == "true"

CLASS_CONFIG = {
    "person": {
        "conf":     float(os.getenv("PERSON_CONF", 0.70)),
        "color":    tuple(map(int, os.getenv("PERSON_COLOR", "0,255,0").split(","))),
        "min_size": int(os.getenv("PERSON_MIN_SIZE", 3000)),
        "max_size": int(os.getenv("PERSON_MAX_SIZE", 110000)),
        "cooldown": int(os.getenv("PERSON_COOLDOWN", 10)),
    },
    "car": {
        "conf":     float(os.getenv("CAR_CONF", 0.40)),
        "color":    tuple(map(int, os.getenv("CAR_COLOR", "255,0,0").split(","))),
        "min_size": int(os.getenv("CAR_MIN_SIZE", 5000)),
        "max_size": int(os.getenv("CAR_MAX_SIZE", 200000)),
        "cooldown": int(os.getenv("CAR_COOLDOWN", 10)),
    },
    "bus": {
        "conf":     float(os.getenv("BUS_CONF", 0.40)),
        "color":    tuple(map(int, os.getenv("BUS_COLOR", "255,165,0").split(","))),
        "min_size": int(os.getenv("BUS_MIN_SIZE", 50000)),
        "max_size": int(os.getenv("BUS_MAX_SIZE", 200000)),
        "cooldown": int(os.getenv("BUS_COOLDOWN", 10)),
    },
    "truck": {
        "conf":     float(os.getenv("TRUCK_CONF", 0.40)),
        "color":    tuple(map(int, os.getenv("TRUCK_COLOR", "128,128,128").split(","))),
        "min_size": int(os.getenv("TRUCK_MIN_SIZE", 100000)),
        "max_size": int(os.getenv("TRUCK_MAX_SIZE", 300000)),
        "cooldown": int(os.getenv("TRUCK_COOLDOWN", 10)),
    },
    "motorcycle": {
        "conf":     float(os.getenv("MOTORCYCLE_CONF", 0.35)),
        "color":    tuple(map(int, os.getenv("MOTORCYCLE_COLOR", "0,255,255").split(","))),
        "min_size": int(os.getenv("MOTORCYCLE_MIN_SIZE", 1000)),
        "max_size": int(os.getenv("MOTORCYCLE_MAX_SIZE", 90000)),
        "cooldown": int(os.getenv("MOTORCYCLE_COOLDOWN", 10)),
    },
}

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
ALLOWED_CLASSES = set(CLASS_CONFIG.keys())

# ================= FOLDERS =================

DETECT_FOLDER = os.path.join(SAVE_FOLDER, "crop")
FRAME_FOLDER  = os.path.join(SAVE_FOLDER, "frame")

os.makedirs(DETECT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER,  exist_ok=True)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ================= GLOBALS =================

JAKARTA_TZ     = timezone(timedelta(hours=7))
active_cameras = {}
camera_lock    = Lock()

preview_frames = {}
preview_lock   = Lock()

webhook_queue  = Queue(maxsize=1000)

# ================= OPTIMASI: SHARED ONNX SESSION =================
# Satu session dipakai semua kamera — eliminasi thread explosion
# ONNX Runtime thread-safe untuk concurrent inference

_shared_session: ort.InferenceSession | None = None
_shared_input_name: str | None = None
_session_lock = Lock()

def build_shared_session() -> tuple[ort.InferenceSession, str]:
    """
    Buat shared ONNX session dengan thread terbatas.
    Dipanggil sekali saat startup, dipakai semua CameraWorker.

    Kenapa ini efektif:
    - Default ONNX: spawn thread = jumlah CPU core, PER SESSION
    - 3 kamera × 28 thread = 84 thread ONNX bersaing → 100% CPU
    - Shared session: cukup 1 set thread untuk semua kamera
    - intra_op=4 artinya max 4 thread untuk operasi dalam satu layer
    - inter_op=2 artinya max 2 thread untuk operasi antar layer
    - Total thread ONNX: 6 thread saja, vs 84 thread sebelumnya
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ONNX_INTRA_THREADS  # default 28 → ubah ke 4
    opts.inter_op_num_threads = ONNX_INTER_THREADS  # default 28 → ubah ke 2
    opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(MODEL_PATH, providers=providers, sess_options=opts)
    except Exception:
        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=opts)

    input_name = sess.get_inputs()[0].name

    # Warmup
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    sess.run(None, {input_name: dummy})

    print(f"[ONNX SHARED] Session created | provider={sess.get_providers()}")
    print(f"[ONNX SHARED] intra_threads={ONNX_INTRA_THREADS} | inter_threads={ONNX_INTER_THREADS}")
    return sess, input_name

def get_shared_session() -> tuple[ort.InferenceSession, str]:
    """Ambil shared session, buat jika belum ada."""
    global _shared_session, _shared_input_name
    with _session_lock:
        if _shared_session is None:
            _shared_session, _shared_input_name = build_shared_session()
    return _shared_session, _shared_input_name


# Inisialisasi shared session saat startup
# Dilakukan di sini agar semua kamera langsung bisa pakai
print("[STARTUP] Initializing shared ONNX session...")
get_shared_session()
print("[STARTUP] Shared ONNX session ready")


# ================= OPTIMASI: CENTRALIZED INFERENCE QUEUE =================
# Semua kamera kirim frame ke antrian terpusat
# 1 inference worker yang menjalankan ONNX — tidak ada race condition
# Camera worker tidak blocking, langsung lanjut ambil frame berikutnya

_infer_input_queue: Queue  = Queue(maxsize=50)   # (camera_id, frame, result_queue)
_INFER_WORKERS            = int(os.getenv("INFER_WORKERS", 2))  # jumlah thread inferensi


def inference_worker(worker_id: int):
    """
    Worker terpusat untuk inferensi ONNX.
    Menerima frame dari semua kamera, jalankan inferensi, kembalikan hasil.

    Keuntungan vs per-kamera:
    - Tidak ada context switching antar ONNX sessions
    - Satu set thread ONNX, utilisasi lebih efisien
    - Camera worker tidak pernah block di inferensi
    """
    sess, input_name = get_shared_session()
    print(f"[INFER WORKER {worker_id}] Started")

    while True:
        try:
            item = _infer_input_queue.get(timeout=1.0)
        except Empty:
            continue

        if item is None:
            break

        cid, frame_infer, result_q = item

        try:
            inp, scale = preprocess(frame_infer)
            outputs    = sess.run(None, {input_name: inp})
            h, w       = frame_infer.shape[:2]
            results    = postprocess(outputs, scale, w, h)
            result_q.put(("ok", results))
        except Exception as e:
            result_q.put(("err", []))
            print(f"[INFER WORKER {worker_id}] Error cam={cid}: {e}")
        finally:
            _infer_input_queue.task_done()


# Start inference workers
for _i in range(_INFER_WORKERS):
    Thread(target=inference_worker, args=(_i,), daemon=True).start()

print(f"[STARTUP] {_INFER_WORKERS} inference worker(s) started")


# ================= UTILS =================

def point_side_of_line(px, py, x1, y1, x2, y2):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def check_line_cross(history, line_pts, line_in_dir):
    if len(history) < 2:
        return None
    (lx1, ly1), (lx2, ly2) = line_pts
    sides = [
        point_side_of_line(px, py, lx1, ly1, lx2, ly2)
        for px, py in history
    ]
    last_direction = None
    for i in range(1, len(sides)):
        prev = sides[i - 1]
        curr = sides[i]
        if prev == 0 or curr == 0:
            continue
        if (prev > 0 and curr > 0) or (prev < 0 and curr < 0):
            continue
        crossed_a_to_b = (prev > 0 and curr < 0)
        if line_in_dir == "A":
            last_direction = "IN" if crossed_a_to_b else "OUT"
        else:
            last_direction = "IN" if not crossed_a_to_b else "OUT"
    return last_direction

def scale_roi(roi_ref, inf_w, inf_h):
    ref_h = int(TARGET_MAX_WIDTH * inf_h / inf_w) if inf_w > 0 else inf_h
    sx = inf_w / TARGET_MAX_WIDTH
    sy = inf_h / ref_h
    return (roi_ref.astype(np.float32) * [sx, sy]).astype(np.int32)

def scale_line_pts(line_pts_ref, inf_w, inf_h):
    ref_h = int(TARGET_MAX_WIDTH * inf_h / inf_w) if inf_w > 0 else inf_h
    sx = inf_w / TARGET_MAX_WIDTH
    sy = inf_h / ref_h
    (rx1, ry1), (rx2, ry2) = line_pts_ref
    return (
        (int(rx1 * sx), int(ry1 * sy)),
        (int(rx2 * sx), int(ry2 * sy)),
    )

def iso_timestamp():
    return datetime.now(JAKARTA_TZ).isoformat(timespec="seconds")

def iso_name(prefix, cls_name, ts_iso, obj_id, direction=None):
    safe_ts  = ts_iso.replace(":", "-")
    dir_part = f"_{direction}" if direction else ""
    return f"{prefix}_{cls_name}_id{obj_id}{dir_part}_{safe_ts}.jpg"

def enforce_limit(folder):
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )
    while len(files) > MAX_IMAGES:
        try:
            os.remove(files.pop(0))
        except Exception:
            pass

def expand_crop_bbox(x1, y1, x2, y2, img_w, img_h):
    w     = x2 - x1
    h     = y2 - y1
    pad_w = int(w * CROP_PADDING)
    pad_h = int(h * CROP_PADDING)
    x1    = max(0,     x1 - pad_w)
    y1    = max(0,     y1 - pad_h)
    x2    = min(img_w, x2 + pad_w)
    y2    = min(img_h, y2 + pad_h)
    return x1, y1, x2, y2

def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def center_dist(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

def parse_roi(raw_roi):
    if raw_roi is None:
        return DEFAULT_ROI_POLYGON.copy()
    if isinstance(raw_roi, np.ndarray):
        return raw_roi.astype(np.int32)
    try:
        arr = np.array(raw_roi, dtype=np.int32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    except Exception:
        pass
    print(f"[ROI PARSE ERROR] Invalid roi: {raw_roi!r}. Using default.")
    return DEFAULT_ROI_POLYGON.copy()

def parse_line(raw_line):
    if raw_line is not None:
        try:
            arr = np.array(raw_line, dtype=np.int32)
            if arr.shape == (2, 2):
                return ((int(arr[0][0]), int(arr[0][1])),
                        (int(arr[1][0]), int(arr[1][1])))
        except Exception:
            pass
        print(f"[LINE PARSE ERROR] Invalid line: {raw_line!r}. Using default.")
    pts = DEFAULT_LINE
    return ((int(pts[0][0]), int(pts[0][1])),
            (int(pts[1][0]), int(pts[1][1])))

def parse_line_in_dir(raw_dir):
    if raw_dir in ("A", "B"):
        return raw_dir
    if raw_dir is not None:
        print(f"[LINE DIR ERROR] Invalid line_in_dir: {raw_dir!r}. Using default '{DEFAULT_LINE_IN_DIR}'.")
    return DEFAULT_LINE_IN_DIR

def prepare_for_save(img: np.ndarray) -> np.ndarray:
    if SAVE_MAX_WIDTH <= 0:
        return img
    h, w = img.shape[:2]
    if w <= SAVE_MAX_WIDTH:
        return img
    scale = SAVE_MAX_WIDTH / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ================= ONNX PRE/POSTPROCESS =================

def preprocess(img: np.ndarray):
    h, w   = img.shape[:2]
    scale  = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    inp = canvas.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    inp = np.expand_dims(inp, axis=0)
    return inp, scale

def postprocess(outputs, scale: float, orig_w: int, orig_h: int):
    out = outputs[0]
    if out.shape[1] < out.shape[2]:
        out = np.transpose(out, (0, 2, 1))
    out = out[0]

    per_class: dict[str, dict] = {}

    for row in out:
        cls_scores = row[4:]
        cls_id     = int(np.argmax(cls_scores))
        conf       = float(cls_scores[cls_id])
        if cls_id >= len(COCO_CLASSES):
            continue
        class_name = COCO_CLASSES[cls_id]
        if class_name not in CLASS_CONFIG:
            continue
        cfg = CLASS_CONFIG[class_name]
        if conf < cfg["conf"]:
            continue

        cx, cy, bw, bh = row[:4]
        x1 = max(0, min(int((cx - bw / 2) / scale), orig_w))
        y1 = max(0, min(int((cy - bh / 2) / scale), orig_h))
        x2 = max(0, min(int((cx + bw / 2) / scale), orig_w))
        y2 = max(0, min(int((cy + bh / 2) / scale), orig_h))
        area = (x2 - x1) * (y2 - y1)

        if class_name not in VEHICLE_CLASSES:
            if cfg.get("min_size") and area < cfg["min_size"]:
                continue
            if cfg.get("max_size") and area > cfg["max_size"]:
                continue

        if class_name not in per_class:
            per_class[class_name] = {"boxes": [], "scores": [], "cls_ids": []}
        per_class[class_name]["boxes"].append([x1, y1, x2 - x1, y2 - y1])
        per_class[class_name]["scores"].append(conf)
        per_class[class_name]["cls_ids"].append(cls_id)

    results = []
    for class_name, data in per_class.items():
        cfg     = CLASS_CONFIG[class_name]
        indices = cv2.dnn.NMSBoxes(
            data["boxes"],
            data["scores"],
            cfg["conf"],
            IOU_THRESHOLD,
        )
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = data["boxes"][i]
                results.append((
                    x, y, x + w, y + h,
                    data["scores"][i],
                    COCO_CLASSES[data["cls_ids"][i]],
                ))
    return results


# ================= SIMPLE TRACKER =================

class ObjectTracker:

    _global_id = 0
    _id_lock   = Lock()

    @classmethod
    def _next_id(cls):
        with cls._id_lock:
            cls._global_id += 1
            return cls._global_id

    def __init__(self):
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list) -> list:
        now = time.time()
        for t in self.tracks.values():
            t["matched"] = False

        enriched = []

        for (x1, y1, x2, y2, conf, cls_name) in detections:
            cx, cy = bbox_center(x1, y1, x2, y2)

            best_id   = None
            best_dist = float("inf")
            for tid, t in self.tracks.items():
                if t["cls_name"] != cls_name:
                    continue
                d = center_dist((cx, cy), (t["cx"], t["cy"]))
                if d < best_dist and d < TRACK_MAX_DIST:
                    best_dist = d
                    best_id   = tid

            cfg = CLASS_CONFIG[cls_name]

            if best_id is not None:
                t = self.tracks[best_id]
                t["matched"] = True

                moved       = center_dist((cx, cy), (t["cx"], t["cy"])) > TRACK_MOVE_THR
                cooldown_ok = (now - t["last_sent"]) >= cfg["cooldown"]
                should_send = (not t["sent_once"]) or (cooldown_ok and moved)

                t["cx"], t["cy"] = cx, cy
                t["history"].append((cx, cy))
                if len(t["history"]) > 30:
                    t["history"].pop(0)
                t["x1"], t["y1"], t["x2"], t["y2"] = x1, y1, x2, y2
                t["miss"] = 0

                if should_send:
                    t["last_sent"] = now
                    t["sent_once"] = True

                enriched.append({
                    "bbox":        (x1, y1, x2, y2),
                    "conf":        conf,
                    "cls_name":    cls_name,
                    "obj_id":      best_id,
                    "should_send": should_send,
                    "is_new":      False,
                })
            else:
                new_id = self._next_id()
                self.tracks[new_id] = {
                    "history":        [(cx, cy)],
                    "id":             new_id,
                    "cls_name":       cls_name,
                    "cx": cx, "cy":   cy,
                    "x1": x1, "y1":   y1, "x2": x2, "y2": y2,
                    "miss":           0,
                    "last_sent":      now,
                    "sent_once":      True,
                    "matched":        True,
                    "last_cross_dir": None,
                }
                enriched.append({
                    "bbox":        (x1, y1, x2, y2),
                    "conf":        conf,
                    "cls_name":    cls_name,
                    "obj_id":      new_id,
                    "should_send": True,
                    "is_new":      True,
                })

        dead_ids = []
        for tid, t in self.tracks.items():
            if not t["matched"]:
                t["miss"] += 1
                if t["miss"] >= TRACK_MAX_MISS:
                    dead_ids.append(tid)
        for tid in dead_ids:
            del self.tracks[tid]
            if DEBUG_MODE:
                print(f"[TRACKER] ID {tid} expired")

        return enriched

    def active_count(self) -> dict:
        count = {}
        for t in self.tracks.values():
            count[t["cls_name"]] = count.get(t["cls_name"], 0) + 1
        return count


# ================= WEBHOOK WORKER =================

def webhook_worker():
    while True:
        item = webhook_queue.get()
        if item is None:
            break

        (
            crop_bytes, frame_clean_bytes,
            crop_name, frame_name,
            name_camera, ts_iso,
            bbox, conf, cls_name,
            cid, client_id,
            obj_id, is_new,
            direction,
        ) = item

        payload = {
            "timestamp":   ts_iso,
            "type":        "object_detection_service",
            "class_name":  cls_name,
            "bbox":        f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "confidence":  round(conf, 4),
            "channel_id":  cid,
            "client_id":   client_id,
            "direction":   direction,
        }

        log_payload = {
            "url":         WEBHOOK_URL,
            "frame_file":  frame_name,
            "channel_id":  cid,
            "camera_name": name_camera,
            "data":        payload,
        }
        print(f"\n[WEBHOOK] Sending:\n{json.dumps(log_payload, indent=4)}")

        sent = False
        for attempt in range(2):
            try:
                if cls_name != "person":
                    resp = requests.post(
                        WEBHOOK_URL,
                        files=[
                            ("files", (frame_name, frame_clean_bytes, "image/jpeg")),
                            ("files", (crop_name,  crop_bytes,        "image/jpeg")),
                        ],
                        data=payload,
                        timeout=10,
                    )
                else:
                    resp = requests.post(
                        WEBHOOK_URL,
                        files=[("files", (frame_name, frame_clean_bytes, "image/jpeg"))],
                        data=payload,
                        timeout=10,
                    )
                resp.raise_for_status()
                print(
                    f"[WEBHOOK OK] {frame_name} "
                    f"| id={obj_id} | class={cls_name} | conf={conf:.3f} "
                    f"| new={is_new} | http={resp.status_code}"
                )
                sent = True
                break
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                print(f"[WEBHOOK FAIL HTTP {status}] {frame_name} | attempt={attempt + 1} | {e}")
                if attempt == 0:
                    time.sleep(2)
            except requests.exceptions.ConnectionError:
                print(f"[WEBHOOK FAIL CONNECTION] {frame_name} | attempt={attempt + 1}")
                if attempt == 0:
                    time.sleep(2)
            except requests.exceptions.Timeout:
                print(f"[WEBHOOK FAIL TIMEOUT] {frame_name} | attempt={attempt + 1}")
                if attempt == 0:
                    time.sleep(1)
            except Exception as e:
                print(f"[WEBHOOK FAIL UNEXPECTED] {frame_name} | {e}")
                break

        if not sent:
            print(f"[WEBHOOK GIVE UP] {frame_name} | id={obj_id} | semua attempt gagal")

        webhook_queue.task_done()


if WEBHOOK_URL:
    Thread(target=webhook_worker, daemon=True).start()


# ================= CAMERA WORKER =================

class CameraWorker:

    def __init__(self, cid, client_id, url, name,
                 roi=None, line=None, line_in_dir=None,
                 line_enabled=None, vehicle_enabled=None):

        self.cid           = cid
        self.client_id     = client_id
        self.name_camera   = name
        self.stream_source = url if DEBUG_MODE else "rtsp://" + url
        self.running       = True

        self.vehicle_enabled = (
            bool(vehicle_enabled)
            if vehicle_enabled is not None
            else DEFAULT_VEHICLE_ENABLED
        )

        self.roi_polygon = parse_roi(roi)
        self.line_pts    = parse_line(line)
        self.line_in_dir = parse_line_in_dir(line_in_dir)

        raw_line_enabled  = (
            bool(line_enabled)
            if line_enabled is not None
            else DEFAULT_LINE_ENABLED
        )
        self.line_enabled = raw_line_enabled if self.vehicle_enabled else False

        self._log_config()

        self.connected        = False
        self.reconnecting     = False
        self.dead             = False

        self.last_frame_time  = 0
        self.bad              = 0
        self.max_bad          = 10
        self.last_time        = 0
        self.last_detect_time = time.time()

        self.tracker = ObjectTracker()

        # OPTIMASI: result queue per kamera untuk terima hasil dari inference worker
        # Camera worker submit frame → inference worker proses → hasil balik lewat queue ini
        self._result_q: Queue = Queue(maxsize=2)

        self.cap = cv2.VideoCapture(self.stream_source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.connected = True
            print(f"[RTSP CONNECTED] {cid} -> {name}")
        else:
            self.dead = True
            print(f"[RTSP FAILED] {cid} -> {name}")

        # OPTIMASI: tidak perlu buat session sendiri
        # Cukup ambil referensi shared session (sudah diinit di startup)
        # self.session = build_session(MODEL_PATH)  ← DIHAPUS
        # self.input_name = ...                      ← DIHAPUS

        print(f"[CAMERA START] {cid} -> {name} | using shared ONNX session")

    def _log_config(self):
        print(
            f"[CAMERA CONFIG] {self.cid} "
            f"| vehicle_enabled={self.vehicle_enabled} "
            f"| line_enabled={self.line_enabled} "
            f"| line={self.line_pts} "
            f"| line_in_dir={self.line_in_dir} "
            f"| roi={self.roi_polygon.tolist()}"
        )

    def stop(self):
        self.running = False
        self.cap.release()
        print(f"[CAMERA STOP] {self.cid}")

    def update_config(self, roi=None, line=None, line_in_dir=None,
                      line_enabled=None, vehicle_enabled=None):
        changed = False

        new_vehicle = (
            bool(vehicle_enabled)
            if vehicle_enabled is not None
            else DEFAULT_VEHICLE_ENABLED
        )
        if new_vehicle != self.vehicle_enabled:
            self.vehicle_enabled = new_vehicle
            changed = True

        new_roi = parse_roi(roi)
        if not np.array_equal(new_roi, self.roi_polygon):
            self.roi_polygon = new_roi
            changed = True

        new_line = parse_line(line)
        if new_line != self.line_pts:
            self.line_pts = new_line
            changed = True

        new_dir = parse_line_in_dir(line_in_dir)
        if new_dir != self.line_in_dir:
            self.line_in_dir = new_dir
            changed = True

        raw_line_enabled = (
            bool(line_enabled)
            if line_enabled is not None
            else DEFAULT_LINE_ENABLED
        )
        effective_line = raw_line_enabled if self.vehicle_enabled else False
        if effective_line != self.line_enabled:
            self.line_enabled = effective_line
            changed = True

        if changed:
            self._log_config()

    def is_inside_roi(self, cx, cy, roi_scaled) -> bool:
        return cv2.pointPolygonTest(roi_scaled, (float(cx), float(cy)), False) >= 0

    def resize_frame(self, frame):
        h, w  = frame.shape[:2]
        scale = min(1.0, TARGET_MAX_WIDTH / w)
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return frame

    def adaptive_fps(self):
        if time.time() - self.last_detect_time > IDLE_TIMEOUT:
            return 1.0 / IDLE_FPS
        return 1.0 / FRAME_FPS

    def _submit_for_inference(self, frame_infer):
        """
        Submit frame ke inference queue terpusat (non-blocking).
        Jika queue penuh (camera terlalu cepat), frame di-drop — tidak apa-apa
        karena frame berikutnya segera datang.
        """
        # Bersihkan result queue lama supaya tidak stale
        while not self._result_q.empty():
            try:
                self._result_q.get_nowait()
            except Empty:
                break

        try:
            _infer_input_queue.put_nowait((self.cid, frame_infer, self._result_q))
            return True
        except Exception:
            # Inference queue penuh → skip frame ini
            return False

    def _get_inference_result(self, timeout=0.5):
        """
        Tunggu hasil inferensi dari worker (blocking dengan timeout).
        Return list deteksi atau [] jika timeout/error.
        """
        try:
            status, result = self._result_q.get(timeout=timeout)
            if status == "ok":
                return result
        except Empty:
            pass
        return []

    def _process_object(self, obj, frame_original, view,
                        orig_w, orig_h, scale_x, scale_y,
                        bbox_orig, roi_scaled, line_scaled):

        x1, y1, x2, y2     = obj["bbox"]
        ox1, oy1, ox2, oy2 = bbox_orig
        conf                = obj["conf"]
        cls_name            = obj["cls_name"]
        obj_id              = obj["obj_id"]
        should_send         = obj["should_send"]
        is_new              = obj["is_new"]

        cfg   = CLASS_CONFIG[cls_name]
        color = cfg["color"]

        ts_iso = iso_timestamp()

        area   = (ox2 - ox1) * (oy2 - oy1)
        width  = ox2 - ox1
        height = oy2 - oy1

        cv2.rectangle(view, (x1, y1), (x2, y2), color, WIDTH_LINE)
        cv2.putText(
            view,
            f"#{obj_id} {cls_name} {conf:.2f} {area} {width} {height}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), WIDTH_LINE,
        )

        if DEBUG_MODE:
            tag = "NEW" if is_new else ("SEND" if should_send else "SKIP")
            print(f"[{self.cid}] ID={obj_id} {cls_name} conf={conf:.2f} -> {tag}")

        cx, cy        = bbox_center(x1, y1, x2, y2)
        direction_out = None

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

        else:
            if not should_send:
                return False

        cx1, cy1, cx2, cy2 = expand_crop_bbox(ox1, oy1, ox2, oy2, orig_w, orig_h)
        crop_original      = frame_original[cy1:cy2, cx1:cx2]

        crop_save  = prepare_for_save(crop_original)
        frame_save = prepare_for_save(frame_original)

        _, crop_jpg       = cv2.imencode(".jpg", crop_save)
        crop_bytes        = crop_jpg.tobytes()

        _, frame_jpg      = cv2.imencode(".jpg", frame_save)
        frame_clean_bytes = frame_jpg.tobytes()

        crop_name  = iso_name("crop",  cls_name, ts_iso, obj_id, direction_out)
        frame_name = iso_name("frame", cls_name, ts_iso, obj_id, direction_out)

        try:
            with open(os.path.join(DETECT_FOLDER, crop_name), "wb") as f:
                f.write(crop_bytes)
            with open(os.path.join(FRAME_FOLDER, frame_name), "wb") as f:
                f.write(frame_clean_bytes)
            enforce_limit(DETECT_FOLDER)
            enforce_limit(FRAME_FOLDER)
        except Exception as e:
            print(f"[SAVE ERROR] {e}")

        if WEBHOOK_URL:
            try:
                webhook_queue.put_nowait((
                    crop_bytes,
                    frame_clean_bytes,
                    crop_name,
                    frame_name,
                    self.name_camera,
                    ts_iso,
                    (ox1, oy1, ox2, oy2),
                    conf, cls_name,
                    self.cid, self.client_id,
                    obj_id, is_new,
                    direction_out,
                ))
            except Exception:
                print("[QUEUE FULL] webhook queue penuh")

        return True

    def run(self):
        while self.running:

            interval = self.adaptive_fps()
            if time.time() - self.last_time < interval:
                time.sleep(0.005)
                continue
            self.last_time = time.time()

            for _ in range(2):
                self.cap.grab()

            ret, frame = self.cap.read()

            if ret:
                self.connected       = True
                self.dead            = False
                self.last_frame_time = time.time()
                self.bad             = 0
            else:
                self.connected = False
                self.bad      += 1
                if self.bad >= self.max_bad:
                    self._reconnect()
                continue

            frame_original = frame
            frame_infer    = self.resize_frame(frame)
            view           = frame_infer.copy()

            orig_h, orig_w = frame_original.shape[:2]
            inf_h,  inf_w  = frame_infer.shape[:2]

            scale_x = orig_w / inf_w
            scale_y = orig_h / inf_h

            roi_scaled  = scale_roi(self.roi_polygon, inf_w, inf_h)
            line_scaled = scale_line_pts(self.line_pts, inf_w, inf_h)

            # ── OPTIMASI: submit ke inference worker, jangan block di sini ──
            submitted = self._submit_for_inference(frame_infer)
            if not submitted:
                # Inference queue penuh → pakai preview lama, lanjut ambil frame
                with preview_lock:
                    preview_frames[self.cid] = view
                continue

            # Tunggu hasil inferensi (timeout 0.5s)
            raw_dets_infer = self._get_inference_result(timeout=0.5)
            raw_dets_infer = raw_dets_infer[:20]

            tracked = self.tracker.update(raw_dets_infer)

            if tracked:
                self.last_detect_time = time.time()

            bbox_orig_map = {}
            for (x1, y1, x2, y2, conf, cls_name) in raw_dets_infer:
                bbox_orig_map[(x1, y1, x2, y2, cls_name)] = (
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y),
                )

            for obj in tracked:
                bx1, by1, bx2, by2 = obj["bbox"]
                cls_name  = obj["cls_name"]
                bbox_orig = bbox_orig_map.get(
                    (bx1, by1, bx2, by2, cls_name),
                    (int(bx1 * scale_x), int(by1 * scale_y),
                     int(bx2 * scale_x), int(by2 * scale_y))
                )
                self._process_object(
                    obj, frame_original, view,
                    orig_w, orig_h, scale_x, scale_y,
                    bbox_orig, roi_scaled, line_scaled
                )

            # ── Overlay ──────────────────────────────────────────────────────
            active = self.tracker.active_count()
            y_off  = 20
            for cls_n, cnt in active.items():
                cv2.putText(
                    view, f"{cls_n}: {cnt}", (8, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), WIDTH_LINE,
                )
                y_off += 20

            if self.vehicle_enabled:
                cv2.polylines(view, [roi_scaled], True, (255, 255, 0), WIDTH_LINE)
            else:
                cv2.polylines(view, [roi_scaled], True, (80, 80, 80), WIDTH_LINE)

            if not self.vehicle_enabled:
                cv2.putText(
                    view, "VEHICLE: OFF", (8, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), WIDTH_LINE,
                )
            elif self.line_enabled:
                (lx1, ly1), (lx2, ly2) = line_scaled
                cv2.line(view, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)

                dx     = lx2 - lx1
                dy     = ly2 - ly1
                length = math.hypot(dx, dy) or 1
                offset = 22
                mid_x  = (lx1 + lx2) // 2
                mid_y  = (ly1 + ly2) // 2
                nx     = int(-dy / length * offset)
                ny     = int( dx / length * offset)

                FONT       = cv2.FONT_HERSHEY_SIMPLEX
                FONT_SCALE = 0.45
                THICKNESS  = WIDTH_LINE

                (wa, ha), _ = cv2.getTextSize("A : IN",  FONT, FONT_SCALE, THICKNESS)
                (wb, hb), _ = cv2.getTextSize("B : OUT", FONT, FONT_SCALE, THICKNESS)

                ax = mid_x + nx
                ay = mid_y + ny
                bx = mid_x - nx
                by = mid_y - ny

                if nx < 0:
                    ax -= wa
                if nx > 0:
                    bx -= wb
                elif nx == 0:
                    ax -= wa // 2
                    bx -= wb // 2

                ay += ha // 2
                by += hb // 2

                cv2.putText(view, "A : IN",  (ax, ay), FONT, FONT_SCALE, (0, 255, 0), THICKNESS)
                cv2.putText(view, "B : OUT", (bx, by), FONT, FONT_SCALE, (0, 0, 255), THICKNESS)
            else:
                cv2.putText(
                    view, "LINE: OFF", (8, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), WIDTH_LINE,
                )

            if self.running:
                with preview_lock:
                    preview_frames[self.cid] = view

    def _reconnect(self):
        print(f"[RECONNECT] {self.cid} -> {self.name_camera}")
        self.reconnecting = True
        self.cap.release()
        time.sleep(2)

        self.cap = cv2.VideoCapture(self.stream_source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.connected    = True
            self.reconnecting = False
            self.dead         = False
            print(f"[RECONNECT OK] {self.cid}")
        else:
            self.dead = True
            print(f"[RECONNECT FAIL] {self.cid} -> {self.name_camera}")

        self.bad     = 0
        self.tracker = ObjectTracker()
        print(f"[TRACKER RESET] {self.cid}")


# ================= LOAD CAMERAS =================

def load_cameras():
    if DEBUG_MODE:
        print("[DEBUG MODE] Load video dari folder lokal")
        videos = []
        for file in os.listdir(DEBUG_VIDEO_DIR):
            if file.lower().endswith(".mp4"):
                videos.append({
                    "cctv_id":         file,
                    "client_id":       "debug",
                    "stream_url":      os.path.join(DEBUG_VIDEO_DIR, file),
                    "name":            file,
                    "roi":             None,
                    "line":            None,
                    "line_in_dir":     None,
                    "line_enabled":    None,
                    "vehicle_enabled": None,
                })
        print(f"[DEBUG] Total video: {len(videos)}")
        return videos
    else:
        print("[PRODUCTION MODE] Load kamera dari API")
        try:
            r    = requests.get(f"{ENDPOINT_URL}?service_id={SERVICE_ID}", timeout=10)
            data = r.json()["data"]
            print(f"[API] Total kamera: {len(data)}")
            return data
        except Exception as e:
            print("[API ERROR]", e)
            return []


# ================= STATUS MONITOR =================

def print_camera_status():
    with camera_lock:
        total        = len(active_cameras)
        connected    = sum(1 for w in active_cameras.values() if w.connected)
        reconnecting = sum(1 for w in active_cameras.values() if w.reconnecting)
        dead         = sum(1 for w in active_cameras.values() if w.dead)

    print("\n========== CAMERA STATUS ==========")
    print(f"Total Camera      : {total}")
    print(f"Connected         : {connected}")
    print(f"Reconnecting      : {reconnecting}")
    print(f"Dead              : {dead}")
    print(f"Webhook Queue     : {webhook_queue.qsize()}")
    print(f"Infer Queue       : {_infer_input_queue.qsize()}")   # ← tambahan monitoring
    print("===================================\n")

def monitoring_worker():
    while True:
        print_camera_status()
        time.sleep(10)


# ================= CAMERA MANAGER =================

def camera_manager():
    print("\n=== CAMERA MANAGER STARTED ===")
    while True:
        cams    = load_cameras()
        api_ids = {c["cctv_id"] for c in cams}

        with camera_lock:
            active_ids  = set(active_cameras.keys())
            new_ids     = api_ids - active_ids
            removed_ids = active_ids - api_ids

            for c in cams:
                cid = c["cctv_id"]

                if cid in new_ids:
                    try:
                        w = CameraWorker(
                            cid,
                            c["client_id"],
                            c["stream_url"],
                            c.get("name", "unknown"),
                            roi             = c.get("roi"),
                            line            = c.get("line"),
                            line_in_dir     = c.get("line_in_dir"),
                            line_enabled    = c.get("line_enabled"),
                            vehicle_enabled = c.get("vehicle_enabled"),
                        )
                        Thread(target=w.run, daemon=True).start()
                        active_cameras[cid] = w
                        print(f"[NEW CAMERA] {cid} -> {c.get('name','?')}")
                    except Exception as e:
                        print(f"[FAILED START] {cid} -> {e}")
                else:
                    active_cameras[cid].update_config(
                        roi             = c.get("roi"),
                        line            = c.get("line"),
                        line_in_dir     = c.get("line_in_dir"),
                        line_enabled    = c.get("line_enabled"),
                        vehicle_enabled = c.get("vehicle_enabled"),
                    )

            for rid in removed_ids:
                try:
                    active_cameras[rid].stop()
                    del active_cameras[rid]
                    print(f"[CAMERA REMOVED] {rid}")
                except Exception:
                    pass

            with preview_lock:
                stale_cids = [cid for cid in list(preview_frames.keys())
                              if cid not in active_cameras]
                for cid in stale_cids:
                    del preview_frames[cid]
                    print(f"[PREVIEW REMOVED] {cid}")

            print(f"[MANAGER] Source API: {len(api_ids)} | Worker aktif: {len(active_cameras)}")

        time.sleep(CAMERA_REFRESH_INTERVAL)


Thread(target=camera_manager,    daemon=True).start()
Thread(target=monitoring_worker, daemon=True).start()


# ================= DISPLAY GRID =================

if ENABLE_VIEW:
    cv2.namedWindow("ONNX Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ONNX Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        with preview_lock:
            snapshot = list(preview_frames.values())

        if not snapshot:
            blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                blank, "No active cameras",
                (DISPLAY_WIDTH // 2 - 140, DISPLAY_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2,
            )
            cv2.imshow("ONNX Detection", blank)
            if cv2.waitKey(200) == ord("q"):
                break
            continue

        total = len(snapshot)
        cols  = math.ceil(math.sqrt(total))
        rows  = math.ceil(total / cols)
        tw    = max(1, DISPLAY_WIDTH  // cols)
        th    = max(1, DISPLAY_HEIGHT // rows)

        imgs = []
        for f in snapshot:
            try:
                imgs.append(cv2.resize(f, (tw, th)))
            except Exception:
                imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        while len(imgs) < rows * cols:
            imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        grid = []
        for r in range(rows):
            row_imgs = imgs[r * cols:(r + 1) * cols]
            try:
                grid.append(cv2.hconcat(row_imgs))
            except Exception:
                grid.append(np.zeros((th, tw * cols, 3), dtype=np.uint8))

        try:
            final = cv2.vconcat(grid)
        except Exception:
            final = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        cv2.imshow("ONNX Detection", final)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

else:
    while True:
        time.sleep(1)