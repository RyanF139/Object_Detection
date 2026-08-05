import cv2
import time
import av
import os
import sys
import site

# ================= FIX WINDOWS CUDA DLLS =================
if os.name == 'nt':
    try:
        # Menambahkan path nvidia/* dll agar onnxruntime-gpu mendeteksi CUDA 11/12 di Windows
        for sp in site.getsitepackages():
            if 'site-packages' in sp:
                nvidia_dir = os.path.join(sp, 'nvidia')
                if os.path.exists(nvidia_dir):
                    for root, dirs, files in os.walk(nvidia_dir):
                        if 'bin' in root:
                            try:
                                os.add_dll_directory(root)
                                os.environ['PATH'] = root + os.pathsep + os.environ.get('PATH', '')
                            except Exception:
                                pass
    except Exception as e:
        pass

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

SERVICE_ID      = os.getenv("SERVICE_ID", "capture_100")
MODEL_PATH      = os.getenv("MODEL_PATH", "models/yolov8s.onnx")
FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", "models/face_detection_yunet_2023mar.onnx")

ENDPOINT_URL            = os.getenv("CCTV_ENDPOINT", "http://localhost:8000/api/webhook/cctv/")
CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 10))
WEBHOOK_URL             = os.getenv("WEBHOOK_URL")

_DEVICE_ENV    = os.getenv("DEVICE", "cuda").lower()
CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", 0))

# ================= RTSP OPTIONS =================
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"   # suppress FFmpeg decode error log


# ================= CUDA CHECK =================

def _check_onnxruntime_cuda() -> bool:
    import subprocess
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("[CUDA CHECK] onnxruntime: Tidak ada physical GPU / driver NVIDIA terdeteksi")
        return False

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("[CUDA CHECK] onnxruntime: CUDAExecutionProvider tidak tersedia")
        return False
    print(f"[CUDA CHECK] onnxruntime: CUDAExecutionProvider OK (device_id={CUDA_DEVICE_ID})")
    return True


def _check_opencv_cuda() -> bool:
    if not hasattr(cv2, "cuda"):
        print("[CUDA CHECK] OpenCV: dikompilasi tanpa CUDA (tidak ada cv2.cuda module)")
        print("             → Build OpenCV dari source dengan -DWITH_CUDA=ON")
        return False

    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count <= 0:
            print("[CUDA CHECK] OpenCV: tidak ada GPU CUDA terdeteksi")
            return False
        if CUDA_DEVICE_ID >= gpu_count:
            print(f"[CUDA CHECK] OpenCV: CUDA_DEVICE_ID={CUDA_DEVICE_ID} >= gpu_count={gpu_count}")
            return False
        print(f"[CUDA CHECK] OpenCV: {gpu_count} GPU terdeteksi OK")
    except Exception as e:
        print(f"[CUDA CHECK] OpenCV cv2.cuda error: {e}")
        return False

    try:
        dummy_det = cv2.FaceDetectorYN_create(
            FACE_MODEL_PATH, "", (32, 32),
            score_threshold=0.5, nms_threshold=0.4, top_k=1,
            backend_id=cv2.dnn.DNN_BACKEND_CUDA,
            target_id=cv2.dnn.DNN_TARGET_CUDA,
        )
        dummy_img = np.full((32, 32, 3), 128, dtype=np.uint8)
        dummy_det.setInputSize((32, 32))
        dummy_det.detect(dummy_img)
        del dummy_det
        print("[CUDA CHECK] OpenCV DNN CUDA backend: OK")
        return True
    except Exception as e:
        print(f"[CUDA CHECK] OpenCV DNN CUDA backend gagal: {e}")
        print("             → Pastikan OpenCV di-build dengan -DOPENCV_DNN_CUDA=ON dan -DWITH_CUDNN=ON")
        return False


def _resolve_device() -> str:
    if _DEVICE_ENV != "cuda":
        print("[DEVICE] Mode: CPU (dipaksa lewat .env DEVICE=cpu)")
        return "cpu"

    print("[DEVICE] Memeriksa ketersediaan CUDA...")
    ort_ok = _check_onnxruntime_cuda()
    ocv_ok = _check_opencv_cuda()

    if ort_ok and ocv_ok:
        print(f"[DEVICE] Mode: FULL GPU — onnxruntime CUDA + OpenCV DNN CUDA (device_id={CUDA_DEVICE_ID})")
        return "cuda"
    elif ort_ok and not ocv_ok:
        print("[DEVICE] Mode: PARTIAL GPU — onnxruntime CUDA OK, OpenCV CUDA GAGAL")
        print("         Vehicle inference -> GPU | Face inference -> CPU")
        return "cuda_partial"
    else:
        print("[DEVICE] Mode: CPU (semua CUDA gagal, fallback otomatis)")
        return "cpu"


DEVICE = _resolve_device()

_ORT_USE_CUDA = DEVICE in ("cuda", "cuda_partial")
_OCV_USE_CUDA = DEVICE == "cuda"

# ================= THRESHOLDS =================

CONF_THRESHOLD   = float(os.getenv("CONF_THRESHOLD",   0.30))
IOU_THRESHOLD    = float(os.getenv("IOU_THRESHOLD",    0.40))
SCORE_THRESHOLD  = float(os.getenv("SCORE_THRESHOLD",  0.35))
BLUR_THRESHOLD   = float(os.getenv("BLUR_THRESHOLD",   0))
MIN_SIZE_CAPTURE = int(os.getenv("MIN_SIZE_CAPTURE",   0))
MAX_FACE_SIZE    = int(os.getenv("MAX_FACE_SIZE",       800))

FACE_COOLDOWN       = float(os.getenv("FACE_COOLDOWN",       8))
FACE_MOVE_THRESHOLD = int(os.getenv("FACE_MOVE_THRESHOLD",   80))
FACE_BUCKET_SIZE    = int(os.getenv("FACE_BUCKET_SIZE",       220))

SAVE_FOLDER      = os.getenv("SAVE_FOLDER",   "image_detection")
FACE_FOLDER_BASE = os.getenv("FACE_FOLDER",   "image_face")
MAX_IMAGES       = int(os.getenv("MAX_IMAGES", 1000))
CROP_PADDING      = float(os.getenv("CROP_PADDING",      0.20))
FACE_CROP_PADDING = float(os.getenv("FACE_CROP_PADDING", 0.35))

SAVE_IMAGE         = os.getenv("SAVE_IMAGE",         "true").lower() == "true"
SAVE_IMAGE_VEHICLE = os.getenv("SAVE_IMAGE_VEHICLE", "true").lower() == "true"
SAVE_IMAGE_FACE    = os.getenv("SAVE_IMAGE_FACE",    "true").lower() == "true"

SAVE_MAX_WIDTH   = int(os.getenv("SAVE_MAX_WIDTH",  0))
TARGET_MAX_WIDTH = int(os.getenv("RESIZE_WIDTH",    640))
IMG_SIZE         = int(os.getenv("IMG_SIZE",        320))

FRAME_FPS    = int(os.getenv("FRAME_FPS",    12))
IDLE_FPS     = int(os.getenv("IDLE_FPS",      3))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 10))

TRACK_MAX_DIST        = int(os.getenv("TRACK_MAX_DIST", 250))
TRACK_PERSON_MAX_DIST = int(os.getenv("TRACK_PERSON_MAX_DIST", 35))
TRACK_MAX_MISS        = int(os.getenv("TRACK_MAX_MISS", 100))
TRACK_MOVE_THR        = int(os.getenv("TRACK_MOVE_THR", 40))

ENABLE_VIEW    = os.getenv("ENABLE_VIEW",    "true").lower() == "true"
DISPLAY_WIDTH  = int(os.getenv("DISPLAY_WIDTH",  1200))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT",  800))

DEBUG_MODE      = os.getenv("DEBUG_MODE",      "false").lower() == "true"
DEBUG_VIDEO_DIR = os.getenv("DEBUG_VIDEO_DIR", "./sample_videos")

ONNX_INTRA_THREADS = int(os.getenv("ONNX_INTRA_THREADS", 4))
ONNX_INTER_THREADS = int(os.getenv("ONNX_INTER_THREADS", 2))
FACE_INFER_WORKERS = int(os.getenv("FACE_INFER_WORKERS", 2))
_INFER_WORKERS     = int(os.getenv("INFER_WORKERS",      2))

# Jumlah frame dibuang dari buffer sebelum retrieve (ambil frame terbaru)
GRAB_SKIP_COUNT = int(os.getenv("GRAB_SKIP_COUNT", 3))

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

DEFAULT_ROI_POLYGON = np.array([[0,0],[640,0],[640,360],[0,360]], dtype=np.int32)
WIDTH_LINE           = 1
DEFAULT_LINE         = [[0, 320], [640, 320]]
DEFAULT_LINE_IN_DIR  = "A"
DEFAULT_LINE_ENABLED    = True
DEFAULT_VEHICLE_ENABLED = os.getenv("VEHICLE_ENABLED", "true").lower() == "true"
DEFAULT_FACE_ENABLED    = os.getenv("FACE_ENABLED",    "true").lower() == "true"
DEFAULT_PERSON_ENABLED  = os.getenv("PERSON_ENABLED",  "true").lower() == "true"
DEFAULT_PERSON_ROIS     = []
DEFAULT_SAVE_IMAGE_VEHICLE = SAVE_IMAGE and SAVE_IMAGE_VEHICLE
DEFAULT_SAVE_IMAGE_FACE    = SAVE_IMAGE and SAVE_IMAGE_FACE

CLASS_CONFIG = {
    "person": {
        "conf":     float(os.getenv("PERSON_CONF",      0.50)),
        "color":    tuple(map(int, os.getenv("PERSON_COLOR",    "0,255,0").split(","))),
        "min_size": int(os.getenv("PERSON_MIN_SIZE",    0)),
        "max_size": int(os.getenv("PERSON_MAX_SIZE",    0)),
        "cooldown": int(os.getenv("PERSON_COOLDOWN",    10)),
    },
    "car": {
        "conf":     float(os.getenv("CAR_CONF",      0.40)),
        "color":    tuple(map(int, os.getenv("CAR_COLOR",    "255,0,0").split(","))),
        "min_size": int(os.getenv("CAR_MIN_SIZE",    5000)),
        "max_size": int(os.getenv("CAR_MAX_SIZE",    1000000)),
        "cooldown": int(os.getenv("CAR_COOLDOWN",    10)),
    },
    "bus": {
        "conf":     float(os.getenv("BUS_CONF",      0.40)),
        "color":    tuple(map(int, os.getenv("BUS_COLOR",    "255,165,0").split(","))),
        "min_size": int(os.getenv("BUS_MIN_SIZE",    50000)),
        "max_size": int(os.getenv("BUS_MAX_SIZE",    200000)),
        "cooldown": int(os.getenv("BUS_COOLDOWN",    10)),
    },
    "truck": {
        "conf":     float(os.getenv("TRUCK_CONF",    0.40)),
        "color":    tuple(map(int, os.getenv("TRUCK_COLOR",  "128,128,128").split(","))),
        "min_size": int(os.getenv("TRUCK_MIN_SIZE",  100000)),
        "max_size": int(os.getenv("TRUCK_MAX_SIZE",  300000)),
        "cooldown": int(os.getenv("TRUCK_COOLDOWN",  10)),
    },
    "motorcycle": {
        "conf":     float(os.getenv("MOTORCYCLE_CONF",    0.35)),
        "color":    tuple(map(int, os.getenv("MOTORCYCLE_COLOR", "0,255,255").split(","))),
        "min_size": int(os.getenv("MOTORCYCLE_MIN_SIZE",  1000)),
        "max_size": int(os.getenv("MOTORCYCLE_MAX_SIZE",  500000)),
        "cooldown": int(os.getenv("MOTORCYCLE_COOLDOWN",  10)),
    },
}

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
ALLOWED_CLASSES = set(CLASS_CONFIG.keys())

# ================= FOLDERS =================

DETECT_FOLDER     = os.path.join(SAVE_FOLDER,      "crop")
FRAME_FOLDER_VEH  = os.path.join(SAVE_FOLDER,      "frame")
FACE_CROP_FOLDER  = os.path.join(FACE_FOLDER_BASE, "face")
FACE_FRAME_FOLDER = os.path.join(FACE_FOLDER_BASE, "frame")

for _d in [DETECT_FOLDER, FRAME_FOLDER_VEH, FACE_CROP_FOLDER, FACE_FRAME_FOLDER]:
    os.makedirs(_d, exist_ok=True)

# ================= GLOBALS =================

JAKARTA_TZ     = timezone(timedelta(hours=7))
active_cameras = {}
camera_lock    = Lock()
preview_frames = {}
preview_lock   = Lock()

webhook_queue      = Queue(maxsize=1000)
face_webhook_queue = Queue(maxsize=1000)

# ================= SHARED ONNX SESSION =================

_shared_session    = None
_shared_input_name = None
_session_lock      = Lock()


def _build_cuda_provider_options() -> dict:
    """Return CUDA EP options yang kompatibel dengan Tesla GPU & Blackwell (RTX 5090).
    - Menghapus parameter alokasi memori paksa (arena/workspace) agar tidak memicu misaligned address di arsitektur Pascal (Tesla P4).
    """
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('CONTAINER_NAME') is not None
    dev_id = "0" if in_docker else str(CUDA_DEVICE_ID)
    
    return {
        "device_id": dev_id,
        "cudnn_conv_algo_search": "DEFAULT",  # Safe and stable fallback
    }


def build_shared_session():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = ONNX_INTRA_THREADS
    opts.inter_op_num_threads     = ONNX_INTER_THREADS
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    # Nonaktifkan fusi Gelu/FastGelu secara eksplisit untuk mencegah kegagalan alamat CUDA pada Blackwell (RTX 5090)
    try:
        opts.add_session_config_entry("session.disable_gelu_fusion", "1")
    except Exception:
        pass

    use_cuda  = _ORT_USE_CUDA
    providers = (
        [("CUDAExecutionProvider", _build_cuda_provider_options()), "CPUExecutionProvider"]
        if use_cuda else ["CPUExecutionProvider"]
    )

    sess       = ort.InferenceSession(MODEL_PATH, providers=providers, sess_options=opts)
    input_name = sess.get_inputs()[0].name

    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # --- warm-up run: fallback ke CPU jika CUDA gagal (CUDNN_STATUS_EXECUTION_FAILED_CUDART) ---
    if use_cuda:
        try:
            sess.run(None, {input_name: dummy})
        except Exception as cuda_err:
            print(f"[ONNX SHARED] CUDA warm-up GAGAL: {cuda_err}")
            print("[ONNX SHARED] Fallback ke CPUExecutionProvider...")
            sess       = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=opts)
            input_name = sess.get_inputs()[0].name
            sess.run(None, {input_name: dummy})
            print("[ONNX SHARED] CPU fallback warm-up OK")
    else:
        sess.run(None, {input_name: dummy})

    print(f"[ONNX SHARED] provider={sess.get_providers()[0]} | device_id={CUDA_DEVICE_ID}")
    print(f"[ONNX SHARED] intra={ONNX_INTRA_THREADS} | inter={ONNX_INTER_THREADS}")
    return sess, input_name


def get_shared_session():
    global _shared_session, _shared_input_name
    with _session_lock:
        if _shared_session is None:
            _shared_session, _shared_input_name = build_shared_session()
    return _shared_session, _shared_input_name


print(f"[STARTUP] Inisialisasi ONNX session | device={DEVICE.upper()}...")
get_shared_session()
print("[STARTUP] ONNX session siap")

# ================= SHARED YUNET SESSION =================

_shared_face_detector = None
_face_session_lock    = Lock()


def build_shared_face_detector():
    backend_id   = cv2.dnn.DNN_BACKEND_CUDA   if _OCV_USE_CUDA else cv2.dnn.DNN_BACKEND_OPENCV
    target_id    = cv2.dnn.DNN_TARGET_CUDA    if _OCV_USE_CUDA else cv2.dnn.DNN_TARGET_CPU
    device_label = "CUDA" if _OCV_USE_CUDA else "CPU"

    det = cv2.FaceDetectorYN_create(
        FACE_MODEL_PATH, "", (640, 640),
        score_threshold=SCORE_THRESHOLD, nms_threshold=0.4, top_k=5000,
        backend_id=backend_id, target_id=target_id,
    )
    try:
        dummy = np.full((640, 640, 3), 128, dtype=np.uint8)
        det.setInputSize((640, 640))
        det.detect(dummy)
    except Exception as e:
        print(f"[YUNET SHARED] Peringatan warmup YuNet gagal: {e}")
        
    print(f"[YUNET SHARED] device={device_label} | model={FACE_MODEL_PATH}")
    return det


def get_shared_face_detector():
    global _shared_face_detector
    with _face_session_lock:
        if _shared_face_detector is None:
            _shared_face_detector = build_shared_face_detector()
    return _shared_face_detector


print(f"[STARTUP] Inisialisasi YuNet session | device={DEVICE.upper()}...")
get_shared_face_detector()
print("[STARTUP] YuNet session siap")

# ================= FACE INFERENCE WORKERS =================

_face_infer_queue: Queue = Queue(maxsize=15)


def face_inference_worker(worker_id: int):
    backend_id   = cv2.dnn.DNN_BACKEND_CUDA   if _OCV_USE_CUDA else cv2.dnn.DNN_BACKEND_OPENCV
    target_id    = cv2.dnn.DNN_TARGET_CUDA    if _OCV_USE_CUDA else cv2.dnn.DNN_TARGET_CPU
    device_label = "CUDA" if _OCV_USE_CUDA else "CPU"

    det = cv2.FaceDetectorYN_create(
        FACE_MODEL_PATH, "", (640, 640),
        score_threshold=SCORE_THRESHOLD, nms_threshold=0.4, top_k=5000,
        backend_id=backend_id, target_id=target_id,
    )
    try:
        dummy = np.full((640, 640, 3), 128, dtype=np.uint8)
        det.setInputSize((640, 640))
        det.detect(dummy)
    except Exception as e:
        print(f"[FACE WORKER {worker_id}] Warmup gagal: {e} — worker tetap jalan")

    print(f"[FACE WORKER {worker_id}] Started | device={device_label}")

    while True:
        try:
            item = _face_infer_queue.get(timeout=1.0)
        except Empty:
            continue
        if item is None:
            break
        cid, frame_resized, inv_scale, result_q = item
        try:
            h, w = frame_resized.shape[:2]
            det.setInputSize((w, h))
            with _gpu_infer_lock:
                _, faces = det.detect(frame_resized)
            result_q.put(("ok", faces, inv_scale))
        except Exception as e:
            result_q.put(("err", None, 1.0))
            print(f"[FACE WORKER {worker_id}] Error cam={cid}: {e}")
        finally:
            _face_infer_queue.task_done()


for _i in range(FACE_INFER_WORKERS):
    Thread(target=face_inference_worker, args=(_i,), daemon=True).start()

print(f"[STARTUP] {FACE_INFER_WORKERS} face inference worker(s) started")

# ================= VEHICLE INFERENCE WORKERS =================

_infer_input_queue: Queue = Queue(maxsize=15)


_gpu_infer_lock = Lock()

def inference_worker(worker_id: int):
    sess, input_name = get_shared_session()
    print(f"[INFER WORKER {worker_id}] Started | provider={sess.get_providers()[0]}")
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
            with _gpu_infer_lock:
                outputs = sess.run(None, {input_name: inp})
            h, w       = frame_infer.shape[:2]
            results    = postprocess(outputs, scale, w, h)
            result_q.put(("ok", results))
        except Exception as e:
            result_q.put(("err", []))
            print(f"[INFER WORKER {worker_id}] Error cam={cid}: {e}")
        finally:
            _infer_input_queue.task_done()


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
    sides = [point_side_of_line(px, py, lx1, ly1, lx2, ly2) for px, py in history]
    last_direction = None
    for i in range(1, len(sides)):
        prev, curr = sides[i - 1], sides[i]
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


def line_dist_px(px, py, lx1, ly1, lx2, ly2):
    line_len = math.hypot(lx2 - lx1, ly2 - ly1)
    if line_len == 0:
        return 0.0
    return point_side_of_line(px, py, lx1, ly1, lx2, ly2) / line_len


def is_just_crossed_line(history, line_pts, line_in_dir, margin=10.0):
    if len(history) < 2:
        return None
    (lx1, ly1), (lx2, ly2) = line_pts
    px_prev, py_prev = history[-2]
    px_curr, py_curr = history[-1]

    prev_dist = line_dist_px(px_prev, py_prev, lx1, ly1, lx2, ly2)
    curr_dist = line_dist_px(px_curr, py_curr, lx1, ly1, lx2, ly2)

    # Wajib melintasi zona mati (deadband) secara bersih dari > margin ke < -margin
    crossed_a_to_b = (prev_dist > margin and curr_dist < -margin)
    crossed_b_to_a = (prev_dist < -margin and curr_dist > margin)

    if not crossed_a_to_b and not crossed_b_to_a:
        return None

    if line_in_dir == "A":
        return "IN" if crossed_a_to_b else "OUT"
    else:
        return "IN" if crossed_b_to_a else "OUT"


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
    return (int(rx1 * sx), int(ry1 * sy)), (int(rx2 * sx), int(ry2 * sy))


def iso_timestamp():
    return datetime.now(JAKARTA_TZ).isoformat(timespec="seconds")


def iso_name(prefix, cls_name, ts_iso, obj_id=None, direction=None):
    safe_ts  = ts_iso.replace(":", "-")
    id_part  = f"_id{obj_id}" if obj_id is not None else ""
    dir_part = f"_{direction}" if direction else ""
    return f"{prefix}_{cls_name}{id_part}{dir_part}_{safe_ts}.jpg"


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


def expand_crop_bbox(x1, y1, x2, y2, img_w, img_h, padding=None):
    if padding is None:
        padding = CROP_PADDING
    w, h   = x2 - x1, y2 - y1
    pad_w  = int(w * padding)
    pad_h  = int(h * padding)
    return max(0, x1 - pad_w), max(0, y1 - pad_h), min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)


def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def center_dist(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def rect_intersects_polygon(x1, y1, x2, y2, polygon) -> bool:
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # Shift corner checks slightly towards the vertical center to avoid empty bounding box corners
    # (which occur when arms/limbs are extended horizontally or diagonally)
    dy_top = int((cy - y1) * 0.35)
    dy_bot = int((y2 - cy) * 0.35)
    
    key_points = [
        (cx, cy),           # Center
        (cx, y1),           # Top-Center (Head)
        (cx, y2),           # Bottom-Center (Feet)
        (x1, cy),           # Left-Center (Left Arm/Hand)
        (x2, cy),           # Right-Center (Right Arm/Hand)
        (x1, y1 + dy_top),  # Upper-Left (Shifted from extreme corner)
        (x2, y1 + dy_top),  # Upper-Right (Shifted from extreme corner)
        (x1, y2 - dy_bot),  # Lower-Left (Shifted from extreme corner)
        (x2, y2 - dy_bot)   # Lower-Right (Shifted from extreme corner)
    ]
    for pt in key_points:
        if cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0:
            return True
    return False


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
                return (int(arr[0][0]), int(arr[0][1])), (int(arr[1][0]), int(arr[1][1]))
        except Exception:
            pass
        print(f"[LINE PARSE ERROR] Invalid line: {raw_line!r}. Using default.")
    pts = DEFAULT_LINE
    return (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1]))


def parse_line_in_dir(raw_dir):
    if raw_dir in ("A", "B"):
        return raw_dir
    if raw_dir is not None:
        print(f"[LINE DIR ERROR] Invalid: {raw_dir!r}. Using default '{DEFAULT_LINE_IN_DIR}'.")
    return DEFAULT_LINE_IN_DIR


def parse_multi_lines(raw_lines):
    if raw_lines is None:
        return []
    if isinstance(raw_lines, str):
        try:
            raw_lines = json.loads(raw_lines)
        except Exception:
            return []
    if not isinstance(raw_lines, list):
        return []
    
    parsed = []
    for idx, item in enumerate(raw_lines):
        if not isinstance(item, dict):
            continue
        line = item.get("line")
        line_in_dir = item.get("line_in_dir", "A")
        line_name = item.get("name", str(idx))
        
        try:
            arr = np.array(line, dtype=np.int32)
            if arr.shape == (2, 2):
                parsed.append({
                    "line": ((int(arr[0][0]), int(arr[0][1])), (int(arr[1][0]), int(arr[1][1]))),
                    "line_in_dir": "A" if line_in_dir == "A" else "B",
                    "index": idx,
                    "name": line_name
                })
        except Exception:
            pass
    return parsed


def parse_multi_rois(raw_rois):
    if raw_rois is None:
        return []
    if isinstance(raw_rois, str):
        try:
            raw_rois = json.loads(raw_rois)
        except Exception:
            return []
    if not isinstance(raw_rois, list):
        return []
    
    parsed = []
    for idx, item in enumerate(raw_rois):
        if not isinstance(item, dict):
            continue
        roi = item.get("roi")
        roi_enabled = item.get("roi_enabled", True)
        if roi_enabled is None:
            roi_enabled = True
        roi_name = item.get("name", str(item.get("index", idx)))
        
        try:
            arr = np.array(roi, dtype=np.int32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                parsed.append({
                    "roi": arr.tolist(),
                    "roi_enabled": bool(roi_enabled),
                    "index": int(item.get("index", idx)),
                    "name": roi_name
                })
        except Exception:
            pass
    return parsed



def prepare_for_save(img: np.ndarray) -> np.ndarray:
    if SAVE_MAX_WIDTH <= 0:
        return img
    h, w = img.shape[:2]
    if w <= SAVE_MAX_WIDTH:
        return img
    scale = SAVE_MAX_WIDTH / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def calc_sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ================= FACE QUALITY FILTER =================

def is_valid_face(f, x, y, fw, fh, scale) -> bool:
    lx, ly = f[4] * scale, f[5] * scale
    rx, ry = f[6] * scale, f[7] * scale
    nx     = f[8] * scale

    eye_angle = abs(math.degrees(math.atan2(ry - ly, rx - lx)))
    if eye_angle > 40:
        return False

    ratio = fw / float(fh)
    if ratio < 0.50 or ratio > 1.50:
        return False

    nose_ratio = (nx - x) / float(fw)
    if nose_ratio < 0.15 or nose_ratio > 0.85:
        return False

    return True


# ================= ONNX PRE/POSTPROCESS =================

def preprocess(img: np.ndarray):
    h, w    = img.shape[:2]
    scale   = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh  = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    inp = canvas.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    inp = np.expand_dims(inp, axis=0)
    inp = np.ascontiguousarray(inp)
    return inp, scale


def postprocess(outputs, scale: float, orig_w: int, orig_h: int):
    out = outputs[0]
    if out.shape[1] < out.shape[2]:
        out = np.transpose(out, (0, 2, 1))
    out = out[0]

    per_class = {}
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
        indices = cv2.dnn.NMSBoxes(data["boxes"], data["scores"], cfg["conf"], IOU_THRESHOLD)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = data["boxes"][i]
                results.append((x, y, x + w, y + h, data["scores"][i], COCO_CLASSES[data["cls_ids"][i]]))
    return results


# ================= TRACKER =================

class ObjectTracker:

    _global_id = 0
    _id_lock   = Lock()

    @classmethod
    def _next_id(cls):
        with cls._id_lock:
            cls._global_id += 1
            return cls._global_id

    def __init__(self):
        self.tracks = {}

    def update(self, detections: list) -> list:
        now = time.time()
        for t in self.tracks.values():
            t["matched"] = False

        enriched = []

        for (x1, y1, x2, y2, conf, cls_name) in detections:
            cx, cy    = bbox_center(x1, y1, x2, y2)
            best_id   = None
            best_dist = float("inf")

            for tid, t in self.tracks.items():
                if t["cls_name"] != cls_name:
                    continue
                limit_dist = TRACK_PERSON_MAX_DIST if cls_name == "person" else TRACK_MAX_DIST
                d = center_dist((cx, cy), (t["cx"], t["cy"]))
                if d < best_dist and d < limit_dist:
                    best_dist = d
                    best_id   = tid

            cfg = CLASS_CONFIG[cls_name]

            if best_id is not None:
                t            = self.tracks[best_id]
                t["matched"] = True
                moved        = center_dist((cx, cy), (t["cx"], t["cy"])) > TRACK_MOVE_THR
                cooldown_ok  = (now - t["last_sent"]) >= cfg["cooldown"]
                should_send  = (not t["sent_once"]) or (cooldown_ok and moved)

                t["cx"], t["cy"] = cx, cy
                history_pt = (cx, y2)
                t["history"].append(history_pt)
                if len(t["history"]) > 30:
                    t["history"].pop(0)
                t["x1"], t["y1"], t["x2"], t["y2"] = x1, y1, x2, y2
                t["miss"] = 0

                if should_send:
                    t["last_sent"] = now
                    t["sent_once"] = True

                enriched.append({
                    "bbox": (x1, y1, x2, y2), "conf": conf, "cls_name": cls_name,
                    "obj_id": best_id, "should_send": should_send, "is_new": False,
                })
            else:
                new_id = self._next_id()
                initial_history_pt = (cx, y2)
                self.tracks[new_id] = {
                    "history": [initial_history_pt], "id": new_id, "cls_name": cls_name,
                    "cx": cx, "cy": cy, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "miss": 0, "last_sent": now, "sent_once": True,
                    "matched": True, "last_cross_dir": None,
                }
                enriched.append({
                    "bbox": (x1, y1, x2, y2), "conf": conf, "cls_name": cls_name,
                    "obj_id": new_id, "should_send": True, "is_new": True,
                })

        dead_ids = [
            tid for tid, t in self.tracks.items()
            if not t["matched"] and t["miss"] + 1 >= TRACK_MAX_MISS
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        for t in self.tracks.values():
            if not t["matched"]:
                t["miss"] += 1

        return enriched

    def active_count(self) -> dict:
        count = {}
        for t in self.tracks.values():
            count[t["cls_name"]] = count.get(t["cls_name"], 0) + 1
        return count


# ================= WEBHOOK WORKERS =================

def vehicle_webhook_worker():
    while True:
        item = webhook_queue.get()
        if item is None:
            break

        unpacked = item
        line_name  = None
        roi_info   = None

        if len(unpacked) == 16:
            # ROI person event — element ke-15 adalah dict {"index", "name"}, ke-16 flag True
            (
                crop_bytes, frame_clean_bytes,
                crop_name, frame_name,
                name_camera, ts_iso,
                bbox, conf, cls_name,
                cid, client_id,
                obj_id, is_new, _, roi_info, _is_roi,
            ) = unpacked
        elif len(unpacked) == 15:
            (
                crop_bytes, frame_clean_bytes,
                crop_name, frame_name,
                name_camera, ts_iso,
                bbox, conf, cls_name,
                cid, client_id,
                obj_id, is_new, direction, line_name,
            ) = unpacked
        else:
            (
                crop_bytes, frame_clean_bytes,
                crop_name, frame_name,
                name_camera, ts_iso,
                bbox, conf, cls_name,
                cid, client_id,
                obj_id, is_new, direction,
            ) = unpacked

        if roi_info is not None:
            # --- ROI detection event (person entered ROI) ---
            payload = {
                "timestamp":  ts_iso,
                "type":       "roi_detection_service",
                "class_name": cls_name,
                "bbox":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "confidence": round(conf, 4),
                "channel_id": cid,
                "client_id":  client_id,
                "roi_index":  roi_info.get("index"),
                "roi_name":   roi_info.get("name"),
            }
        else:
            # --- Line / vehicle detection event ---
            payload = {
                "timestamp":  ts_iso,
                "type":       "line_detection_service" if cls_name == "person" else "object_detection_service",
                "class_name": cls_name,
                "bbox":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "confidence": round(conf, 4),
                "channel_id": cid,
                "client_id":  client_id,
                "direction":  direction,
            }
            if line_name is not None:
                if isinstance(line_name, dict):
                    raw_name = line_name.get("name")
                    raw_idx  = line_name.get("index")
                    payload["line_name"]  = raw_name if raw_name and isinstance(raw_name, str) else f"Line {raw_idx}"
                    payload["line_index"] = raw_idx
                else:
                    payload["line_name"]  = str(line_name)
                    payload["line_index"] = line_name

        print(f"\n[WEBHOOK {cls_name.upper()}] {json.dumps({'url': WEBHOOK_URL, 'frame': frame_name, 'data': payload}, indent=2)}")

        sent = False
        for attempt in range(2):
            try:
                resp = requests.post(
                    WEBHOOK_URL,
                    files=[
                        ("files", (frame_name, frame_clean_bytes, "image/jpeg")),
                        ("files", (crop_name,  crop_bytes,        "image/jpeg")),
                    ],
                    data=payload,
                    timeout=10,
                )
                resp.raise_for_status()
                print(f"[WEBHOOK {cls_name.upper()} OK] {frame_name} | id={obj_id} | http={resp.status_code}")
                sent = True
                break
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                print(f"[WEBHOOK {cls_name.upper()} FAIL HTTP {status}] attempt={attempt + 1}")
                if attempt == 0:
                    time.sleep(2)
            except requests.exceptions.ConnectionError:
                print(f"[WEBHOOK {cls_name.upper()} FAIL CONNECTION] attempt={attempt + 1}")
                if attempt == 0:
                    time.sleep(2)
            except requests.exceptions.Timeout:
                print(f"[WEBHOOK {cls_name.upper()} FAIL TIMEOUT] attempt={attempt + 1}")
                if attempt == 0:
                    time.sleep(1)
            except Exception as e:
                print(f"[WEBHOOK {cls_name.upper()} FAIL UNEXPECTED] {e}")
                break

        if not sent:
            print(f"[WEBHOOK {cls_name.upper()} GIVE UP] {frame_name} | id={obj_id}")

        webhook_queue.task_done()


def face_webhook_worker():
    while True:
        item = face_webhook_queue.get()
        if item is None:
            break
        try:
            face_bytes, frame_bytes, face_name, frame_name, \
                name_camera, ts_iso, bbox, score, cid, client_id = item

            payload = {
                "timestamp":  ts_iso,
                "type":       "face_detection_service",
                "bbox":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "confidence": round(score, 4),
                "channel_id": cid,
                "client_id":  client_id,
            }
            print(f"\n[WEBHOOK FACE] {json.dumps({'url': WEBHOOK_URL, 'frame': frame_name, 'data': payload}, indent=2)}")
            resp = requests.post(
                WEBHOOK_URL,
                files=[("files", (frame_name, frame_bytes, "image/jpeg"))],
                data=payload,
                timeout=10,
            )
            resp.raise_for_status()
            print(f"[WEBHOOK FACE OK] {face_name} | conf={score:.3f} | http={resp.status_code}")
        except Exception as e:
            print(f"[WEBHOOK FACE FAIL] {e}")
        finally:
            face_webhook_queue.task_done()


if WEBHOOK_URL:
    Thread(target=vehicle_webhook_worker, daemon=True).start()
    Thread(target=face_webhook_worker,    daemon=True).start()


# ================= CAMERA WORKER =================

class CameraWorker:

    def __init__(self, cid, client_id, url, name,
                 roi=None, line=None, line_in_dir=None,
                 line_enabled=None, vehicle_enabled=None,
                 face_enabled=None,
                 save_image_vehicle=None, save_image_face=None,
                 person_enabled=None, person_lines=None, person_rois=None):

        self.cid           = cid
        self.client_id     = client_id
        self.name_camera   = name
        self.stream_source = self._build_stream_url(url)
        self.running       = True

        self.vehicle_enabled = bool(vehicle_enabled) if vehicle_enabled is not None else DEFAULT_VEHICLE_ENABLED
        self.roi_polygon     = parse_roi(roi)
        self.line_pts        = parse_line(line)
        self.line_in_dir     = parse_line_in_dir(line_in_dir)
        raw_line_enabled     = bool(line_enabled) if line_enabled is not None else DEFAULT_LINE_ENABLED
        self.line_enabled    = raw_line_enabled if self.vehicle_enabled else False

        self.face_enabled  = bool(face_enabled) if face_enabled is not None else DEFAULT_FACE_ENABLED
        self.face_memory   = {}

        self.person_enabled = bool(person_enabled) if person_enabled is not None else DEFAULT_PERSON_ENABLED
        self.person_lines   = parse_multi_lines(person_lines)
        self.person_rois    = parse_multi_rois(person_rois)

        self.save_image_vehicle = (
            bool(save_image_vehicle) if save_image_vehicle is not None else DEFAULT_SAVE_IMAGE_VEHICLE
        )
        self.save_image_face = (
            bool(save_image_face) if save_image_face is not None else DEFAULT_SAVE_IMAGE_FACE
        )

        self._log_config()

        self.connected    = False
        self.reconnecting = False
        self.dead         = False

        self.last_frame_time  = 0
        self.bad              = 0
        self.max_bad          = 10
        self.last_time        = 0
        self.last_detect_time = time.time()
        self.last_face_time   = time.time()
        self.last_person_time = time.time()

        self.tracker        = ObjectTracker()
        self._result_q      = Queue(maxsize=2)
        self._face_result_q = Queue(maxsize=2)
        self._roi_send_times = {}   # cooldown per ROI index, level kamera (tidak hilang saat obj_id berganti)

        self.cap = self._open_capture()
        
        self.latest_frame = None
        self.latest_ret = False
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print(f"[CAMERA START] {cid} -> {name} | vehicle={self.vehicle_enabled} | face={self.face_enabled} | person={self.person_enabled}")

    # ------------------------------------------------------------------ stream

    def _build_stream_url(self, url: str) -> str:
        """
        Debug mode  → pakai path file langsung
        Produksi    → rtsp:// + url asli
        Parameter FFmpeg sudah diset via OPENCV_FFMPEG_CAPTURE_OPTIONS (env),
        tidak perlu ditempel ke URL supaya lebih bersih & kompatibel.
        """
        if DEBUG_MODE:
            return url
        return f"rtsp://{url}"

    def _open_capture(self):
        try:
            # options untuk av.open
            options = {
                'rtsp_transport': 'tcp',
                'stimeout': '5000000',      # 5 seconds socket timeout
                'max_delay': '500000',
                'flags': 'low_delay'
            }
            container = av.open(self.stream_source, options=options)
            container.streams.video[0].thread_type = "AUTO"  # Multi-thread H264 decode
            self.connected = True
            self.dead      = False
            print(f"[RTSP CONNECTED PyAV] {self.cid} -> {self.name_camera}")
            return container
        except Exception as e:
            print(f"[RTSP FAILED PyAV] {self.cid} -> {self.name_camera} ({e})")
            self.connected = False
            self.dead = True
            return None

    # ------------------------------------------------------------------ config

    def _log_config(self):
        print(
            f"[CAMERA CONFIG] {self.cid} "
            f"| vehicle={self.vehicle_enabled} "
            f"| face={self.face_enabled} "
            f"| person={self.person_enabled} "
            f"| save_img_vehicle={self.save_image_vehicle} "
            f"| save_img_face={self.save_image_face} "
            f"| line_enabled={self.line_enabled} "
            f"| line={self.line_pts} "
            f"| line_in_dir={self.line_in_dir} "
            f"| person_lines={self.person_lines} "
            f"| person_rois={self.person_rois}"
        )

    def stop(self):
        self.running = False
        self.cap.release()
        print(f"[CAMERA STOP] {self.cid}")

    def update_config(self, roi=None, line=None, line_in_dir=None,
                      line_enabled=None, vehicle_enabled=None, face_enabled=None,
                      save_image_vehicle=None, save_image_face=None,
                      person_enabled=None, person_lines=None, person_rois=None):
        changed = False

        for attr, new_val, default in [
            ("vehicle_enabled",    vehicle_enabled,    DEFAULT_VEHICLE_ENABLED),
            ("face_enabled",       face_enabled,       DEFAULT_FACE_ENABLED),
            ("person_enabled",     person_enabled,     DEFAULT_PERSON_ENABLED),
            ("save_image_vehicle", save_image_vehicle, DEFAULT_SAVE_IMAGE_VEHICLE),
            ("save_image_face",    save_image_face,    DEFAULT_SAVE_IMAGE_FACE),
        ]:
            val = bool(new_val) if new_val is not None else default
            if val != getattr(self, attr):
                setattr(self, attr, val)
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

        raw_le       = bool(line_enabled) if line_enabled is not None else DEFAULT_LINE_ENABLED
        effective_le = raw_le if self.vehicle_enabled else False
        if effective_le != self.line_enabled:
            self.line_enabled = effective_le
            changed = True

        new_pl = parse_multi_lines(person_lines)
        if new_pl != self.person_lines:
            self.person_lines = new_pl
            changed = True

        new_pr = parse_multi_rois(person_rois)
        if new_pr != self.person_rois:
            self.person_rois = new_pr
            changed = True

        if changed:
            self._log_config()

    # ------------------------------------------------------------------ helpers

    def is_inside_roi(self, cx, cy, roi_scaled) -> bool:
        return cv2.pointPolygonTest(roi_scaled, (float(cx), float(cy)), False) >= 0

    def resize_frame(self, frame):
        h, w  = frame.shape[:2]
        scale = min(1.0, TARGET_MAX_WIDTH / w)
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return frame

    def resize_for_face(self, frame):
        h, w    = frame.shape[:2]
        scale   = min(1.0, TARGET_MAX_WIDTH / w)
        resized = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
        if resized.shape[1] < 640:
            up      = 640 / resized.shape[1]
            resized = cv2.resize(resized, None, fx=up, fy=up)
            scale   = scale / up
        return resized, 1.0 / scale

    def adaptive_fps(self):
        idle = (
            time.time() - self.last_detect_time > IDLE_TIMEOUT and
            time.time() - self.last_face_time   > IDLE_TIMEOUT
        )
        return 1.0 / IDLE_FPS if idle else 1.0 / FRAME_FPS

    def cleanup_face_memory(self):
        now   = time.time()
        stale = [k for k, v in self.face_memory.items() if now - v[0] > 60]
        for k in stale:
            del self.face_memory[k]

    # ------------------------------------------------------------------ frame read

    def _capture_loop(self):
        while self.running:
            if not self.connected or self.reconnecting or self.cap is None:
                time.sleep(0.01)
                continue
            
            try:
                video_stream = self.cap.streams.video[0]
                for frame in self.cap.decode(video_stream):
                    if not self.running or self.reconnecting:
                        break
                    
                    # Simpan frame mentah (PyAV frame), hindari konversi warna 
                    # di thread ini agar CPU tidak 100% saat pakai 16 kamera.
                    self.latest_av_frame = frame
                    self.latest_ret = True
                
                # Jika keluar dari loop tapi masih running, stream terputus
                if self.running and not self.reconnecting:
                    self.latest_ret = False
                    self.connected = False
                    time.sleep(0.1)

            except Exception as e:
                print(f"[_capture_loop ERROR] {self.cid}: {e}")
                self.latest_ret = False
                self.connected = False
                time.sleep(0.5)

    def _read_latest_frame(self):
        """
        Ambil frame terbaru dari buffer memory yang di-update oleh thread terpisah.
        Melakukan konversi ke Numpy (BGR) HANYA saat diminta, sehingga frame 
        yang ter-skip tidak membebani CPU sama sekali.
        """
        if not getattr(self, 'latest_ret', False) or getattr(self, 'latest_av_frame', None) is None:
            return False, None
            
        try:
            img = self.latest_av_frame.to_ndarray(format='bgr24')
            return True, img
        except Exception:
            return False, None

    # ------------------------------------------------------------------ vehicle inference

    def _submit_for_inference(self, frame_infer):
        while not self._result_q.empty():
            try:
                self._result_q.get_nowait()
            except Empty:
                break
        try:
            _infer_input_queue.put_nowait((self.cid, frame_infer, self._result_q))
            return True
        except Exception:
            return False

    def _get_inference_result(self, timeout=0.5):
        try:
            status, result = self._result_q.get(timeout=timeout)
            if status == "ok":
                return result
        except Empty:
            pass
        return []

    # ------------------------------------------------------------------ vehicle processing

    def _process_vehicle_object(self, obj, frame_original, view,
                                orig_w, orig_h, scale_x, scale_y,
                                bbox_orig, roi_scaled, line_scaled):

        x1, y1, x2, y2     = obj["bbox"]
        ox1, oy1, ox2, oy2 = bbox_orig
        conf                = obj["conf"]
        cls_name            = obj["cls_name"]
        obj_id              = obj["obj_id"]

        cfg    = CLASS_CONFIG[cls_name]
        color  = cfg["color"]
        ts_iso = iso_timestamp()
        area   = (ox2 - ox1) * (oy2 - oy1)
        width  = ox2 - ox1
        height = oy2 - oy1

        cv2.rectangle(view, (x1, y1), (x2, y2), color, WIDTH_LINE)
        cv2.putText(
            view, f"#{obj_id} {cls_name} {conf:.2f} {area} {width}x{height}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), WIDTH_LINE,
        )

        cx, cy = bbox_center(x1, y1, x2, y2)

        if not self.vehicle_enabled or not self.line_enabled:
            return False

        track_data = self.tracker.tracks.get(obj_id)
        if track_data is None:
            return False
        track_data.setdefault("last_cross_dir", None)

        if "first_check" not in track_data:
            track_data["first_check"] = True
            (lx1, ly1), (lx2, ly2) = line_scaled
            roi_center_x = np.mean(roi_scaled[:, 0])
            roi_center_y = np.mean(roi_scaled[:, 1])
            roi_side = point_side_of_line(roi_center_x, roi_center_y, lx1, ly1, lx2, ly2)
            veh_side = point_side_of_line(cx, y2, lx1, ly1, lx2, ly2)
            # Jika sisi ROI dan sisi kendaraan berbeda tanda (+/-), berarti kendaraan sudah melewati garis
            if roi_side * veh_side < 0:
                track_data["line_crossed_first"] = True

        # 1. Simpan/perbarui cache capture saat kendaraan berada di posisi TERDALAM di ROI (hanya jika ROI dikunjungi sebelum Line)
        dist_roi = cv2.pointPolygonTest(roi_scaled, (float(cx), float(cy)), True)
        if dist_roi >= 0:
            if not track_data.get("line_crossed_first", False):
                track_data["roi_visited"] = True
                if dist_roi > track_data.get("best_roi_dist", -1.0):
                    cx1, cy1, cx2, cy2 = expand_crop_bbox(ox1, oy1, ox2, oy2, orig_w, orig_h, CROP_PADDING)
                    crop_original      = frame_original[cy1:cy2, cx1:cx2]
                    if crop_original.size > 0:
                        crop_save          = prepare_for_save(crop_original)
                        frame_save         = prepare_for_save(frame_original)

                        _, crop_jpg       = cv2.imencode(".jpg", crop_save)
                        _, frame_jpg      = cv2.imencode(".jpg", frame_save)

                        track_data["best_roi_dist"] = dist_roi
                        track_data["roi_capture"]   = {
                            "crop_bytes": crop_jpg.tobytes(),
                            "frame_clean_bytes": frame_jpg.tobytes(),
                            "bbox_orig": (ox1, oy1, ox2, oy2),
                            "conf": conf,
                            "ts_iso": ts_iso,
                        }

        # Kunci single webhook per track ID kendaraan
        if track_data.get("line_sent", False):
            return False

        # 2. Cek apakah titik tengah kendaraan baru saja menyeberangi garis pada frame ini
        direction = is_just_crossed_line(track_data["history"], line_scaled, self.line_in_dir, margin=2.0)
        if direction is not None:
            # Jika melintasi garis sebelum pernah mengunjungi ROI, tandai line_crossed_first
            if not track_data.get("roi_visited", False):
                track_data["line_crossed_first"] = True
                return False

        if direction is None or direction == track_data["last_cross_dir"]:
            return False

        # Tolak jika pernah melintasi garis duluan sebelum ROI
        if track_data.get("line_crossed_first", False):
            return False

        # Wajib pernah mengunjungi ROI sebelum melintasi garis
        if not track_data.get("roi_visited", False):
            return False

        # 3. Syarat Event: Kendaraan harus pernah di-capture saat berada di dalam ROI
        roi_cap = track_data.get("roi_capture")
        if roi_cap is None:
            return False

        track_data["last_cross_dir"] = direction
        track_data["line_sent"]       = True
        print(f"[VEHICLE] ID={obj_id} {cls_name} dir={direction} | cam={self.cid}")

        crop_bytes        = roi_cap["crop_bytes"]
        frame_clean_bytes = roi_cap["frame_clean_bytes"]
        cap_ox1, cap_oy1, cap_ox2, cap_oy2 = roi_cap["bbox_orig"]
        cap_conf          = roi_cap["conf"]
        cap_ts            = roi_cap["ts_iso"]

        crop_name  = iso_name("crop",  cls_name, cap_ts, obj_id, direction)
        frame_name = iso_name("frame", cls_name, cap_ts, obj_id, direction)

        if self.save_image_vehicle:
            try:
                with open(os.path.join(DETECT_FOLDER,    crop_name),  "wb") as f: f.write(crop_bytes)
                with open(os.path.join(FRAME_FOLDER_VEH, frame_name), "wb") as f: f.write(frame_clean_bytes)
                enforce_limit(DETECT_FOLDER)
                enforce_limit(FRAME_FOLDER_VEH)
            except Exception as e:
                print(f"[SAVE ERROR VEHICLE] {e}")
        else:
            print(f"[SKIP SAVE VEHICLE] {frame_name}")

        if WEBHOOK_URL:
            try:
                webhook_queue.put_nowait((
                    crop_bytes, frame_clean_bytes, crop_name, frame_name,
                    self.name_camera, cap_ts,
                    (cap_ox1, cap_oy1, cap_ox2, cap_oy2), cap_conf, cls_name,
                    self.cid, self.client_id,
                    obj_id, obj["is_new"], direction,
                ))
            except Exception:
                print("[QUEUE FULL] vehicle webhook queue penuh")

        return True

    # ------------------------------------------------------------------ person processing

    def _process_person_object(self, obj, frame_original, view,
                               orig_w, orig_h, scale_x, scale_y,
                               bbox_orig, roi_scaled, person_lines_scaled,
                               person_rois_scaled=None):
        if not self.person_enabled or (not person_lines_scaled and not person_rois_scaled):
            return False

        x1, y1, x2, y2     = obj["bbox"]
        ox1, oy1, ox2, oy2 = bbox_orig
        conf                = obj["conf"]
        cls_name            = obj["cls_name"]
        obj_id              = obj["obj_id"]

        cfg    = CLASS_CONFIG[cls_name]
        color  = cfg["color"]
        ts_iso = iso_timestamp()
        area   = (ox2 - ox1) * (oy2 - oy1)
        width  = ox2 - ox1
        height = oy2 - oy1

        cv2.rectangle(view, (x1, y1), (x2, y2), color, WIDTH_LINE)
        cv2.putText(
            view, f"#{obj_id} {cls_name} {conf:.2f} {area} {width}x{height}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, WIDTH_LINE,
        )

        track_data = self.tracker.tracks.get(obj_id)
        if track_data is None:
            return False
        track_data.setdefault("last_cross_dirs", {})
        track_data.setdefault("last_roi_states", {})

        # Pemicu line dan ROI diproses masing-masing secara mandiri
        has_event = False

        # --- A. PERSON LINE DETECTION ---
        if person_lines_scaled:
            for item in person_lines_scaled:
                line_pts = item["line"]
                line_in_dir = item["line_in_dir"]
                idx_line = item["index"]

                dir_cross = check_line_cross(track_data["history"], line_pts, line_in_dir)
                if dir_cross is not None:
                    last_dir = track_data["last_cross_dirs"].get(idx_line)
                    if dir_cross != last_dir:
                        track_data["last_cross_dirs"][idx_line] = dir_cross
                        
                        # Kirim event line detection langsung
                        print(f"[PERSON LINE] ID={obj_id} dir={dir_cross} | line_idx={idx_line} | cam={self.cid}")
                        crop_name = iso_name("crop", f"{cls_name}_line{idx_line}", ts_iso, obj_id, dir_cross)
                        frame_name = iso_name("frame", f"{cls_name}_line{idx_line}", ts_iso, obj_id, dir_cross)
                        self._trigger_person_webhook(
                            ox1, oy1, ox2, oy2, orig_w, orig_h, frame_original,
                            crop_name, frame_name, ts_iso, conf, cls_name, obj_id, obj["is_new"],
                            dir_cross, {"index": idx_line, "name": item.get("name", idx_line)}
                        )
                        has_event = True
                        break

        # --- B. PERSON ROI DETECTION ---
        if person_rois_scaled:
            now_time = time.time()
            cx_person = (x1 + x2) // 2
            # Gunakan HANYA titik kaki (cx, y2) agar tubuh bagian atas di luar ROI tidak memicu alarm
            foot_pt = (float(cx_person), float(y2))

            for item in person_rois_scaled:
                roi_pts = np.array(item["roi"], dtype=np.int32)
                idx_roi = item["index"]

                # Cek apakah kaki berada di dalam polygon ROI
                is_inside = cv2.pointPolygonTest(roi_pts, foot_pt, False) >= 0
                was_inside = track_data["last_roi_states"].get(idx_roi, False)

                if is_inside:
                    track_data["last_roi_states"][idx_roi] = True

                    # Cooldown diambil dari level kamera agar tidak hilang saat obj_id berganti
                    last_send = self._roi_send_times.get(idx_roi, 0)
                    # Kirim terus-menerus selama di dalam ROI (cooldown 5 detik per event)
                    if now_time - last_send >= 5.0:
                        self._roi_send_times[idx_roi] = now_time
                        
                        # Kirim event ROI detection langsung
                        print(f"[PERSON ROI] ID={obj_id} roi_idx={idx_roi} | cam={self.cid}")
                        crop_name = iso_name("crop", f"{cls_name}_roi{idx_roi}", ts_iso, obj_id, "IN")
                        frame_name = iso_name("frame", f"{cls_name}_roi{idx_roi}", ts_iso, obj_id, "IN")
                        self._trigger_person_webhook(
                            ox1, oy1, ox2, oy2, orig_w, orig_h, frame_original,
                            crop_name, frame_name, ts_iso, conf, cls_name, obj_id, obj["is_new"],
                            None, {"index": idx_roi, "name": item.get("name", idx_roi)}, is_roi=True
                        )
                        has_event = True
                        break
                else:
                    # Reset state masuk/keluar per-track
                    track_data["last_roi_states"][idx_roi] = False

        if has_event:
            self.last_person_time = time.time()
            return True
        return False

    def _trigger_person_webhook(self, ox1, oy1, ox2, oy2, orig_w, orig_h, frame_original,
                                crop_name, frame_name, ts_iso, conf, cls_name, obj_id, is_new,
                                direction, info_val, is_roi=False):
        cx1, cy1, cx2, cy2 = expand_crop_bbox(ox1, oy1, ox2, oy2, orig_w, orig_h, CROP_PADDING)
        crop_original      = frame_original[cy1:cy2, cx1:cx2]
        if crop_original.size == 0:
            return False
        crop_save          = prepare_for_save(crop_original)
        frame_save         = prepare_for_save(frame_original)

        _, crop_jpg       = cv2.imencode(".jpg", crop_save)
        crop_bytes        = crop_jpg.tobytes()
        _, frame_jpg      = cv2.imencode(".jpg", frame_save)
        frame_clean_bytes = frame_jpg.tobytes()

        if self.save_image_vehicle:
            try:
                with open(os.path.join(DETECT_FOLDER,    crop_name),  "wb") as f: f.write(crop_bytes)
                with open(os.path.join(FRAME_FOLDER_VEH, frame_name), "wb") as f: f.write(frame_clean_bytes)
                enforce_limit(DETECT_FOLDER)
                enforce_limit(FRAME_FOLDER_VEH)
            except Exception as e:
                print(f"[SAVE ERROR PERSON] {e}")

        if WEBHOOK_URL:
            try:
                if not is_roi:
                    webhook_queue.put_nowait((
                        crop_bytes, frame_clean_bytes, crop_name, frame_name,
                        self.name_camera, ts_iso,
                        (ox1, oy1, ox2, oy2), conf, cls_name,
                        self.cid, self.client_id,
                        obj_id, is_new, direction, info_val
                    ))
                else:
                    webhook_queue.put_nowait((
                        crop_bytes, frame_clean_bytes, crop_name, frame_name,
                        self.name_camera, ts_iso,
                        (ox1, oy1, ox2, oy2), conf, cls_name,
                        self.cid, self.client_id,
                        obj_id, is_new, None, info_val, True
                    ))
            except Exception:
                print("[QUEUE FULL] person webhook queue penuh")

        return True

    # ------------------------------------------------------------------ face inference

    def _submit_for_face_inference(self, frame_resized, inv_scale) -> bool:
        while not self._face_result_q.empty():
            try:
                self._face_result_q.get_nowait()
            except Empty:
                break
        try:
            _face_infer_queue.put_nowait((self.cid, frame_resized, inv_scale, self._face_result_q))
            return True
        except Exception:
            return False

    def _get_face_result(self, timeout=0.5):
        try:
            status, faces, inv_scale = self._face_result_q.get(timeout=timeout)
            if status == "ok":
                return faces, inv_scale
        except Empty:
            pass
        return None, 1.0

    # ------------------------------------------------------------------ face detection

    def _run_face_detection(self, frame_original, view):
        if not self.face_enabled:
            return

        resized, inv_scale = self.resize_for_face(frame_original)
        if not self._submit_for_face_inference(resized, inv_scale):
            return

        faces, inv_scale = self._get_face_result(timeout=0.5)
        if faces is None:
            return

        orig_h, orig_w = frame_original.shape[:2]

        for f in faces:
            score = float(f[14])
            if score < SCORE_THRESHOLD:
                continue

            x, y, fw, fh = f[:4].astype(int)
            x  = int(x  * inv_scale); y  = int(y  * inv_scale)
            fw = int(fw * inv_scale); fh = int(fh * inv_scale)

            if fw < MIN_SIZE_CAPTURE or fw > MAX_FACE_SIZE:
                continue
            if not is_valid_face(f, x, y, fw, fh, inv_scale):
                continue

            x  = max(0, x);  y  = max(0, y)
            fw = min(fw, orig_w - x); fh = min(fh, orig_h - y)
            cx = x + fw // 2; cy = y + fh // 2

            bucket = (cx // FACE_BUCKET_SIZE, cy // FACE_BUCKET_SIZE)
            now    = time.time()

            if bucket in self.face_memory:
                last_time, last_pos = self.face_memory[bucket]
                if now - last_time < FACE_COOLDOWN:
                    continue
                if math.hypot(cx - last_pos[0], cy - last_pos[1]) < FACE_MOVE_THRESHOLD:
                    continue

            self.face_memory[bucket] = (now, (cx, cy))
            self.last_face_time = time.time()

            fx1, fy1, fx2, fy2 = expand_crop_bbox(x, y, x + fw, y + fh, orig_w, orig_h, FACE_CROP_PADDING)
            face_img = frame_original[fy1:fy2, fx1:fx2]

            if face_img.size == 0 or calc_sharpness(face_img) < BLUR_THRESHOLD:
                continue

            vscale = view.shape[1] / orig_w
            cv2.rectangle(view,
                          (int(x * vscale), int(y * vscale)),
                          (int((x + fw) * vscale), int((y + fh) * vscale)),
                          (0, 255, 0), WIDTH_LINE)
            cv2.putText(view, f"face {score:.2f}",
                        (int(x * vscale), max(0, int(y * vscale) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), WIDTH_LINE)

            ts_iso     = iso_timestamp()
            face_name  = iso_name("face",  "face", ts_iso)
            frame_name = iso_name("frame", "face", ts_iso)

            fb1 = cv2.imencode(".jpg", face_img)[1].tobytes()
            fb2 = cv2.imencode(".jpg", frame_original)[1].tobytes()

            if self.save_image_face:
                try:
                    with open(os.path.join(FACE_CROP_FOLDER,  face_name),  "wb") as fp: fp.write(fb1)
                    with open(os.path.join(FACE_FRAME_FOLDER, frame_name), "wb") as fp: fp.write(fb2)
                    enforce_limit(FACE_CROP_FOLDER)
                    enforce_limit(FACE_FRAME_FOLDER)
                except Exception as e:
                    print(f"[SAVE ERROR FACE] {e}")
            else:
                print(f"[SKIP SAVE FACE] {frame_name}")

            if WEBHOOK_URL:
                try:
                    face_webhook_queue.put_nowait((
                        fb1, fb2, face_name, frame_name,
                        self.name_camera, ts_iso,
                        (x, y, fw, fh), score,
                        self.cid, self.client_id,
                    ))
                except Exception:
                    print("[QUEUE FULL] face webhook queue penuh")

            print(f"[{self.cid}] FACE SAVED {face_name} | conf={score:.3f}")

    # ------------------------------------------------------------------ main loop

    def run(self):
        while self.running:
            interval = self.adaptive_fps()
            if time.time() - self.last_time < interval:
                time.sleep(0.005)
                continue
            self.last_time = time.time()

            # ── Ambil frame terbaru, flush buffer lama ─────────────────
            ret, frame = self._read_latest_frame()

            if ret and frame is not None:
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
            scale_x        = orig_w / inf_w
            scale_y        = orig_h / inf_h

            roi_scaled  = scale_roi(self.roi_polygon, inf_w, inf_h)
            line_scaled = scale_line_pts(self.line_pts, inf_w, inf_h)
            person_lines_scaled = []
            if self.person_enabled and self.person_lines:
                for item in self.person_lines:
                    scaled_pts = scale_line_pts(item["line"], inf_w, inf_h)
                    person_lines_scaled.append({
                        "line": scaled_pts,
                        "line_in_dir": item["line_in_dir"],
                        "index": item["index"],
                        "name": item.get("name")
                    })

            person_rois_scaled = []
            if self.person_enabled and self.person_rois:
                for item in self.person_rois:
                    if item.get("roi_enabled", True):
                        scaled_pts = scale_roi(np.array(item["roi"], dtype=np.int32), inf_w, inf_h)
                        person_rois_scaled.append({
                            "roi": scaled_pts,
                            "index": item["index"],
                            "name": item.get("name")
                        })

            # ── YOLO Inference (Vehicle & Person) ──────────────────────
            if self.vehicle_enabled or self.person_enabled:
                if self._submit_for_inference(frame_infer):
                    raw_dets = self._get_inference_result(timeout=0.5)[:20]
                    tracked  = self.tracker.update(raw_dets)

                    if tracked:
                        self.last_detect_time = time.time()

                    bbox_orig_map = {
                        (x1, y1, x2, y2, cls_name): (
                            int(x1 * scale_x), int(y1 * scale_y),
                            int(x2 * scale_x), int(y2 * scale_y),
                        )
                        for (x1, y1, x2, y2, conf, cls_name) in raw_dets
                    }

                    for obj in tracked:
                        bx1, by1, bx2, by2 = obj["bbox"]
                        cls_name = obj["cls_name"]
                        bbox_orig = bbox_orig_map.get(
                            (bx1, by1, bx2, by2, cls_name),
                            (int(bx1*scale_x), int(by1*scale_y),
                             int(bx2*scale_x), int(by2*scale_y)),
                        )
                        if cls_name == "person":
                            self._process_person_object(
                                obj, frame_original, view,
                                orig_w, orig_h, scale_x, scale_y,
                                bbox_orig, roi_scaled, person_lines_scaled,
                                person_rois_scaled,
                            )
                        else:
                            self._process_vehicle_object(
                                obj, frame_original, view,
                                orig_w, orig_h, scale_x, scale_y,
                                bbox_orig, roi_scaled, line_scaled,
                            )

            # ── Face detection ─────────────────────────────────────────
            self._run_face_detection(frame_original, view)
            self.cleanup_face_memory()

            # ── Overlay ────────────────────────────────────────────────
            active = self.tracker.active_count()
            y_off  = 20
            for cls_n, cnt in active.items():
                cv2.putText(view, f"{cls_n}: {cnt}", (8, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), WIDTH_LINE)
                y_off += 20

            if self.vehicle_enabled:
                cv2.polylines(view, [roi_scaled], True, (255, 255, 0), WIDTH_LINE)
                if self.line_enabled:
                    (lx1, ly1), (lx2, ly2) = line_scaled
                    cv2.line(view, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)
                    dx, dy  = lx2 - lx1, ly2 - ly1
                    length  = math.hypot(dx, dy) or 1
                    offset  = 22
                    mid_x   = (lx1 + lx2) // 2
                    mid_y   = (ly1 + ly2) // 2
                    nx, ny  = int(-dy / length * offset), int(dx / length * offset)
                    FONT, FS, TH = cv2.FONT_HERSHEY_SIMPLEX, 0.45, WIDTH_LINE
                    (wa, ha), _ = cv2.getTextSize("A : IN",  FONT, FS, TH)
                    (wb, hb), _ = cv2.getTextSize("B : OUT", FONT, FS, TH)
                    ax, ay = mid_x + nx, mid_y + ny + ha // 2
                    bx, by = mid_x - nx, mid_y - ny + hb // 2
                    if nx < 0:  ax -= wa
                    if nx > 0:  bx -= wb
                    elif nx == 0:
                        ax -= wa // 2; bx -= wb // 2
                    cv2.putText(view, "A : IN",  (ax, ay), FONT, FS, (0, 255, 0), TH)
                    cv2.putText(view, "B : OUT", (bx, by), FONT, FS, (0, 0, 255), TH)
                else:
                    cv2.putText(view, "LINE: OFF", (8, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), WIDTH_LINE)
                    y_off += 20
            else:
                cv2.putText(view, "VEHICLE: OFF", (8, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), WIDTH_LINE)
                y_off += 20

            # ── Overlay: Person Multi-Lines ───────────────────────────────
            if self.person_enabled and person_lines_scaled:
                for item in person_lines_scaled:
                    (lx1, ly1), (lx2, ly2) = item["line"]
                    idx_line = item["index"]
                    line_in_dir = item["line_in_dir"]

                    cv2.line(view, (lx1, ly1), (lx2, ly2), (255, 0, 255), 1)

                    dx, dy  = lx2 - lx1, ly2 - ly1
                    length  = math.hypot(dx, dy) or 1
                    offset  = 22
                    mid_x   = (lx1 + lx2) // 2
                    mid_y   = (ly1 + ly2) // 2
                    nx, ny  = int(-dy / length * offset), int(dx / length * offset)
                    FONT, FS, TH = cv2.FONT_HERSHEY_SIMPLEX, 0.45, WIDTH_LINE

                    label_a = f"L{idx_line} A:IN" if line_in_dir == "A" else f"L{idx_line} A:OUT"
                    label_b = f"L{idx_line} B:OUT" if line_in_dir == "A" else f"L{idx_line} B:IN"

                    (wa, ha), _ = cv2.getTextSize(label_a,  FONT, FS, TH)
                    (wb, hb), _ = cv2.getTextSize(label_b, FONT, FS, TH)
                    ax, ay = mid_x + nx, mid_y + ny + ha // 2
                    bx, by = mid_x - nx, mid_y - ny + hb // 2
                    if nx < 0:  ax -= wa
                    if nx > 0:  bx -= wb
                    elif nx == 0:
                        ax -= wa // 2; bx -= wb // 2
                    cv2.putText(view, label_a,  (ax, ay), FONT, FS, (0, 255, 0), TH)
                    cv2.putText(view, label_b, (bx, by), FONT, FS, (0, 0, 255), TH)

            # ── Overlay: Person Multi-ROIs ────────────────────────────────
            if self.person_enabled and person_rois_scaled:
                for item in person_rois_scaled:
                    roi_pts = item["roi"]
                    idx_roi = item["index"]
                    cv2.polylines(view, [roi_pts], True, (255, 0, 255), WIDTH_LINE)
                    if len(roi_pts) > 0:
                        cv2.putText(view, f"ROI {idx_roi}", (int(roi_pts[0][0]), int(roi_pts[0][1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), WIDTH_LINE)

            if not self.face_enabled:
                cv2.putText(view, "FACE: OFF", (8, y_off + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), WIDTH_LINE)

            # Device mode indicator
            mode_label = {
                "cuda":         "GPU: FULL CUDA",
                "cuda_partial": "GPU: ORT CUDA | CV CPU",
                "cpu":          "CPU MODE",
            }.get(DEVICE, DEVICE.upper())
            cv2.putText(view, mode_label, (8, view.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (0, 200, 0) if "cuda" in DEVICE else (100, 100, 100), WIDTH_LINE)

            siv_color = (0, 200, 0) if self.save_image_vehicle else (80, 80, 80)
            sif_color = (0, 200, 0) if self.save_image_face    else (80, 80, 80)
            cv2.putText(view, "SAVE VEH: ON"  if self.save_image_vehicle else "SAVE VEH: OFF",
                        (8, view.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, siv_color, WIDTH_LINE)
            cv2.putText(view, "SAVE FACE: ON" if self.save_image_face    else "SAVE FACE: OFF",
                        (8, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sif_color, WIDTH_LINE)

            if self.running:
                with preview_lock:
                    preview_frames[self.cid] = view

    # ------------------------------------------------------------------ reconnect

    def _reconnect(self):
        print(f"[RECONNECT] {self.cid} -> {self.name_camera}")
        self.reconnecting = True
        if self.cap is not None:
            try:
                self.cap.close()
            except Exception:
                pass
        time.sleep(2)
        self.cap = self._open_capture()
        self.bad     = 0
        self.tracker = ObjectTracker()
        self.reconnecting = False
        print(f"[TRACKER RESET] {self.cid}")


# ================= LOAD CAMERAS =================

def load_cameras():
    if DEBUG_MODE:
        print("[DEBUG MODE] Load video dari folder lokal")
        videos = []
        for file in os.listdir(DEBUG_VIDEO_DIR):
            if file.lower().endswith(".mp4"):
                videos.append({
                    "cctv_id": file, "client_id": "debug",
                    "stream_url": os.path.join(DEBUG_VIDEO_DIR, file),
                    "name": file,
                    "roi": None, "line": None, "line_in_dir": None,
                    "line_enabled": None, "vehicle_enabled": None,
                    "face_enabled": None,
                    "save_image_vehicle": None, "save_image_face": None,
                    "person_enabled": True,
                    "person_lines": [
                        {"line": [[50, 100], [300, 300]], "line_in_dir": "A", "name": "Garis_A"},
                        {"line": [[320, 300], [580, 100]], "line_in_dir": "B", "name": "Garis_B"}
                    ],
                    "person_rois": [
                        {"roi": [[50, 50], [200, 50], [200, 200], [50, 200]], "roi_enabled": True, "name": "ROI_Orang_A"},
                        {"roi": [[350, 50], [500, 50], [500, 200], [350, 200]], "roi_enabled": True, "name": "ROI_Orang_B"}
                    ]
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


# ================= MONITORING =================

def print_camera_status():
    with camera_lock:
        total        = len(active_cameras)
        connected    = sum(1 for w in active_cameras.values() if w.connected)
        reconnecting = sum(1 for w in active_cameras.values() if w.reconnecting)
        dead         = sum(1 for w in active_cameras.values() if w.dead)
        save_veh_on  = sum(1 for w in active_cameras.values() if w.save_image_vehicle)
        save_face_on = sum(1 for w in active_cameras.values() if w.save_image_face)

    print("\n========== CAMERA STATUS ==========")
    print(f"Total Camera      : {total}")
    print(f"Connected         : {connected}")
    print(f"Reconnecting      : {reconnecting}")
    print(f"Dead              : {dead}")
    print(f"Save Img Vehicle  : {save_veh_on}/{total} ON")
    print(f"Save Img Face     : {save_face_on}/{total} ON")
    print(f"Device Mode       : {DEVICE.upper()} (device_id={CUDA_DEVICE_ID})")
    print(f"ORT CUDA          : {_ORT_USE_CUDA}")
    print(f"OpenCV CUDA       : {_OCV_USE_CUDA}")
    print(f"GRAB_SKIP_COUNT   : {GRAB_SKIP_COUNT}")
    print(f"Vehicle Wbhk Queue: {webhook_queue.qsize()}")
    print(f"Face Wbhk Queue   : {face_webhook_queue.qsize()}")
    print(f"Vehicle Infer Q   : {_infer_input_queue.qsize()}")
    print(f"Face Infer Q      : {_face_infer_queue.qsize()}")
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
            dead_active_ids = {cid for cid, w in active_cameras.items() if w.dead or not w.connected}
            new_ids     = (api_ids - active_ids) | dead_active_ids
            removed_ids = active_ids - api_ids

            for c in cams:
                cid = c["cctv_id"]
                if cid in new_ids:
                    try:
                        if cid in active_cameras:
                            try:
                                active_cameras[cid].stop()
                            except Exception:
                                pass
                            del active_cameras[cid]

                        w = CameraWorker(
                            cid, c["client_id"], c["stream_url"], c.get("name", "unknown"),
                            roi=c.get("roi"), line=c.get("line"),
                            line_in_dir=c.get("line_in_dir"), line_enabled=c.get("line_enabled"),
                            vehicle_enabled=c.get("vehicle_enabled"), face_enabled=c.get("face_enabled"),
                            save_image_vehicle=c.get("save_image_vehicle"),
                            save_image_face=c.get("save_image_face"),
                            person_enabled=c.get("person_enabled"),
                            person_lines=c.get("person_lines"),
                            person_rois=c.get("person_rois"),
                        )
                        Thread(target=w.run, daemon=True).start()
                        active_cameras[cid] = w
                        print(f"[NEW/RECONNECTED CAMERA] {cid} -> {c.get('name','?')}")
                    except Exception as e:
                        print(f"[FAILED START] {cid} -> {e}")
                else:
                    active_cameras[cid].update_config(
                        roi=c.get("roi"), line=c.get("line"),
                        line_in_dir=c.get("line_in_dir"), line_enabled=c.get("line_enabled"),
                        vehicle_enabled=c.get("vehicle_enabled"), face_enabled=c.get("face_enabled"),
                        save_image_vehicle=c.get("save_image_vehicle"),
                        save_image_face=c.get("save_image_face"),
                        person_enabled=c.get("person_enabled"),
                        person_lines=c.get("person_lines"),
                        person_rois=c.get("person_rois"),
                    )

            for rid in removed_ids:
                try:
                    active_cameras[rid].stop()
                    del active_cameras[rid]
                    print(f"[CAMERA REMOVED] {rid}")
                except Exception:
                    pass

            with preview_lock:
                for cid in [c for c in list(preview_frames) if c not in active_cameras]:
                    del preview_frames[cid]

            print(f"[MANAGER] API: {len(api_ids)} kamera | Aktif: {len(active_cameras)} worker")

        time.sleep(CAMERA_REFRESH_INTERVAL)


Thread(target=camera_manager,    daemon=True).start()
Thread(target=monitoring_worker, daemon=True).start()


# ================= DISPLAY GRID =================

if ENABLE_VIEW:
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        with preview_lock:
            snapshot = list(preview_frames.values())

        if not snapshot:
            blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "No active cameras",
                        (DISPLAY_WIDTH // 2 - 140, DISPLAY_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.imshow("Detection", blank)
            if cv2.waitKey(200) == ord("q"):
                break
            continue

        total = len(snapshot)
        cols  = math.ceil(math.sqrt(total))
        rows  = math.ceil(total / cols)
        tw    = max(1, DISPLAY_WIDTH  // cols)
        th    = max(1, DISPLAY_HEIGHT // rows)

        imgs = [cv2.resize(f, (tw, th)) for f in snapshot]
        while len(imgs) < rows * cols:
            imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        try:
            final = cv2.vconcat([
                cv2.hconcat(imgs[r * cols:(r + 1) * cols])
                for r in range(rows)
            ])
        except Exception:
            final = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        cv2.imshow("Detection", final)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

else:
    while True:
        time.sleep(1)