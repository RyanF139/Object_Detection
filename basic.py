import cv2
import time
import requests
import numpy as np
import onnxruntime as ort
from datetime import datetime

# ================= CONFIG =================

RTSP_URL = "rtsp://admin:Moderat2025@192.168.18.30:554/Streaming/Channels/101"

IMG_SIZE = 640                # ukuran inference (ringan)
RESIZE_FRAME = (1280, 720)     # ukuran frame HD (display)

VIEW_SIZE = (800, 500)         # ukuran window (bebas)
ENABLE_VIEW = True
ENABLE_DEBUG = True

WEBHOOK_URL = ""

ALLOWED_CLASSES = ["person", "car", "motorcycle", "truck", "bus"]

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASS_CONFIG = {
    "person": {
        "conf": 0.6,
        "color": (0, 255, 0),
        "min_size": 150
    },
    "car": {
        "conf": 0.4,
        "color": (255, 0, 0),
        "min_size": 1000
    },
    "motorcycle": {
        "conf": 0.5,
        "color": (0, 255, 255),
        "min_size": 200
    },
    "truck": {
        "conf": 0.4,
        "color": (0, 0, 255),
        "min_size": 500
    },
    "bus": {
        "conf": 0.4,
        "color": (255, 255, 0),
        "min_size": 600
    }
}
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table",
    "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ================= LOAD ONNX =================

session = ort.InferenceSession("models/yolov8s.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# ================= PREPROCESS =================

def preprocess(img):
    h, w = img.shape[:2]

    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = img_resized

    img_input = canvas.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    return img_input, scale, nw, nh

# ================= POSTPROCESS =================

def postprocess(outputs, scale):
    outputs = outputs[0]

    if outputs.shape[1] < outputs.shape[2]:
        outputs = np.transpose(outputs, (0,2,1))

    outputs = outputs[0]

    boxes = []
    scores = []
    class_ids = []

    for i in range(outputs.shape[0]):
        row = outputs[i]

        cls_scores = row[4:]
        cls_id = np.argmax(cls_scores)
        conf = cls_scores[cls_id]

        if cls_id >= len(COCO_CLASSES):
            continue

        class_name = COCO_CLASSES[cls_id]

        if class_name not in CLASS_CONFIG:
            continue

        cfg = CLASS_CONFIG[class_name]

        if conf < cfg["conf"]:
            continue

        x, y, w, h = row[:4]

        x1 = int((x - w/2) / scale)
        y1 = int((y - h/2) / scale)
        x2 = int((x + w/2) / scale)
        y2 = int((y + h/2) / scale)
        
        # 🔥 FILTER MIN SIZE
        area = (x2 - x1) * (y2 - y1)

        if area < cfg["min_size"]:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))
        class_ids.append(cls_id)

    # 🔥 NMS DI SINI
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

    results = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cls_id = class_ids[i]

            results.append((
                x,
                y,
                x + w,
                y + h,
                scores[i],
                COCO_CLASSES[cls_id]
            ))

    return results

# ================= RTSP =================

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 🔥 WINDOW SETTING (sekali saja)
if ENABLE_VIEW:
    cv2.namedWindow("ONNX Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ONNX Detection", VIEW_SIZE[0], VIEW_SIZE[1])

frame_count = 0

# ================= MAIN LOOP =================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Reconnect...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_count = 0
        continue

    # 🔥 resize ke HD target
    frame = cv2.resize(frame, RESIZE_FRAME)

    frame_count += 1

    # 🔥 skip frame (hemat CPU)
    if frame_count % 2 != 0:
        continue

    start = time.time()

    # 🔥 inference kecil
    inp, scale, _, _ = preprocess(frame)
    outputs = session.run(None, {input_name: inp})

    detections = postprocess(outputs, scale)

    # 🔥 limit deteksi
    detections = detections[:10]

    data = []

    for x1, y1, x2, y2, conf, cls in detections:
        data.append({
            "class": cls,
            "conf": float(conf),
            "bbox": [x1, y1, x2, y2]
        })

        if ENABLE_VIEW:

            color = CLASS_CONFIG[cls]["color"]
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"{cls} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

    # ================= WEBHOOK =================

    if data and WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={
                "timestamp": datetime.now().isoformat(),
                "detections": data
            }, timeout=2)
        except:
            pass

    # ================= FPS =================

    if ENABLE_DEBUG:
        elapsed = time.time() - start
        if elapsed > 0:
            print(f"FPS: {1/elapsed:.2f}")

    # ================= SHOW =================

    if ENABLE_VIEW:
        cv2.imshow("ONNX Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ================= CLEANUP =================

cap.release()
cv2.destroyAllWindows()