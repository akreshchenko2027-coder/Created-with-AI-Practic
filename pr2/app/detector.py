import time
import io
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

class DetectionError(Exception):
    """Кастомна помилка для збоїв під час детекції"""
    pass

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

def detect(image_bytes: bytes, conf_threshold: float = 0.25) -> dict:
    if not image_bytes:
        raise DetectionError("Отримано порожній файл")

    img = None

    # Варіант 1: Декодування через OpenCV
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    # Варіант 2: Фолбек на PIL (якщо OpenCV повернув None чи впав)
    if img is None:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise DetectionError(f"Не вдалося декодувати зображення (OpenCV та PIL): {e}")

    model = get_model()

    start_time = time.perf_counter()
    results = model(img, conf=conf_threshold, verbose=False)
    inference_time = time.perf_counter() - start_time

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(coord, 2) for coord in xyxy]
            })

    return {
        "count": len(detections),
        "inference_time_sec": round(inference_time, 4),
        "detections": detections
    }