#!/usr/bin/env python3
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional vision dependency
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "vision" / "models" / "model_78.onnx"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "vision" / "resource" / "cone_unknown.png"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "vision" / "detection_result.jpg"


def require_cv2():
    if cv2 is None:
        raise RuntimeError("Vision detection requires OpenCV. Install opencv-python before using this module.")
    return cv2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou(box1, box2):
    """
    box format: [x1, y1, x2, y2, conf]
    """
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(boxes, iou_thresh=0.5):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_thresh]

    return keep


class Detector:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        input_w=320,
        input_h=180,
        conf_thresh=0.5,
        iou_thresh=0.5,
    ):
        self.input_w = input_w
        self.input_h = input_h
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX detector model not found: {model_path}")

        require_cv2()
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("Vision detection requires onnxruntime. Install it before using Detector.") from exc
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image):
        cv = require_cv2()
        """
        image: BGR image from OpenCV
        returns:
            input_tensor: (1, 3, 180, 320)
            resized_bgr: resized image for drawing if needed
            orig_w, orig_h: original image size
        """
        orig_h, orig_w = image.shape[:2]

        resized = cv.resize(image, (self.input_w, self.input_h))
        rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0

        chw = np.transpose(rgb, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)

        return input_tensor, orig_w, orig_h

    def infer(self, image):
        input_tensor, orig_w, orig_h = self.preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # expected shape: (1, 5, 5, 10)
        pred = output[0]
        boxes = self.decode(pred, orig_w, orig_h)
        boxes = nms(boxes, self.iou_thresh)

        results = []
        for b in boxes:
            x1, y1, x2, y2, conf = b
            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "bottom_center": (int((x1 + x2) / 2), int(y2)),
            })
        return results
    
    def decode(self, pred, orig_w, orig_h):
        """
        pred shape: (5, 5, 10)
        interpreted as (channels, grid_h, grid_w)
        channels = [conf, x, y, w, h]
        """
        conf_map = pred[0]
        x_map = pred[1]
        y_map = pred[2]
        w_map = pred[3]
        h_map = pred[4]

        grid_h, grid_w = conf_map.shape
        boxes = []

        for gy in range(grid_h):
            for gx in range(grid_w):
                conf = float(conf_map[gy, gx])

                if conf < self.conf_thresh:
                    continue

                tx = float(x_map[gy, gx])
                ty = float(y_map[gy, gx])
                tw = float(w_map[gy, gx])
                th = float(h_map[gy, gx])

                # likely interpretation:
                # x,y are offsets inside current cell
                # w,h are normalized box size
                cx = (gx + tx) / grid_w
                cy = (gy + ty) / grid_h
                bw = tw
                bh = th

                x1 = (cx - bw / 2.0) * orig_w
                y1 = (cy - bh / 2.0) * orig_h
                x2 = (cx + bw / 2.0) * orig_w
                y2 = (cy + bh / 2.0) * orig_h

                x1 = np.clip(x1, 0, orig_w - 1)
                y1 = np.clip(y1, 0, orig_h - 1)
                x2 = np.clip(x2, 0, orig_w - 1)
                y2 = np.clip(y2, 0, orig_h - 1)

                boxes.append([x1, y1, x2, y2, conf])

        return boxes

    def draw(self, image, detections):
        cv = require_cv2()
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            bx, by = det["bottom_center"]

            cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.circle(vis, (bx, by), 4, (0, 0, 255), -1)
            cv.putText(
                vis,
                f"{conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return vis


if __name__ == "__main__":
    cv = require_cv2()
    detector = Detector(DEFAULT_MODEL_PATH)

    image = cv.imread(str(DEFAULT_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read {DEFAULT_IMAGE_PATH}")
    detections = detector.infer(image)
    print(detections)

    vis = detector.draw(image, detections)
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(DEFAULT_OUTPUT_PATH), vis)
    print(f"Saved {DEFAULT_OUTPUT_PATH}")
