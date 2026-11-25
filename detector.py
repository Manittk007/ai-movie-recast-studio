"""YuNet face detector using ONNX Runtime + numpy + Pillow (no OpenCV)."""
import numpy as np
import onnxruntime as ort
from PIL import Image

STRIDES = (8, 16, 32)


def _nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= thr]
    return keep


class FaceDetector:
    def __init__(self, model_path, score_thr=0.6, nms_thr=0.3, max_side=640):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.max_side = max_side

    def detect(self, img):
        """img: PIL.Image. Returns a list of dicts with box [x1,y1,x2,y2]
        in original pixels, score, and 5 landmark points."""
        ow, oh = img.size
        scale = min(1.0, self.max_side / max(ow, oh))
        nw = max(32, int(round(ow * scale / 32)) * 32)
        nh = max(32, int(round(oh * scale / 32)) * 32)
        small = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR, 0..255
        blob = np.ascontiguousarray(arr.transpose(2, 0, 1))[None]
        outs = dict(zip(self.out_names, self.sess.run(None, {self.in_name: blob})))

        boxes, scores, kpss = [], [], []
        for s in STRIDES:
            cls = outs["cls_%d" % s][0, :, 0]
            obj = outs["obj_%d" % s][0, :, 0]
            bb = outs["bbox_%d" % s][0]
            kp = outs["kps_%d" % s][0]
            cols, rows = nw // s, nh // s
            if cls.shape[0] != rows * cols:
                raise ValueError("Unexpected output size for stride %d" % s)
            idx = np.arange(cls.shape[0])
            r, c = idx // cols, idx % cols
            score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            k = score > self.score_thr
            if not k.any():
                continue
            r, c, bb, kp, score = r[k], c[k], bb[k], kp[k], score[k]
            cx = (c + bb[:, 0]) * s
            cy = (r + bb[:, 1]) * s
            w = np.exp(bb[:, 2]) * s
            h = np.exp(bb[:, 3]) * s
            boxes.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1))
            scores.append(score)
            kx = (kp[:, 0::2] + c[:, None]) * s
            ky = (kp[:, 1::2] + r[:, None]) * s
            kpss.append(np.stack([kx, ky], 2))

        if not boxes:
            return []
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        kpss = np.concatenate(kpss)
        keep = _nms(boxes, scores, self.nms_thr)
        sx, sy = ow / nw, oh / nh
        faces = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            faces.append({
                "box": [float(max(0, x1 * sx)), float(max(0, y1 * sy)),
                        float(min(ow, x2 * sx)), float(min(oh, y2 * sy))],
                "score": float(scores[i]),
                "kps": [[float(p[0] * sx), float(p[1] * sy)] for p in kpss[i]],
            })
        return faces
