"""Face swap: inswapper_128 + ArcFace identity (ONNX Runtime, numpy, Pillow).
These models are non-commercial research only: personal use, authorized
footage and consenting faces only."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

MODELS = Path(__file__).resolve().parent / "models"
ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def _session(name):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    return ort.InferenceSession(str(MODELS / name), sess_options=o,
                                providers=["CPUExecutionProvider"])


def similarity(kps, size):
    dst = ARC * (size / 112.0)
    src = np.asarray(kps, dtype=np.float64)
    sm, dm = src.mean(0), dst.mean(0)
    sc, dc = src - sm, dst - dm
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    scale = (S * d).sum() / ((sc ** 2).sum() / len(src))
    t = dm - scale * R @ sm
    return np.hstack([scale * R, t[:, None]])


def _inv(M):
    return np.linalg.inv(np.vstack([M, [0.0, 0.0, 1.0]]))[:2]


def align(img, kps, size):
    M = similarity(kps, size)
    out = img.transform((size, size), Image.AFFINE, tuple(_inv(M).flatten()),
                        resample=Image.BILINEAR)
    return out, M


def _soft_ellipse(box, blur):
    m = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return m.filter(ImageFilter.GaussianBlur(blur))


class Swapper:
    def __init__(self):
        self.swap = _session("inswapper_128.onnx")
        self.arc = _session("w600k_r50.onnx")
        self.arc_in = self.arc.get_inputs()[0].name
        self.emap = np.load(MODELS / "emap.npy").astype(np.float32)
        self.mask = _soft_ellipse([18, 14, 110, 118], 8)
        core = _soft_ellipse([34, 30, 94, 104], 10)
        self.core = np.asarray(core, dtype=np.float32)[:, :, None] / 255.0

    def identity(self, img, face):
        crop, _ = align(img.convert("RGB"), face["kps"], 112)
        x = (np.asarray(crop, dtype=np.float32) - 127.5) / 127.5
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
        e = self.arc.run(None, {self.arc_in: x})[0].reshape(1, -1)
        e = e / np.linalg.norm(e)
        lat = e @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def _match_colour(self, y, x):
        """Shift y's colour statistics to match x inside the face core."""
        w = self.core
        ws = float(w.sum()) + 1e-6
        out = y.copy()
        for c in range(3):
            my = float((y[:, :, c:c + 1] * w).sum() / ws)
            mt = float((x[:, :, c:c + 1] * w).sum() / ws)
            sy = float(np.sqrt((((y[:, :, c:c + 1] - my) ** 2) * w).sum() / ws)) + 1e-4
            st = float(np.sqrt((((x[:, :, c:c + 1] - mt) ** 2) * w).sum() / ws)) + 1e-4
            gain = min(1.33, max(0.75, st / sy))
            out[:, :, c] = (y[:, :, c] - my) * gain + mt
        return np.clip(out, 0.0, 1.0)

    def swap_face(self, img, face, latent):
        img = img.convert("RGB")
        crop, M = align(img, face["kps"], 128)
        x = np.asarray(crop, dtype=np.float32) / 255.0
        xin = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
        y = self.swap.run(None, {"target": xin, "source": latent})[0][0]
        y = np.clip(y.transpose(1, 2, 0), 0.0, 1.0)
        y = self._match_colour(y, x)
        y8 = (y * 255.0).astype(np.uint8)
        coef = tuple(M.flatten())
        full = Image.fromarray(y8).transform(img.size, Image.AFFINE, coef,
                                             resample=Image.BILINEAR)
        mask = self.mask.transform(img.size, Image.AFFINE, coef,
                                   resample=Image.BILINEAR)
        return Image.composite(full, img, mask)
