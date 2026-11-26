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
    """2x3 matrix mapping image points to the standard aligned face."""
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


def _mask():
    m = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(m).ellipse([14, 10, 114, 122], fill=255)
    return m.filter(ImageFilter.GaussianBlur(7))


class Swapper:
    def __init__(self):
        self.swap = _session("inswapper_128.onnx")
        self.arc = _session("w600k_r50.onnx")
        self.arc_in = self.arc.get_inputs()[0].name
        self.emap = np.load(MODELS / "emap.npy").astype(np.float32)
        self.mask = _mask()

    def identity(self, img, face):
        """Identity vector (1,512) from a reference photo and its detected face."""
        crop, _ = align(img.convert("RGB"), face["kps"], 112)
        x = (np.asarray(crop, dtype=np.float32) - 127.5) / 127.5
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
        e = self.arc.run(None, {self.arc_in: x})[0].reshape(1, -1)
        e = e / np.linalg.norm(e)
        lat = e @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def swap_face(self, img, face, latent):
        """Return a copy of img with one face replaced."""
        img = img.convert("RGB")
        crop, M = align(img, face["kps"], 128)
        x = np.asarray(crop, dtype=np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
        y = self.swap.run(None, {"target": x, "source": latent})[0][0]
        y = np.clip(y.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        coef = tuple(M.flatten())
        full = Image.fromarray(y).transform(img.size, Image.AFFINE, coef,
                                            resample=Image.BILINEAR)
        mask = self.mask.transform(img.size, Image.AFFINE, coef,
                                   resample=Image.BILINEAR)
        return Image.composite(full, img, mask)
