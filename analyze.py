"""Scan a project video: detect + track faces, save one thumbnail per track.

Usage: python analyze.py PROJECT_ID [start_sec] [seconds]
Writes tracks.json and thumbs/ID.jpg inside the project folder.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from detector import FaceDetector
from tracker import Tracker

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RECAST_DATA", Path.home() / "recast_data"))
MODEL = BASE / "models" / "face_detection_yunet_2026may.onnx"
FPS = 12


def analyze(pid, start=0.0, dur=None, progress=None):
    p = DATA / pid
    meta = json.loads((p / "meta.json").read_text())
    video = p / meta["video"]
    dur = dur or meta["info"]["duration"]
    (p / "thumbs").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="recast_"))
    try:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur),
            "-i", str(video),
            "-vf", "fps=%d,scale=640:640:force_original_aspect_ratio=decrease" % FPS,
            "-q:v", "3", str(tmp / "f%05d.jpg")], check=True)
        frames = sorted(tmp.glob("f*.jpg"))
        det, trk = FaceDetector(str(MODEL)), Tracker()
        tracks = {}
        for i, fp in enumerate(frames):
            img = Image.open(fp).convert("RGB")
            for f in trk.update(det.detect(img), i):
                t = tracks.setdefault(f["track"], {
                    "id": f["track"], "first": i, "last": i, "count": 0,
                    "best": 0.0})
                t["last"] = i
                t["count"] += 1
                area = (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])
                if f["score"] * area > t["best"]:
                    t["best"] = f["score"] * area
                    x1, y1, x2, y2 = f["box"]
                    pad = 0.25 * max(x2 - x1, y2 - y1)
                    crop = img.crop((max(0, x1 - pad), max(0, y1 - pad),
                                     min(img.width, x2 + pad), min(img.height, y2 + pad)))
                    crop.thumbnail((160, 160))
                    crop.save(p / "thumbs" / ("%d.jpg" % f["track"]))
            img.close()
            if progress and i % 12 == 0:
                progress(i + 1, len(frames))
        out = {"fps": FPS, "frames": len(frames),
               "tracks": sorted(tracks.values(), key=lambda t: -t["count"])}
        (p / "tracks.json").write_text(json.dumps(out, indent=2))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze.py PROJECT_ID [start_sec] [seconds]")
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    d = float(sys.argv[3]) if len(sys.argv) > 3 else None
    res = analyze(sys.argv[1], s, d, lambda a, b: print("%d/%d" % (a, b)))
    for t in res["tracks"]:
        print("Track %d: frames %d-%d, seen %d" % (t["id"], t["first"], t["last"], t["count"]))
