"""Usage: python test_track.py VIDEO [start_sec] [seconds]
Detects and tracks faces at 12 fps, prints each track, saves ~/track_test.jpg"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from detector import FaceDetector
from tracker import Tracker

if len(sys.argv) < 2:
    sys.exit("Usage: python test_track.py VIDEO [start_sec] [seconds]")

video = Path(sys.argv[1]).expanduser()
start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
model = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2026may.onnx"

tmp = Path(tempfile.mkdtemp(prefix="recast_"))
try:
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur),
        "-i", str(video),
        "-vf", "fps=12,scale=640:640:force_original_aspect_ratio=decrease",
        "-q:v", "3", str(tmp / "f%05d.jpg")], check=True)
    frames = sorted(tmp.glob("f*.jpg"))
    if not frames:
        sys.exit("No frames extracted.")

    det = FaceDetector(str(model))
    trk = Tracker()
    stats = {}
    mid = len(frames) // 2
    t0 = time.time()
    for i, fp in enumerate(frames):
        img = Image.open(fp).convert("RGB")
        faces = trk.update(det.detect(img), i)
        for f in faces:
            s = stats.setdefault(f["track"], [i, i, 0])
            s[1] = i
            s[2] += 1
        if i == mid:
            d = ImageDraw.Draw(img)
            for f in faces:
                x1, y1, x2, y2 = f["box"]
                d.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
                d.text((x1 + 4, y1 + 4), "ID %d" % f["track"], fill=(255, 255, 0))
            img.save(Path.home() / "track_test.jpg")
        img.close()
    el = time.time() - t0
    print("%d frames, %.3f s per frame (detect + track)" % (len(frames), el / len(frames)))
    for tid, (a, b, n) in sorted(stats.items()):
        print(" ID %d: frames %d-%d, seen in %d" % (tid, a, b, n))
    print("Saved", Path.home() / "track_test.jpg")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
