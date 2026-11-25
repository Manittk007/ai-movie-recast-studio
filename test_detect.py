"""Usage: python test_detect.py IMAGE.jpg  -> prints faces, saves IMAGE_boxes.jpg"""
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from detector import FaceDetector

if len(sys.argv) < 2:
    sys.exit("Give an image path, e.g. python test_detect.py ~/frame.jpg")

path = Path(sys.argv[1]).expanduser()
model = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2026may.onnx"
img = Image.open(path).convert("RGB")
det = FaceDetector(str(model))

t = time.time()
faces = det.detect(img)
print("Image %dx%d, found %d face(s) in %.2f s" % (img.width, img.height, len(faces), time.time() - t))

draw = ImageDraw.Draw(img)
for n, f in enumerate(faces, 1):
    x1, y1, x2, y2 = f["box"]
    print(" #%d score %.2f box %d,%d,%d,%d" % (n, f["score"], x1, y1, x2, y2))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
    draw.text((x1 + 4, y1 + 4), str(n), fill=(255, 255, 0))

out = path.with_name(path.stem + "_boxes.jpg")
img.save(out)
print("Saved", out)
