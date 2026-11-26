"""Usage: python swap_test.py REFERENCE_PHOTO TARGET_FRAME
Swaps the largest face in TARGET using the identity from REFERENCE.
Saves ~/swap_test.jpg and prints timings."""
import sys
import time
from pathlib import Path

from PIL import Image

from detector import FaceDetector
from swap import Swapper

if len(sys.argv) < 3:
    sys.exit("Usage: python swap_test.py REFERENCE_PHOTO TARGET_FRAME")


def largest(faces):
    return max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))


base = Path(__file__).resolve().parent
det = FaceDetector(str(base / "models" / "face_detection_yunet_2026may.onnx"))
t = time.time()
sw = Swapper()
print("Models loaded in %.1f s" % (time.time() - t))

ref = Image.open(Path(sys.argv[1]).expanduser()).convert("RGB")
rf = det.detect(ref)
if not rf:
    sys.exit("No face found in the reference photo.")
t = time.time()
latent = sw.identity(ref, largest(rf))
print("Identity vector in %.2f s" % (time.time() - t))

tgt = Image.open(Path(sys.argv[2]).expanduser()).convert("RGB")
tgt.thumbnail((960, 960))
tf = det.detect(tgt)
if not tf:
    sys.exit("No face found in the target frame.")
face = largest(tf)
out = None
for n in (1, 2):
    t = time.time()
    out = sw.swap_face(tgt, face, latent)
    print("Swap run %d: %.2f s (run 1 includes warm-up)" % (n, time.time() - t))
out.save(Path.home() / "swap_test.jpg")
print("Saved", Path.home() / "swap_test.jpg")
