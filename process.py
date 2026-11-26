"""Process a project: detect, track, swap MAPPED faces only, rebuild MP4.
Usage: python process.py PROJECT_ID [start_sec] [seconds]
Resumable: run the same command again to continue.
To stop cleanly, create a file named 'cancel' in the project folder.
Models are non-commercial research only: personal, authorized, consented use."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from detector import FaceDetector
from swap import Swapper
from tracker import Tracker

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RECAST_DATA", Path.home() / "recast_data"))
MODEL = BASE / "models" / "face_detection_yunet_2026may.onnx"
FPS = 12
SIZE = 720  # longest side of working frames


def largest(faces):
    return max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))


def run(pid, start=0.0, dur=None, progress=None):
    p = DATA / pid
    meta = json.loads((p / "meta.json").read_text())
    mf = p / "mappings.json"
    maps = json.loads(mf.read_text()) if mf.exists() else []
    if not maps:
        raise ValueError("No character mappings saved for this project.")
    video = p / meta["video"]
    total = meta["info"]["duration"]
    dur = max(0.5, min(dur or total, total - start))

    work = p / "work"
    key = {"start": start, "dur": dur, "fps": FPS, "size": SIZE}
    sf = work / "state.json"
    if sf.exists() and json.loads(sf.read_text()) != key:
        shutil.rmtree(work, ignore_errors=True)
    fr, out = work / "frames", work / "out"
    fr.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    sf.write_text(json.dumps(key))

    if not (work / "extracted").exists():
        for old in fr.glob("f*.jpg"):
            old.unlink()
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur),
            "-i", str(video), "-vf",
            "fps=%d,scale=%d:%d:force_original_aspect_ratio=decrease:force_divisible_by=2" % (FPS, SIZE, SIZE),
            "-q:v", "2", str(fr / "f%05d.jpg")], check=True)
        (work / "extracted").write_text("ok")
    frames = sorted(fr.glob("f*.jpg"))
    if not frames:
        raise ValueError("No frames extracted.")

    det, sw = FaceDetector(str(MODEL)), Swapper()
    latents = {}
    for m in maps:
        ref = Image.open(p / "refs" / m["ref"]).convert("RGB")
        rf = det.detect(ref)
        if not rf:
            raise ValueError("No face found in reference photo " + m["ref"])
        latents[int(m["track"])] = sw.identity(ref, largest(rf))
    print("Swapping tracks:", sorted(latents))

    trk, smooth = Tracker(), {}
    swapped = {t: 0 for t in latents}
    cancel = p / "cancel"
    t0 = time.time()
    for i, fp in enumerate(frames):
        if cancel.exists():
            cancel.unlink()
            raise RuntimeError("Cancelled. Run the same command to resume.")
        img = Image.open(fp).convert("RGB")
        faces = trk.update(det.detect(img), i)
        todo = []
        for f in faces:
            tid = f["track"]
            if tid not in latents:
                continue
            k = np.array(f["kps"], dtype=np.float64)
            prev = smooth.get(tid)
            if prev and prev[0] == i - 1:  # reduce jitter between frames
                k = 0.6 * prev[1] + 0.4 * k
            smooth[tid] = (i, k)
            f["kps"] = k
            todo.append(f)
        dst = out / fp.name
        if not dst.exists():
            for f in todo:
                img = sw.swap_face(img, f, latents[f["track"]])
                swapped[f["track"]] += 1
            tmp = out / (fp.name + ".part")
            img.save(tmp, "JPEG", quality=95)
            os.replace(tmp, dst)
        img.close()
        if progress and (i % 3 == 0 or i == len(frames) - 1):
            progress(i + 1, len(frames), time.time() - t0)

    final = p / "result.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
        "-i", str(out / "f%05d.jpg"), "-ss", str(start), "-t", str(dur),
        "-i", str(video), "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart",
        "-metadata", "comment=AI-generated face replacement",
        str(final)], check=True)
    (p / "DISCLOSURE.txt").write_text(
        "This video contains AI-generated face replacement made with AI Movie "
        "Recast Studio. Faces were replaced only with consent.\n")
    shutil.rmtree(work, ignore_errors=True)
    return final, swapped


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python process.py PROJECT_ID [start_sec] [seconds]")
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    d = float(sys.argv[3]) if len(sys.argv) > 3 else None

    def show(a, b, t):
        eta = t / a * (b - a)
        print("\r%d/%d frames, about %d min left   " % (a, b, eta / 60), end="", flush=True)

    res, counts = run(sys.argv[1], s, d, show)
    print("\nSaved", res)
    print("Faces swapped per track:", counts)
