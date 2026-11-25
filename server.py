"""AI Movie Recast Studio - local server (V1, part 1).

Handles projects, video upload, video info, preview frames and
reference photo upload. Detection, tracking and swapping come in
later files. Binds to 127.0.0.1 only: nothing is exposed to the network.
"""
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
import analyze as analyzer
import mapping as mapper
import threading
BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RECAST_DATA", Path.home() / "recast_data"))
DATA.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def project_dir(pid):
    """Return a project folder; reject anything that is not a plain hex id."""
    if not pid.isalnum() or len(pid) != 32:
        abort(404)
    p = DATA / pid
    if not p.is_dir():
        abort(404)
    return p


def read_meta(p):
    f = p / "meta.json"
    return json.loads(f.read_text()) if f.exists() else {}


def write_meta(p, meta):
    (p / "meta.json").write_text(json.dumps(meta, indent=2))


def probe(path):
    """Read duration, size and fps with ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise ValueError(out.stderr.strip() or "ffprobe failed")
    d = json.loads(out.stdout)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "width": s["width"],
        "height": s["height"],
        "fps": round(fps, 3),
        "duration": round(float(d["format"]["duration"]), 2),
    }


# ---------- pages ----------
@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/style.css")
def css():
    return send_from_directory(BASE, "style.css")


@app.get("/app.js")
def js():
    return send_from_directory(BASE, "app.js")


# ---------- projects ----------
@app.get("/api/projects")
def list_projects():
    items = []
    for p in sorted(DATA.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_dir() and (p / "meta.json").exists():
            m = read_meta(p)
            m["id"] = p.name
            items.append(m)
    return jsonify(items)


@app.post("/api/projects")
def create_project():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No video file received."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in VIDEO_EXT:
        return jsonify(error="Unsupported video type: " + ext), 400

    pid = uuid.uuid4().hex
    p = DATA / pid
    (p / "refs").mkdir(parents=True)
    video = p / ("source" + ext)
    f.save(video)
    try:
        info = probe(video)
    except Exception as e:  # corrupt or unreadable file
        shutil.rmtree(p, ignore_errors=True)
        return jsonify(error="Could not read video: %s" % e), 400

    meta = {
        "name": (request.form.get("name") or Path(f.filename).stem)[:60],
        "video": video.name,
        "info": info,
        "consent": False,
    }
    write_meta(p, meta)
    meta["id"] = pid
    return jsonify(meta)


@app.get("/api/projects/<pid>")
def get_project(pid):
    p = project_dir(pid)
    m = read_meta(p)
    m["id"] = pid
    m["refs"] = sorted(x.name for x in (p / "refs").iterdir())
    return jsonify(m)


@app.delete("/api/projects/<pid>")
def delete_project(pid):
    shutil.rmtree(project_dir(pid))
    return jsonify(ok=True)


@app.get("/api/projects/<pid>/video")
def get_video(pid):
    p = project_dir(pid)
    return send_file(p / read_meta(p)["video"], conditional=True)


@app.get("/api/projects/<pid>/frame")
def get_frame(pid):
    """Return one low-resolution JPEG at time t (seconds)."""
    p = project_dir(pid)
    m = read_meta(p)
    try:
        t = max(0.0, float(request.args.get("t", 0)))
        w = min(960, max(120, int(request.args.get("w", 360))))
    except ValueError:
        return jsonify(error="Bad parameters."), 400
    t = min(t, max(0.0, m["info"]["duration"] - 0.1))
    out = p / "preview.jpg"
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", str(p / m["video"]),
        "-frames:v", "1", "-vf", "scale=%d:-2" % w, "-q:v", "4", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not out.exists():
        return jsonify(error=r.stderr.strip() or "Frame extraction failed."), 500
    return send_file(out, mimetype="image/jpeg", max_age=0)


# ---------- reference photos ----------
@app.post("/api/projects/<pid>/refs")
def upload_ref(pid):
    p = project_dir(pid)
    f = request.files.get("photo")
    if not f or not f.filename:
        return jsonify(error="No photo received."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in IMAGE_EXT:
        return jsonify(error="Unsupported image type: " + ext), 400
    if request.form.get("consent") != "yes":
        return jsonify(error="Consent of the person in the photo is required."), 400
    name = uuid.uuid4().hex[:10] + ext
    f.save(p / "refs" / name)
    return jsonify(name=name)


@app.get("/api/projects/<pid>/refs/<name>")
def get_ref(pid, name):
    p = project_dir(pid)
    return send_from_directory(p / "refs", name)

# ---------- analysis and mappings ----------
JOBS = {}


@app.post("/api/projects/<pid>/analyze")
def start_analyze(pid):
    p = project_dir(pid)
    if JOBS.get(pid, {}).get("state") == "running":
        return jsonify(error="Analysis already running."), 409
    m = read_meta(p)
    # Quick test mode: first 10 seconds only, unless full=1
    full = request.args.get("full") == "1"
    dur = None if full else min(10.0, m["info"]["duration"])
    JOBS[pid] = {"state": "running", "done": 0, "total": 1, "error": ""}

    def run():
        try:
            def prog(a, b):
                JOBS[pid].update(done=a, total=b)
            analyzer.analyze(pid, 0.0, dur, prog)
            JOBS[pid]["state"] = "done"
        except Exception as e:
            JOBS[pid].update(state="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/projects/<pid>/analyze")
def analyze_status(pid):
    p = project_dir(pid)
    job = JOBS.get(pid, {"state": "idle", "done": 0, "total": 1, "error": ""})
    out = dict(job)
    f = p / "tracks.json"
    out["tracks"] = json.loads(f.read_text())["tracks"] if f.exists() else []
    return jsonify(out)


@app.get("/api/projects/<pid>/thumbs/<name>")
def get_thumb(pid, name):
    p = project_dir(pid)
    return send_from_directory(p / "thumbs", name, max_age=0)


@app.get("/api/projects/<pid>/mappings")
def get_mappings(pid):
    return jsonify(mapper.load(project_dir(pid)))


@app.put("/api/projects/<pid>/mappings")
def put_mappings(pid):
    p = project_dir(pid)
    try:
        saved = mapper.save(p, request.get_json(force=True) or [], p / "refs")
    except (ValueError, KeyError, TypeError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(saved)
if __name__ == "__main__":
    print("Open http://127.0.0.1:8080 in your phone browser")
    app.run(host="127.0.0.1", port=8080, threaded=True)
