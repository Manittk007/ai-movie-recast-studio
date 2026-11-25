"""Character mappings: track id -> reference photo (consent required)."""
import json
from pathlib import Path


def load(project):
    f = Path(project) / "mappings.json"
    return json.loads(f.read_text()) if f.exists() else []


def save(project, mappings, ref_dir):
    clean = []
    seen = set()
    for m in mappings:
        tid = int(m["track"])
        ref = Path(str(m["ref"])).name
        label = str(m.get("label", ""))[:40]
        if tid in seen:
            raise ValueError("Track %d is mapped twice." % tid)
        if not (Path(ref_dir) / ref).is_file():
            raise ValueError("Reference photo not found: " + ref)
        if m.get("consent") is not True:
            raise ValueError("Consent is required for every mapping.")
        seen.add(tid)
        clean.append({"track": tid, "ref": ref, "label": label, "consent": True})
    (Path(project) / "mappings.json").write_text(json.dumps(clean, indent=2))
    return clean
