"""Simple face tracker: box overlap (IoU) + motion prediction.

Gives each face a stable track id while it stays visible. A track is kept
alive for max_missed frames after its face disappears, so brief gaps keep
the same id. Re-identifying a face after a long absence or a turn-away
needs a face-embedding model, which is a later step.
"""


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box, frame):
        self.id = tid
        self.box = list(box)
        self.vel = [0.0] * 4
        self.missed = 0
        self.first = frame
        self.last = frame

    def predict(self):
        k = self.missed + 1
        return [b + v * k for b, v in zip(self.box, self.vel)]

    def update(self, box, frame):
        k = self.missed + 1
        for i in range(4):
            self.vel[i] = 0.5 * self.vel[i] + 0.5 * (box[i] - self.box[i]) / k
        self.box = list(box)
        self.missed = 0
        self.last = frame


class Tracker:
    def __init__(self, iou_thr=0.3, max_missed=36):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 1

    def update(self, faces, frame):
        """faces: list from FaceDetector.detect. Returns the faces that were
        matched or newly created, each with a 'track' id added."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            pred = t.predict()
            for fi, f in enumerate(faces):
                v = iou(pred, f["box"])
                if v >= self.iou_thr:
                    pairs.append((v, ti, fi))
        pairs.sort(reverse=True)
        used_t, used_f = set(), set()
        for v, ti, fi in pairs:
            if ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            self.tracks[ti].update(faces[fi]["box"], frame)
            faces[fi]["track"] = self.tracks[ti].id
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for fi, f in enumerate(faces):
            if fi not in used_f:
                t = Track(self.next_id, f["box"], frame)
                self.next_id += 1
                self.tracks.append(t)
                f["track"] = t.id
        return faces
