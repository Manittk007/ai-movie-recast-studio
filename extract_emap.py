"""Extract the 512x512 'emap' table stored inside inswapper_128.onnx.
Saves models/emap.npy. Needs no onnx package: reads the protobuf bytes."""
import mmap
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent / "models" / "inswapper_128.onnx"
OUT = MODEL.with_name("emap.npy")


def varint(buf, pos):
    shift = result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def fields(buf, pos, end):
    """Yield (field_number, wire_type, value). Wire type 2 gives (start, end)."""
    while pos < end:
        key, pos = varint(buf, pos)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = varint(buf, pos)
            yield num, wt, v
        elif wt == 1:
            yield num, wt, (pos, pos + 8)
            pos += 8
        elif wt == 2:
            n, pos = varint(buf, pos)
            yield num, wt, (pos, pos + n)
            pos += n
        elif wt == 5:
            yield num, wt, (pos, pos + 4)
            pos += 4
        else:
            raise ValueError("Unsupported protobuf wire type %d" % wt)


with open(MODEL, "rb") as fh:
    buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    graph = None
    for num, wt, v in fields(buf, 0, len(buf)):
        if num == 7 and wt == 2:
            graph = v
    if graph is None:
        sys.exit("Could not find the model graph.")
    last = None
    for num, wt, v in fields(buf, graph[0], graph[1]):
        if num == 5 and wt == 2:
            last = v
    if last is None:
        sys.exit("No stored tables found in the model.")
    dims, dtype, raw = [], None, None
    for num, wt, v in fields(buf, last[0], last[1]):
        if num == 1:
            if wt == 0:
                dims.append(v)
            else:
                p = v[0]
                while p < v[1]:
                    d, p = varint(buf, p)
                    dims.append(d)
        elif num == 2:
            dtype = v
        elif num == 9 and wt == 2:
            raw = v
    print("Last stored table: dims=%s dtype=%s" % (dims, dtype))
    if dtype != 1 or raw is None or dims != [512, 512]:
        sys.exit("Unexpected table. Send me the line above.")
    arr = np.frombuffer(buf[raw[0]:raw[1]], dtype="<f4").reshape(512, 512).copy()
    np.save(OUT, arr)
    buf.close()
print("Saved", OUT, "min %.3f max %.3f" % (arr.min(), arr.max()))
