"""Minimal pure-Python TensorBoard tfevents scalar reader (no tensorboard dep).

Usage: python read_tb.py <run_dir_or_tfevents_file> [substr_filter]
Prints per-tag: n, first step, first/mid/last simple_value.
"""
import os
import struct
import sys


def _read_records(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    n = len(data)
    while i + 12 <= n:
        (length,) = struct.unpack("<Q", data[i:i + 8])
        i += 8 + 4  # length + crc32(length)
        if i + length + 4 > n:
            break
        payload = data[i:i + length]
        i += length + 4  # payload + crc32(payload)
        yield payload


def _varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, i


def _parse_event(buf):
    """Return (step, [(tag, value), ...]) from an Event protobuf."""
    step = None
    values = []
    i = 0
    n = len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field = key >> 3
        wire = key & 7
        if wire == 0:
            val, i = _varint(buf, i)
            if field == 2:
                step = val
        elif wire == 1:
            i += 8
        elif wire == 2:
            ln, i = _varint(buf, i)
            sub = buf[i:i + ln]
            i += ln
            if field == 5:  # summary
                values.extend(_parse_summary(sub))
        elif wire == 5:
            i += 4
        else:
            break
    return step, values


def _parse_summary(buf):
    out = []
    i = 0
    n = len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field = key >> 3
        wire = key & 7
        if wire == 2:
            ln, i = _varint(buf, i)
            sub = buf[i:i + ln]
            i += ln
            if field == 1:  # value
                tv = _parse_value(sub)
                if tv is not None:
                    out.append(tv)
        elif wire == 0:
            _, i = _varint(buf, i)
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            break
    return out


def _parse_value(buf):
    tag = None
    simple = None
    i = 0
    n = len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field = key >> 3
        wire = key & 7
        if wire == 2:
            ln, i = _varint(buf, i)
            data = buf[i:i + ln]
            i += ln
            if field == 1:
                tag = data.decode("utf-8", "replace")
        elif wire == 5:
            if field == 2:
                (simple,) = struct.unpack("<f", buf[i:i + 4])
            i += 4
        elif wire == 0:
            _, i = _varint(buf, i)
        elif wire == 1:
            i += 8
        else:
            break
    if tag is not None and simple is not None:
        return tag, simple
    return None


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    if os.path.isdir(path):
        files = [
            os.path.join(path, f) for f in os.listdir(path)
            if f.startswith("events.out.tfevents")
        ]
        files.sort()
    else:
        files = [path]

    series = {}  # tag -> list of (step, value)
    for fp in files:
        for payload in _read_records(fp):
            try:
                step, vals = _parse_event(payload)
            except Exception:
                continue
            for tag, v in vals:
                series.setdefault(tag, []).append((step if step is not None else 0, v))

    run = os.path.basename(path.rstrip("/"))
    print(f"run: {run}  ({len(series)} tags)")
    for tag in sorted(series):
        if filt and filt not in tag:
            continue
        pts = series[tag]
        n = len(pts)
        first = pts[0][1]
        mid = pts[n // 2][1]
        last = pts[-1][1]
        step = pts[-1][0]
        print(f"  {tag:50s} n={n:4d} step={step:<7d} "
              f"first={first:+.4f} mid={mid:+.4f} last={last:+.4f}")


if __name__ == "__main__":
    main()
