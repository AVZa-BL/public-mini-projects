#!/usr/bin/env python3
"""Renders AppIcon.iconset for Meeting Transcriber without any image library.

    python3 make_icon.py            # (re)write AppIcon.iconset/*.png
    python3 make_icon.py --check    # verify the committed PNGs match this script

The icon: a teal rounded square with seven white bars forming a sound wave.
Rendered once at 1024 px with
analytic anti-aliasing, then box-filtered down to the other sizes macOS wants.
"""
import math
import os
import struct
import sys
import zlib

SIZE = 1024
ICONSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AppIcon.iconset")
# iconutil's expected file names and pixel sizes.
FILES = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
]


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def coverage(distance):
    """Anti-aliased coverage from a signed distance (negative = inside)."""
    return clamp(0.5 - distance, 0.0, 1.0)


def rounded_rect(x, y, cx, cy, half_w, half_h, radius):
    dx = abs(x - cx) - (half_w - radius)
    dy = abs(y - cy) - (half_h - radius)
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside - radius


def circle(x, y, cx, cy, radius):
    return math.hypot(x - cx, y - cy) - radius


def arrow(x, y, cx, cy, length, shaft, head):
    """Right-pointing arrow centred at (cx, cy)."""
    lx = x - cx
    ly = y - cy
    half = length / 2
    shaft_distance = rounded_rect(lx, ly, -head / 2, 0, half - head / 2, shaft / 2, shaft / 2)
    tip = half
    base = half - head
    if lx < base:
        head_distance = base - lx
    else:
        width_here = head * 0.95 * (tip - lx) / head
        head_distance = abs(ly) - width_here
        if lx > tip:
            head_distance = max(head_distance, lx - tip)
    return min(shaft_distance, head_distance)


def triangle_right(x, y, cx, cy, size):
    """Play triangle pointing right, centred at (cx, cy); negative inside."""
    lx = x - cx + size * 0.35
    ly = abs(y - cy)
    # left edge at lx = 0, slanted edge from (0, size*0.6) to (size, 0)
    edge = (ly * size + lx * size * 0.6 - size * size * 0.6) / math.hypot(size, size * 0.6)
    return max(-lx, edge)


def blend(dst, src, alpha):
    return tuple(dst[i] + (src[i] - dst[i]) * alpha for i in range(3))


def frame(color, x, y, cx, cy, hw, hh, fill, edge):
    """A video frame: rounded slab with a darker border."""
    d = rounded_rect(x, y, cx, cy, hw, hh, min(hw, hh) * 0.18)
    color = blend(color, edge, coverage(d))
    return blend(color, fill, coverage(d + max(6.0, hh * 0.08))), d


def render(size):
    pixels = bytearray()
    scale = size / 1024.0
    top = (22, 160, 160)     # teal
    bottom = (8, 70, 110)    # deep blue
    white = (255, 255, 255)
    # Seven rounded bars of a sound wave: (centre x, half height).
    bars = [(222, 70), (332, 150), (442, 250), (512, 330), (582, 250), (692, 150), (802, 70)]
    for py in range(size):
        for px in range(size):
            x = (px + 0.5) / scale
            y = (py + 0.5) / scale
            back = rounded_rect(x, y, 512, 512, 462, 462, 230)
            back_alpha = coverage(back)
            if back_alpha <= 0.0:
                pixels += bytes((0, 0, 0, 0))
                continue
            t = clamp((y - 50) / 924.0, 0.0, 1.0)
            color = tuple(top[i] + (bottom[i] - top[i]) * t for i in range(3))
            for cx, half in bars:
                color = blend(color, white, coverage(rounded_rect(x, y, cx, 512, 36, half, 36)))
            r, g, b = (int(round(clamp(c, 0, 255))) for c in color)
            pixels += bytes((r, g, b, int(round(back_alpha * 255))))
    return bytes(pixels)


def downscale(rgba, size, factor):
    out = bytearray()
    new = size // factor
    area = factor * factor
    for ny in range(new):
        for nx in range(new):
            r = g = b = a = 0
            for dy in range(factor):
                row = ((ny * factor + dy) * size + nx * factor) * 4
                for dx in range(factor):
                    o = row + dx * 4
                    pa = rgba[o + 3]
                    # Premultiplied average so edges do not darken.
                    r += rgba[o] * pa
                    g += rgba[o + 1] * pa
                    b += rgba[o + 2] * pa
                    a += pa
            if a:
                out += bytes((round(r / a), round(g / a), round(b / a), round(a / area)))
            else:
                out += bytes((0, 0, 0, 0))
    return bytes(out)


def png(rgba, size):
    raw = b"".join(b"\x00" + rgba[y * size * 4:(y + 1) * size * 4] for y in range(size))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def build():
    full = render(SIZE)
    files = {}
    for name, size in FILES:
        data = full if size == SIZE else downscale(full, SIZE, SIZE // size)
        files[name] = png(data, size)
    return files


def main(argv):
    files = build()
    if "--check" in argv:
        missing = [n for n in files if not os.path.exists(os.path.join(ICONSET, n))]
        if missing:
            print("missing:", ", ".join(missing))
            return 1
        stale = [n for n, data in files.items() if open(os.path.join(ICONSET, n), "rb").read() != data]
        if stale:
            print("out of date (run make_icon.py):", ", ".join(stale))
            return 1
        print("AppIcon.iconset matches make_icon.py (%d files)." % len(files))
        return 0
    os.makedirs(ICONSET, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(ICONSET, name), "wb") as handle:
            handle.write(data)
        print("wrote %s (%d bytes)" % (name, len(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
