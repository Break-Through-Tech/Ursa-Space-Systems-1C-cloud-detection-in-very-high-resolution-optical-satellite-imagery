#!/usr/bin/env python3
"""Turn a CVAT "CVAT for images 1.1" export into mask_gt.png files.

Use this instead of the "Segmentation mask 1.1" export when shapes were drawn
with the polyline tool. CVAT's mask export skips polylines because a line has
no area. Here a polyline is closed and filled like a polygon.

Drawing order (later shapes cover earlier ones):
  1. full-frame shapes (covering >= 99% of the image) first, as the base layer
  2. everything else by z_order, then in the order CVAT lists them

Output values match apply_cvat_masks.py:
  0 = not cloud, 255 = cloud, 64 = smoke, 128 = unsure

Usage:
  cvat_xml_to_masks.py EXPORT(.zip|.xml) CROPS MANIFEST [--preview DIR]

MANIFEST is the manifest.csv written by prep_for_cvat.py (cvat_name, crop_dir).
"""
import argparse
import csv
import io
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT_VALUES = {"background": 0, "cloud": 255, "smoke": 64, "unsure": 128}


def load_xml(path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return ET.fromstring(z.read("annotations.xml"))
    return ET.parse(path).getroot()


def points(shape):
    return [tuple(map(float, p.split(","))) for p in shape.get("points").split(";")]


EDGE = 30  # px: a polyline end this close to the image edge counts as touching it


def _border_pos(p, w, h):
    """Position of a point along the image border, clockwise from (0, 0)."""
    x, y = p
    dists = [y, w - x, h - y, x]  # top, right, bottom, left
    side = dists.index(min(dists))
    return [x, w + y, w + h + (w - x), 2 * w + h + (h - y)][side]


def _corners_between(a, b, w, h):
    """Corners passed when walking clockwise along the border from a to b."""
    per = 2 * (w + h)
    corners = [(w, 0, w), (w, h, w + h), (0, h, 2 * w + h), (0, 0, 0)]
    span = (b - a) % per
    out = []
    for cx, cy, pos in corners:
        if 0 < (pos - a) % per < span:
            out.append(((pos - a) % per, (cx, cy)))
    return [c for _, c in sorted(out)]


def close_polyline(pts, w, h):
    """Close an open polyline that starts and ends on the image edge.

    A line drawn across the image from edge to edge splits it in two. We close
    it along the border on the side that gives the smaller area, since that is
    almost always the object that was being traced. Other polylines are closed
    with a straight line, the same as a polygon.
    """
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    if abs(x0 - x1) + abs(y0 - y1) < EDGE:
        return pts

    def on_edge(x, y):
        return min(x, y, w - x, h - y) <= EDGE

    if not (on_edge(x0, y0) and on_edge(x1, y1)):
        return pts
    a, b = _border_pos(pts[-1], w, h), _border_pos(pts[0], w, h)
    cw = pts + _corners_between(a, b, w, h)
    ccw = pts + _corners_between(b, a, w, h)[::-1]

    def area(poly):
        return abs(sum(px * qy - qx * py for (px, py), (qx, qy) in zip(poly, poly[1:] + poly[:1]))) / 2

    return min(cw, ccw, key=area)


def shape_layer(shape, w, h):
    """Return a boolean array for one shape, or None for unsupported shapes."""
    tag = shape.tag
    if tag == "mask":
        left, top = int(float(shape.get("left"))), int(float(shape.get("top")))
        bw, bh = int(float(shape.get("width"))), int(float(shape.get("height")))
        rle = [int(v) for v in shape.get("rle").split(",")]
        flat = np.zeros(bw * bh, dtype=bool)
        pos, val = 0, False
        for run in rle:
            if val:
                flat[pos:pos + run] = True
            pos += run
            val = not val
        out = np.zeros((h, w), dtype=bool)
        box = flat.reshape(bh, bw)
        out[top:top + bh, left:left + bw] = box[: h - top, : w - left]
        return out
    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    if tag in ("polygon", "polyline"):
        pts = points(shape)
        if len(pts) < 3:
            return None
        if tag == "polyline":
            pts = close_polyline(pts, w, h)
        d.polygon(pts, fill=1, outline=1)
    elif tag == "box":
        x0, y0 = float(shape.get("xtl")), float(shape.get("ytl"))
        x1, y1 = float(shape.get("xbr")), float(shape.get("ybr"))
        d.rectangle([x0, y0, x1, y1], fill=1)
    elif tag == "ellipse":
        cx, cy = float(shape.get("cx")), float(shape.get("cy"))
        rx, ry = float(shape.get("rx")), float(shape.get("ry"))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=1)
    else:
        return None
    return np.array(img, dtype=bool)


def render(image_el):
    w, h = int(image_el.get("width")), int(image_el.get("height"))
    layers = []
    for i, s in enumerate(image_el):
        label = s.get("label")
        if label not in OUT_VALUES:
            continue
        layer = shape_layer(s, w, h)
        if layer is None:
            continue
        full = layer.mean() >= 0.99
        layers.append((0 if full else 1, int(s.get("z_order", 0)), i, label, layer))
    out = np.zeros((h, w), dtype=np.uint8)
    for *_, label, layer in sorted(layers, key=lambda t: t[:3]):
        out[layer] = OUT_VALUES[label]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("crops", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--preview", type=Path, help="also write small side-by-side previews here")
    a = ap.parse_args()

    with open(a.manifest, newline="") as f:
        manifest = {r["cvat_name"]: r["crop_dir"] for r in csv.DictReader(f)}

    root = load_xml(a.export)
    seen = set()
    for im in root.iter("image"):
        name = Path(im.get("name")).stem
        if name not in manifest:
            print(f"skip {name}: not in manifest", file=sys.stderr)
            continue
        seen.add(name)
        mask = render(im)
        crop = a.crops / manifest[name]
        crop.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(crop / "mask_gt.png")
        pct = {k: (mask == v).mean() * 100 for k, v in OUT_VALUES.items() if k != "background"}
        print(f"{name}: " + " ".join(f"{k}={v:.0f}%" for k, v in pct.items()))
        if a.preview:
            a.preview.mkdir(parents=True, exist_ok=True)
            src = crop / "image.png"
            if src.exists():
                rgb = Image.open(src).convert("RGB").resize((256, 256))
                m = Image.fromarray(mask).resize((256, 256), Image.NEAREST).convert("RGB")
                pair = Image.new("RGB", (512, 256))
                pair.paste(rgb, (0, 0))
                pair.paste(m, (256, 0))
                pair.save(a.preview / f"{name}.jpg")
    for name in sorted(set(manifest) - seen):
        print(f"missing from export: {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
