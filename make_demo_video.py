import re
from pathlib import Path

import cv2
import numpy as np
from svg.path import parse_path
from svgpathtools import svg2paths2


def parse_size(value: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", value or "")
    if not match:
        raise ValueError(f"Could not parse numeric size from '{value}'")
    return float(match.group(0))


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    if not color or color == "none":
        return (0, 0, 0)

    c = color.strip().lower()
    if c.startswith("#"):
        c = c[1:]

    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return (0, 0, 0)

    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return (b, g, r)


def load_polygons(svg_path: Path, scale: float = 70, x_offset: float = 350, y_offset: float = 350):
    _, attributes, svg_att = svg2paths2(str(svg_path))
    width = parse_size(svg_att.get("width", "1"))
    height = parse_size(svg_att.get("height", "1"))

    polygons = []
    for attr in attributes:
        d = attr.get("d")
        fill = attr.get("fill", "#000000")
        if not d or fill == "none":
            continue

        path = parse_path(d)
        segment_count = max(len(list(path)), 1)
        sample_count = segment_count + 2
        points = []
        for i in range(sample_count + 1):
            p = path.point(i / sample_count)
            x = int((p.real / width) * scale) - int(x_offset)
            y = int((p.imag / height) * scale) - int(y_offset)
            points.append((x, -y))

        if len(points) >= 3:
            polygons.append((points, hex_to_bgr(fill)))
    return polygons


def draw_partial_path(frame: np.ndarray, pts: np.ndarray, progress: float, color=(0, 0, 0), thickness: int = 2):
    if len(pts) < 2:
        return

    progress = min(max(progress, 0.0), 1.0)
    max_seg = len(pts) - 1
    pos = progress * max_seg
    full_idx = int(pos)
    frac = pos - full_idx

    partial = pts[: full_idx + 1].tolist()
    if full_idx < max_seg:
        p0 = pts[full_idx].astype(np.float32)
        p1 = pts[full_idx + 1].astype(np.float32)
        p = (p0 * (1.0 - frac) + p1 * frac).astype(np.int32)
        partial.append((int(p[0]), int(p[1])))

    if len(partial) >= 2:
        cv2.polylines(
            frame,
            [np.array(partial, dtype=np.int32)],
            False,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def build_video(svg_file: str = "hanumanji.svg", out_file: str = "demo.mp4"):
    polygons = load_polygons(Path(svg_file))
    if not polygons:
        raise RuntimeError("No drawable polygons found in SVG.")

    all_points = np.array([p for poly, _ in polygons for p in poly], dtype=np.int32)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)

    margin = 50
    base_width = int(max_x - min_x + 2 * margin)
    base_height = int(max_y - min_y + 2 * margin)

    target_min_side = 720
    min_side = max(1, min(base_width, base_height))
    render_scale = max(2, int(np.ceil(target_min_side / min_side)))
    width = base_width * render_scale
    height = base_height * render_scale

    shifted = []
    for poly, color in polygons:
        pts = np.array(
            [
                ((x - min_x + margin) * render_scale, (max_y - y + margin) * render_scale)
                for x, y in poly
            ],
            dtype=np.int32,
        )
        shifted.append((pts, color))

    fps = 60
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer. Try installing additional codecs.")

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for _ in range(fps):
        writer.write(canvas)

    for pts, color in shifted:
        seg_dist = np.diff(pts.astype(np.float32), axis=0)
        path_len = float(np.linalg.norm(seg_dist, axis=1).sum())
        stroke_frames = max(10, int(path_len / 14))

        for i in range(1, stroke_frames + 1):
            frame = canvas.copy()
            draw_partial_path(frame, pts, i / stroke_frames, color=(0, 0, 0), thickness=2)
            writer.write(frame)

        cv2.fillPoly(canvas, [pts], color)
        cv2.polylines(canvas, [pts], True, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        for _ in range(2):
            writer.write(canvas)

    for _ in range(fps * 2):
        writer.write(canvas)

    writer.release()


if __name__ == "__main__":
    build_video()
    print("Saved demo video to demo.mp4")
