"""
Sparse Pyramidal Lucas-Kanade optical flow with trail visualisation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class FlowConfig:
    # Pyramid
    pyramid_levels: int = 4     # kept low so coarsest level stays large enough
                                 # for the patch to fit. At 1280x720, level 4 =
                                 # 80x45 — comfortably fits a window_size=5 patch.
    window_size: int = 5        # half-window; full patch = (2*w+1) x (2*w+1)
    lk_iterations: int = 10

    # LK solver
    det_threshold: float = 1e-3
    epsilon: float = 0.03

    # Point detection
    max_points: int = 200
    seed_quality: float = 0.05
    seed_min_distance: int = 15

    # Trail rendering
    max_trail_len: int = 60
    trail_thickness: int = 2
    stale_window: int = 8        # frames of history to check for movement
    stale_min_spread: float = 3.0  # drop point if it moved less than this many px
    canvas_decay: float = 0.92

    # Visualisation
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: float = 0.0


# ─────────────────────────────────────────────
# 1. Gaussian Pyramid
# ─────────────────────────────────────────────

def pyrdown(img: np.ndarray) -> np.ndarray:
    """
    Downsample img by 2x — identical algorithm to cv2.pyrDown.
    1. Convolve with [1,4,6,4,1]/16 as two separable 1-D passes.
    2. Subsample every other row and column.
    """
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    blurred = np.apply_along_axis(
        lambda row: np.convolve(row, kernel_1d, mode="same"), axis=1,
        arr=img.astype(np.float32),
    )
    blurred = np.apply_along_axis(
        lambda col: np.convolve(col, kernel_1d, mode="same"), axis=0,
        arr=blurred,
    )
    return blurred[::2, ::2]


def build_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a Gaussian pyramid with `levels` downsampled images."""
    pyramid = [img]
    for _ in range(levels):
        img = pyrdown(img)
        pyramid.append(img)
    return pyramid


# ─────────────────────────────────────────────
# 2. Lucas-Kanade step for one point
# ─────────────────────────────────────────────

def lucas_kanade_step(
    img1: np.ndarray,
    img2: np.ndarray,
    x1: float, y1: float,   # source position in img1 (fixed each level)
    x2: float, y2: float,   # current predicted position in img2 (updated each iter)
    half: int,
    det_threshold: float,
) -> tuple[float, float, bool]:
    """
    Solve the LK system for one point.

    p1 extracted at (x1,y1) from img1 — the original appearance, fixed.
    p2 extracted at (x2,y2) from img2 — the current displacement estimate.
    It = p2 - p1 is the residual being minimised each iteration.

    Returns (du, dv, valid). valid=False if patch is out of bounds or
    the region is too flat/ambiguous to solve reliably.
    """
    h, w = img1.shape

    cx1, cy1 = int(round(x1)), int(round(y1))
    cx2, cy2 = int(round(x2)), int(round(y2))

    # Both patches must fit inside their respective images
    if (cx1 - half < 0 or cx1 + half >= w or
            cy1 - half < 0 or cy1 + half >= h):
        return 0.0, 0.0, False
    if (cx2 - half < 0 or cx2 + half >= w or
            cy2 - half < 0 or cy2 + half >= h):
        return 0.0, 0.0, False

    p1 = img1[cy1 - half: cy1 + half + 1,
               cx1 - half: cx1 + half + 1].astype(np.float32)
    p2 = img2[cy2 - half: cy2 + half + 1,
               cx2 - half: cx2 + half + 1].astype(np.float32)

    Ix = cv2.Sobel(p1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(p1, cv2.CV_32F, 0, 1, ksize=3)
    It = p2 - p1

    Sxx = float(np.sum(Ix * Ix))
    Syy = float(np.sum(Iy * Iy))
    Sxy = float(np.sum(Ix * Iy))
    Sxt = float(np.sum(Ix * It))
    Syt = float(np.sum(Iy * It))

    det = Sxx * Syy - Sxy * Sxy
    if abs(det) < det_threshold:
        return 0.0, 0.0, False

    du = (Syy * (-Sxt) - Sxy * (-Syt)) / det
    dv = (Sxx * (-Syt) - Sxy * (-Sxt)) / det

    return du, dv, True


# ─────────────────────────────────────────────
# 3. Pyramidal LK — sparse
# ─────────────────────────────────────────────

def pyramidal_lk(
    img1: np.ndarray,
    img2: np.ndarray,
    points: np.ndarray,     # (N, 2) float32 — [x, y] per point in img1
    cfg: FlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track each point from img1 to img2 using coarse-to-fine LK.

    Key fix: a point is NOT marked lost when its patch falls outside the
    coarse-level image. Coarse levels can be tiny (e.g. 80x45 at level 4)
    and the patch may not fit there — that's normal. The point simply keeps
    g=(0,0) for that level and gets refined at finer levels where it fits.
    Previously, any out-of-bounds at any level permanently killed the point,
    meaning zero points survived past the first (coarsest) level.

    A point is only marked lost if it goes out of the full-resolution image
    after the final level-0 refinement.
    """
    h, w = img1.shape[:2]
    # Clamp levels so the coarsest image is at least 4x the window patch
    min_dim = min(h, w)
    patch_full = 2 * cfg.window_size + 1
    max_safe = int(np.log2(min_dim / (patch_full * 4)))
    levels = max(0, min(cfg.pyramid_levels, max_safe))

    pyr1 = build_pyramid(img1, levels)
    pyr2 = build_pyramid(img2, levels)

    n = len(points)
    # g: displacement guess in current level's coordinate space
    g = np.zeros((n, 2), dtype=np.float32)

    for lvl in reversed(range(levels + 1)):
        scale = 2.0 ** lvl
        img1_lvl = pyr1[lvl]
        img2_lvl = pyr2[lvl]
        lh, lw = img1_lvl.shape[:2]

        pts_lvl = points / scale        # original points scaled to this level

        for i in range(n):
            x1, y1 = float(pts_lvl[i, 0]), float(pts_lvl[i, 1])
            dx, dy = 0.0, 0.0

            for _ in range(cfg.lk_iterations):
                x2 = x1 + g[i, 0] + dx
                y2 = y1 + g[i, 1] + dy

                du, dv, valid = lucas_kanade_step(
                    img1_lvl, img2_lvl,
                    x1, y1, x2, y2,
                    cfg.window_size, cfg.det_threshold,
                )
                if not valid:
                    # Patch out of bounds or flat region at this level —
                    # keep dx,dy as-is and move on; do NOT mark point lost.
                    break

                dx += du
                dy += dv

                if du * du + dv * dv < cfg.epsilon ** 2:
                    break

            # Accumulate incremental displacement into the guess
            g[i, 0] += dx
            g[i, 1] += dy

        # Upsample guess for the next finer level
        if lvl > 0:
            g *= 2.0

    # Compose results — only discard points that left the frame
    new_points = points + g
    in_bounds = (
        (new_points[:, 0] >= 0) & (new_points[:, 0] < w) &
        (new_points[:, 1] >= 0) & (new_points[:, 1] < h)
    )
    status = in_bounds.astype(np.uint8)

    return new_points.astype(np.float32), status


# ─────────────────────────────────────────────
# 4. Point detection
# ─────────────────────────────────────────────

def detect_points(gray: np.ndarray, cfg: FlowConfig) -> np.ndarray:
    """Detect Shi-Tomasi corners. Returns (N, 2) float32 [x, y]."""
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=cfg.max_points,
        qualityLevel=cfg.seed_quality,
        minDistance=cfg.seed_min_distance,
    )
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2)


# ─────────────────────────────────────────────
# 5. Trail drawing
# ─────────────────────────────────────────────

def draw_trails(
    flow_canvas: np.ndarray,
    trails: list[list[tuple[int, int]]],
    cfg: FlowConfig,
) -> np.ndarray:
    """Draw each point's position history as a fading trail."""
    for trail in trails:
        n = len(trail)
        if n < 2:
            continue
        for i in range(n - 1):
            intensity = (i + 1) / n     # dim at tail, bright at head
            cv2.line(flow_canvas, trail[i], trail[i + 1],
                     (intensity, intensity, intensity),
                     cfg.trail_thickness, cv2.LINE_AA)
    return flow_canvas


# ─────────────────────────────────────────────
# 6. Main video loop
# ─────────────────────────────────────────────

def run_video(source: int | str = 0, cfg: FlowConfig | None = None) -> None:
    if cfg is None:
        cfg = FlowConfig()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source '{source}'")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("Error: cannot read first frame")
        cap.release()
        return

    frame1 = cv2.resize(frame1, (cfg.frame_width, cfg.frame_height))
    h, w = frame1.shape[:2]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flow_canvas = np.zeros((h, w, 3), dtype=np.float32)

    points = detect_points(prev_gray, cfg)
    print(f"Initial points detected: {len(points)}")

    trails: list[list[tuple[int, int]]] = [
        [(int(p[0]), int(p[1]))] for p in points
    ]
    frame_count = 0

    frame_interval = (1.0 / cfg.target_fps) if cfg.target_fps > 0 else 0.0
    last_time = time.monotonic()

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame2 = cv2.resize(frame2, (cfg.frame_width, cfg.frame_height))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # ── Track existing points ────────────────────────────────
        if len(points) > 0:
            new_points, status = pyramidal_lk(prev_gray, gray, points, cfg)

            alive  = status == 1
            points = new_points[alive]
            trails = [t for t, ok in zip(trails, alive) if ok]

            for trail, pt in zip(trails, points):
                trail.append((int(pt[0]), int(pt[1])))
                if len(trail) > cfg.max_trail_len:
                    trail.pop(0)

        # ── Drop stale points that have stopped moving ────────────
        # A point whose last few positions are all within a small radius
        # has drifted onto static background. Keeping it blocks fresh seeds
        # from landing on nearby moving objects.
        if len(points) > 0:
            active = []
            keep_trails = []
            for i, (pt, trail) in enumerate(zip(points, trails)):
                if len(trail) >= cfg.stale_window:
                    recent = trail[-cfg.stale_window:]
                    xs = [p[0] for p in recent]
                    ys = [p[1] for p in recent]
                    spread = ((max(xs) - min(xs)) ** 2 +
                              (max(ys) - min(ys)) ** 2) ** 0.5
                    if spread < cfg.stale_min_spread:
                        continue   # drop this point — it has barely moved
                active.append(pt)
                keep_trails.append(trail)
            points = np.array(active, dtype=np.float32) if active else np.empty((0, 2), dtype=np.float32)
            trails = keep_trails

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(points)} points tracked")

        # ── Top-up points without wiping existing trails ──────────
        # Add new seeds whenever below target. Use a smaller occupancy radius
        # (half of seed_min_distance) so fresh seeds can fill in close to
        # surviving points — important when a tracked point has drifted slightly
        # away from the good feature it was originally seeded on.
        if len(points) < cfg.max_points:
            candidates = detect_points(gray, cfg)

            occupied = np.zeros(gray.shape, dtype=np.uint8)
            occupy_r = max(1, cfg.seed_min_distance // 2)
            for pt in points:
                cv2.circle(occupied, (int(pt[0]), int(pt[1])),
                           occupy_r, 255, -1)

            new_seeds = []
            for pt in candidates:
                x, y = int(pt[0]), int(pt[1])
                if occupied[y, x] == 0:
                    new_seeds.append(pt)
                    cv2.circle(occupied, (x, y), occupy_r, 255, -1)

            if new_seeds:
                extra = np.array(new_seeds, dtype=np.float32)
                points = np.vstack([points, extra]) if len(points) > 0 else extra
                for pt in new_seeds:
                    trails.append([(int(pt[0]), int(pt[1]))])

        # ── Render ───────────────────────────────────────────────
        flow_canvas *= cfg.canvas_decay
        flow_canvas = draw_trails(flow_canvas, trails, cfg)

        canvas_vis = (np.clip(flow_canvas, 0.0, 1.0) * 255).astype(np.uint8)
        canvas_vis = cv2.applyColorMap(canvas_vis, cv2.COLORMAP_HOT)

        vis = cv2.addWeighted(frame2, 0.6, canvas_vis, 0.8, 0)
        out.write(vis)

        prev_gray = gray

        last_time = time.monotonic()

    cap.release()
    out.release()
    print("Output saved to output.mp4")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = FlowConfig(
        pyramid_levels=4,
        window_size=5,
        lk_iterations=10,
        max_points=200,
        max_trail_len=60,
        target_fps=30.0,
    )
    run_video("data/OPTICAL_FLOW.mp4", cfg)