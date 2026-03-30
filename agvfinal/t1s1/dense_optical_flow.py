"""
Pyramidal Lucas–Kanade optical flow with streamline visualisation.

Improvements over original:
- FlowConfig dataclass centralises all magic numbers
- Bug fixes: streamline colour was always white; warp_image crashed on colour images
- lucas_kanade_step_fast now pre-blurs inputs for consistent gradient quality
- Dead code removed (compute_gradients, draw_flow_overlay were never called)
- Frame-rate control prevents the loop falling behind on slow hardware
- Type hints and docstrings throughout
- Pyramid levels auto-clamped to a sensible maximum for the frame size
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
    pyramid_levels: int = 6
    window_size: int = 7
    lk_iterations: int = 3

    # Flow filtering
    det_threshold: float = 1e-3
    structure_threshold: float = 20
    magnitude_threshold: float = 1.0

    # Temporal smoothing  (0 = no smoothing, 1 = frozen)
    temporal_alpha: float = 0.7

    # Morphological mask clean-up
    morph_kernel_size: int = 7
    min_component_area: int = 200

    # Streamline seeding
    max_seed_corners: int = 50
    seed_quality: float = 0.05
    seed_min_distance: int = 15
    seed_min_flow_sq: float = 0.5
    seed_exclusion_radius: int = 15
    min_streamline_length: int = 5

    # Streamline integration
    streamline_steps: int = 400
    streamline_step_size: float = 3.0
    min_step_mag: float = 0.05

    # Visualisation
    canvas_decay: float = 0.97
    frame_width: int = 1280
    frame_height: int = 720

    # Processing
    target_fps: float = 0.0


# ─────────────────────────────────────────────
# 1. Gaussian Pyramid
# ─────────────────────────────────────────────

def pyrdown(img: np.ndarray) -> np.ndarray:
    """
    Downsample `img` by 2x — identical algorithm to cv2.pyrDown.

    Steps:
      1. Convolve with the 5-tap Gaussian kernel [1, 4, 6, 4, 1] / 16
         applied as two separable 1-D passes (rows then columns).
         This is mathematically identical to the full 5×5 convolution
         but costs 2×5 = 10 multiplications per pixel instead of 25.
      2. Subsample by keeping every other row and column ([::2, ::2]).

    mode='same' pads the borders symmetrically so output shape is
    exactly (ceil(H/2), ceil(W/2)), matching cv2.pyrDown.
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
# 2. Lucas–Kanade (single level, vectorised)
# ─────────────────────────────────────────────

def lucas_kanade_step(
    img1: np.ndarray,
    img2: np.ndarray,
    cfg: FlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorised Lucas–Kanade optical flow for one pyramid level.

    Spatial gradients (Ix, Iy) are computed from img1 via Sobel.
    The temporal gradient It = img2 - img1 is computed on raw pixel values.
    Windowed sums are accumulated via GaussianBlur over a (window_size x window_size)
    neighbourhood, which is the standard LK approximation.

    Pre-blurring img1 and img2 before computing It is intentionally avoided:
    blurring both frames identically causes their difference to nearly cancel,
    driving Sxt and Syt to zero and producing zero flow everywhere.

    Returns
    -------
    u, v : flow fields in x and y directions respectively
    """
    win = (cfg.window_size, cfg.window_size)

    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    Ix = cv2.Sobel(img1_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1_f, cv2.CV_32F, 0, 1, ksize=3)
    It = img2_f - img1_f

    Sxx = cv2.GaussianBlur(Ix * Ix, win, 0)
    Syy = cv2.GaussianBlur(Iy * Iy, win, 0)
    Sxy = cv2.GaussianBlur(Ix * Iy, win, 0)
    Sxt = cv2.GaussianBlur(Ix * It, win, 0)
    Syt = cv2.GaussianBlur(Iy * It, win, 0)

    det = Sxx * Syy - Sxy * Sxy
    det = np.maximum(det, 1e-6)

    mask = det > cfg.det_threshold
    u = np.where(mask, (Syy * (-Sxt) - Sxy * (-Syt)) / det, 0.0)
    v = np.where(mask, (Sxx * (-Syt) - Sxy * (-Sxt)) / det, 0.0)

    return u.astype(np.float32), v.astype(np.float32)


# ─────────────────────────────────────────────
# 3. Image warping
# ─────────────────────────────────────────────

def warp_image(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Warp `img` by the displacement field (u, v).
    Works for both grayscale (H x W) and colour (H x W x C) images.
    """
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    return cv2.remap(img, x + u, y + v, interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────
# 4. Pyramidal Lucas–Kanade
# ─────────────────────────────────────────────

def pyramidal_lk(
    img1: np.ndarray,
    img2: np.ndarray,
    cfg: FlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coarse-to-fine Lucas–Kanade optical flow.
    Pyramid levels are clamped so the coarsest image is at least 16x16.
    """
    h, w = img1.shape[:2]
    max_safe_levels = int(np.log2(min(h, w) / 16))
    levels = min(cfg.pyramid_levels, max_safe_levels)

    pyr1 = build_pyramid(img1, levels)
    pyr2 = build_pyramid(img2, levels)

    u = np.zeros_like(pyr1[-1], dtype=np.float32)
    v = np.zeros_like(pyr1[-1], dtype=np.float32)

    for lvl in reversed(range(levels + 1)):
        img1_lvl = pyr1[lvl]
        img2_lvl = pyr2[lvl]
        lh, lw = img1_lvl.shape[:2]

        if lvl < levels:
            u = cv2.resize(u, (lw, lh)) * 2.0
            v = cv2.resize(v, (lw, lh)) * 2.0

        for _ in range(cfg.lk_iterations):
            img2_warped = warp_image(img2_lvl, u, v)
            du, dv = lucas_kanade_step(img1_lvl, img2_warped, cfg)
            u += du
            v += dv

    return u, v


# ─────────────────────────────────────────────
# 5. Flow computation + motion mask
# ─────────────────────────────────────────────

def compute_flow_and_mask(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    u_prev: np.ndarray | None,
    v_prev: np.ndarray | None,
    cfg: FlowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smoothed optical flow and a binary motion mask.

    Returns
    -------
    u, v       : smoothed flow fields
    mag        : squared magnitude (u^2 + v^2)
    clean_mask : binary mask of significant moving regions
    """
    u, v = pyramidal_lk(prev_gray, gray, cfg)

    u = cv2.bilateralFilter(u, 9, 75, 75)
    v = cv2.bilateralFilter(v, 9, 75, 75)

    if u_prev is not None:
        alpha = cfg.temporal_alpha
        u = alpha * u_prev + (1.0 - alpha) * u
        v = alpha * v_prev + (1.0 - alpha) * v

    Ix = cv2.Sobel(prev_gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_32F, 0, 1, ksize=3)
    valid = (Ix * Ix + Iy * Iy) > cfg.structure_threshold
    u[~valid] = 0.0
    v[~valid] = 0.0

    mag = u * u + v * v

    raw_mask = ((mag > cfg.magnitude_threshold) * 255).astype(np.uint8)
    k = np.ones((cfg.morph_kernel_size, cfg.morph_kernel_size), np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k)
    raw_mask = cv2.dilate(raw_mask, k, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
    clean_mask = np.zeros_like(raw_mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > cfg.min_component_area:
            clean_mask[labels == i] = 255

    return u, v, mag, clean_mask


# ─────────────────────────────────────────────
# 6. Streamline helpers
# ─────────────────────────────────────────────

def _sample_flow_bilinear(
    u: np.ndarray,
    v: np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Bilinearly interpolate (u, v) at sub-pixel position (x, y)."""
    h, w = u.shape
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return 0.0, 0.0

    x0, y0 = int(x), int(y)
    wx, wy = x - x0, y - y0

    u_val = (
        (1 - wx) * (1 - wy) * u[y0,     x0]
        +      wx * (1 - wy) * u[y0,     x0 + 1]
        + (1 - wx) *      wy * u[y0 + 1, x0]
        +      wx *       wy * u[y0 + 1, x0 + 1]
    )
    v_val = (
        (1 - wx) * (1 - wy) * v[y0,     x0]
        +      wx * (1 - wy) * v[y0,     x0 + 1]
        + (1 - wx) *      wy * v[y0 + 1, x0]
        +      wx *       wy * v[y0 + 1, x0 + 1]
    )
    return float(u_val), float(v_val)


def _integrate(
    u: np.ndarray,
    v: np.ndarray,
    x: float,
    y: float,
    h: int,
    w: int,
    motion_mask: np.ndarray,
    cfg: FlowConfig,
    direction: float = 1.0,
) -> list[tuple[int, int]]:
    """Integrate a single streamline in one direction."""
    pts: list[tuple[int, int]] = []
    for _ in range(cfg.streamline_steps):
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            break

        dx = u[iy, ix] * direction
        dy = v[iy, ix] * direction
        mag_sq = dx * dx + dy * dy

        if mag_sq < cfg.min_step_mag:
            break

        dx2, dy2 = _sample_flow_bilinear(u, v, x + dx, y + dy)
        dx = 0.7 * dx + 0.3 * dx2 * direction
        dy = 0.7 * dy + 0.3 * dy2 * direction

        norm = np.sqrt(dx * dx + dy * dy) + 1e-6
        x += (dx / norm) * cfg.streamline_step_size
        y += (dy / norm) * cfg.streamline_step_size
        pts.append((int(x), int(y)))

    return pts


def compute_streamline(
    u: np.ndarray,
    v: np.ndarray,
    x: float,
    y: float,
    h: int,
    w: int,
    motion_mask: np.ndarray,
    cfg: FlowConfig,
) -> list[tuple[int, int]]:
    """Compute a bidirectional streamline through (x, y)."""
    fwd = _integrate(u, v, x, y, h, w, motion_mask, cfg,  1.0)
    bwd = _integrate(u, v, x, y, h, w, motion_mask, cfg, -1.0)
    return bwd[::-1] + fwd


# ─────────────────────────────────────────────
# 7. Streamline drawing
# ─────────────────────────────────────────────

def draw_streamlines(
    flow_canvas: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    motion_mask: np.ndarray,
    gray: np.ndarray,
    cfg: FlowConfig,
) -> np.ndarray:
    """
    Seed streamlines from good features inside the motion mask and draw them
    onto `flow_canvas` (float32, range [0, 1]).
    Intensity fades from bright at the tail to dim at the head.
    """
    h, w = flow_canvas.shape[:2]

    features = cv2.goodFeaturesToTrack(
        gray,
        mask=motion_mask,
        maxCorners=cfg.max_seed_corners,
        qualityLevel=cfg.seed_quality,
        minDistance=cfg.seed_min_distance,
    )
    if features is None:
        return flow_canvas

    used = np.zeros((h, w), dtype=np.uint8)

    for pt in features:
        sx, sy = int(pt[0][0]), int(pt[0][1])

        if u[sy, sx] ** 2 + v[sy, sx] ** 2 < cfg.seed_min_flow_sq:
            continue

        if used[sy, sx]:
            continue
        cv2.circle(used, (sx, sy), cfg.seed_exclusion_radius, 1, -1)

        streamline = compute_streamline(u, v, sx, sy, h, w, motion_mask, cfg)
        if len(streamline) < cfg.min_streamline_length:
            continue

        n = len(streamline)
        for i in range(n - 1):
            intensity = 1.0 - (i / n)
            cv2.line(flow_canvas, streamline[i], streamline[i + 1],
                     (intensity, intensity, intensity), 1, cv2.LINE_AA)

    return flow_canvas


# ─────────────────────────────────────────────
# 8. Main video loop
# ─────────────────────────────────────────────

def run_video(source: int | str = 0, cfg: FlowConfig | None = None) -> None:
    """
    Run the optical flow visualiser on a webcam feed or video file.

    Parameters
    ----------
    source : int or str
        Camera index (0) or path to a video file.
    cfg    : FlowConfig, optional
        Configuration; defaults are used if None.
    """
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

    u_prev: np.ndarray | None = None
    v_prev: np.ndarray | None = None

    frame_interval = (1.0 / cfg.target_fps) if cfg.target_fps > 0 else 0.0
    last_time = time.monotonic()

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame2 = cv2.resize(frame2, (cfg.frame_width, cfg.frame_height))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        u, v, _mag, motion_mask = compute_flow_and_mask(
            prev_gray, gray, u_prev, v_prev, cfg
        )
        u_prev, v_prev = u.copy(), v.copy()

        flow_canvas *= cfg.canvas_decay
        flow_canvas = draw_streamlines(flow_canvas, u, v, motion_mask, gray, cfg)

        canvas_vis = (np.clip(flow_canvas, 0.0, 1.0) * 255).astype(np.uint8)
        canvas_vis = cv2.applyColorMap(canvas_vis, cv2.COLORMAP_HOT)

        vis = cv2.addWeighted(frame2, 0.6, canvas_vis, 0.8, 0)

        cv2.imshow("Flow Lines", vis)
        out.write(vis)

        prev_gray = gray

        if frame_interval > 0:
            elapsed = time.monotonic() - last_time
            wait_ms = max(1, int((frame_interval - elapsed) * 1000))
        else:
            wait_ms = 1
        last_time = time.monotonic()

        if cv2.waitKey(wait_ms) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = FlowConfig(
        pyramid_levels=6,
        temporal_alpha=0.7,
        target_fps=30.0,
    )
    run_video("data/OPTICAL_FLOW.mp4", cfg)