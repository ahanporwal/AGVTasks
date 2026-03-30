from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import pybullet as p

from simulation_setup import setup_simulation


# ══════════════════════════════════════════════════════════════════════
# TUNABLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# ── Camera ────────────────────────────────────────────────────────────
CAM_W, CAM_H = 320, 240
CAM_FOV      = 60.0
CAM_NEAR     = 0.05
CAM_FAR      = 20.0
_CAM_FWD     = 0.55
_CAM_UP      = 0.30

# ── Optical flow — point detection ───────────────────────────────────
LK_MAX_POINTS   = 200
LK_QUALITY      = 0.01
LK_MIN_DIST     = 7
LK_RESEED_BELOW = LK_MAX_POINTS // 2

# ── Optical flow — pyramidal LK solver ───────────────────────────────
LK_WIN_SIZE    = 7
LK_MAX_LEVEL   = 3
LK_ITERATIONS  = 20
LK_EPSILON     = 0.01
LK_DET_THRESH  = 1e-4

# ── Potential field ───────────────────────────────────────────────────
ALPHA         = 0.8
GAMMA         = 0.12
LAMBDA_X      = 0.60
LAMBDA_Y      = 0.80
TTC_EPS       = 0.05
TTC_LARGE     = 1e4
TTC_DANGER    = 3.0
TTC_LATERAL_K = 0.45   # lowered from 20 — 1/TTC² urgency is ~9x larger at ttc=1s

# ── TTC trust gate ────────────────────────────────────────────────────
TTC_MIN_SPEED = 0.6      # m/s

# ── Morse road potential ──────────────────────────────────────────────
MORSE_A     = 260.0
MORSE_B     = 0.06
LANE_HALF   = CAM_H * 0.36
GAUSS_SIGMA = 5.0

# ── Center-lane restoring force ───────────────────────────────────────
CENTER_K = 0.018

# ── World obstacle memory ─────────────────────────────────────────────
OBS_MEMORY_TTL    = 90
OBS_MERGE_RADIUS  = 0.40
OBS_REPROJECT_Z   = 0.35
OBS_MEMORY_CAP    = 60
OBS_HITS_MIN      = 2

# ── Steering commitment ───────────────────────────────────────────────
COMMIT_MIN_FRAMES     = 35
COMMIT_CLEAR_FRAMES   = 35
COMMIT_FLIP_RATIO     = 4.0
WALL_COOLDOWN_FRAMES  = 20
SPATIAL_GATE_FLOOR    = 0.15
COMMIT_CONFIRM_FRAMES =  3
TTC_COMMIT            =  5.0
N_DANGER_MIN          =  2

# ── Spatial gate ─────────────────────────────────────────────────────
SPATIAL_GATE_BASE       = 0.25
SPATIAL_GATE_HOLD_SCALE = 0.20

# ── SMC + actuator ────────────────────────────────────────────────────
C_R         = 3.5
C_L         = 3.0
U0          = 1.4
A0          = 1.2
SMC_EMA     = 0.55
MAX_STEER   = 0.38
BASE_SPEED  = 16.0
V_DESIRED   = 4.5
MOTOR_FORCE = 1400.0
STEER_FORCE = 900.0
DT          = 1.0 / 60.0

# ── Road boundary (world frame) ───────────────────────────────────────
ROAD_HALF_W    = 1.10
WALL_K         = 0.30
WALL_D         = 0.55
LANE_OFFSET    = 0.38
FLOW_MIN_SPEED = 0.20

# ── Stall recovery ────────────────────────────────────────────────────
STALL_SPEED_THRESH   = 0.3
STALL_FRAMES_THRESH  = 15
STALL_ESCAPE_THRESH  = 45
STALL_ESCAPE_FRAMES  = 40
STALL_YAW_CAP_SPEED  = 0.5

# ── Escape steer blend ────────────────────────────────────────────────
ESCAPE_REP_BLEND = 0.55

# ── Goal ──────────────────────────────────────────────────────────────
GOAL_X = 29.5

# ── Road ROI ─────────────────────────────────────────────────────────
ROI_HORIZON_Y  = int(CAM_H * 0.35)
ROI_TOP_HALF_W = int(CAM_W * 0.18)
ROI_BOT_HALF_W = int(CAM_W * 0.46)
ROI_BOTTOM_Y   = CAM_H


# ══════════════════════════════════════════════════════════════════════
# ROAD MASK  (built once at import time)
# ══════════════════════════════════════════════════════════════════════

def build_road_mask() -> np.ndarray:
    mask = np.zeros((CAM_H, CAM_W), dtype=np.uint8)
    cx   = CAM_W // 2
    pts  = np.array([
        [cx - ROI_BOT_HALF_W,  ROI_BOTTOM_Y],
        [cx + ROI_BOT_HALF_W,  ROI_BOTTOM_Y],
        [cx + ROI_TOP_HALF_W,  ROI_HORIZON_Y],
        [cx - ROI_TOP_HALF_W,  ROI_HORIZON_Y],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

_ROAD_MASK = build_road_mask()


# ══════════════════════════════════════════════════════════════════════
# OPTICAL FLOW — custom sparse pyramidal Lucas-Kanade
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FlowConfig:
    pyramid_levels   : int   = LK_MAX_LEVEL
    window_size      : int   = LK_WIN_SIZE
    lk_iterations    : int   = LK_ITERATIONS
    det_threshold    : float = LK_DET_THRESH
    epsilon          : float = LK_EPSILON
    max_points       : int   = LK_MAX_POINTS
    seed_quality     : float = LK_QUALITY
    seed_min_distance: int   = LK_MIN_DIST


_FLOW_CFG = FlowConfig()


def pyrdown(img: np.ndarray) -> np.ndarray:
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
    pyramid = [img]
    for _ in range(levels):
        img = pyrdown(img)
        pyramid.append(img)
    return pyramid


def lucas_kanade_step(
    img1: np.ndarray, img2: np.ndarray,
    x1: float, y1: float,
    x2: float, y2: float,
    half: int,
    det_threshold: float,
) -> tuple[float, float, bool]:
    h, w = img1.shape
    cx1, cy1 = int(round(x1)), int(round(y1))
    cx2, cy2 = int(round(x2)), int(round(y2))
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


def pyramidal_lk(
    img1: np.ndarray, img2: np.ndarray,
    points: np.ndarray,
    cfg: FlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img1.shape[:2]
    min_dim    = min(h, w)
    patch_full = 2 * cfg.window_size + 1
    max_safe   = max(0, int(np.log2(max(min_dim / (patch_full * 4), 1))))
    levels     = min(cfg.pyramid_levels, max_safe)
    pyr1 = build_pyramid(img1, levels)
    pyr2 = build_pyramid(img2, levels)
    n = len(points)
    g = np.zeros((n, 2), dtype=np.float32)
    for lvl in reversed(range(levels + 1)):
        scale    = 2.0 ** lvl
        img1_lvl = pyr1[lvl]
        img2_lvl = pyr2[lvl]
        pts_lvl  = points / scale
        for i in range(n):
            x1, y1 = float(pts_lvl[i, 0]), float(pts_lvl[i, 1])
            dx, dy  = 0.0, 0.0
            for _ in range(cfg.lk_iterations):
                x2 = x1 + g[i, 0] + dx
                y2 = y1 + g[i, 1] + dy
                du, dv, valid = lucas_kanade_step(
                    img1_lvl, img2_lvl,
                    x1, y1, x2, y2,
                    cfg.window_size, cfg.det_threshold,
                )
                if not valid:
                    break
                dx += du
                dy += dv
                if du * du + dv * dv < cfg.epsilon ** 2:
                    break
            g[i, 0] += dx
            g[i, 1] += dy
        if lvl > 0:
            g *= 2.0
    new_points = points + g
    in_bounds  = (
        (new_points[:, 0] >= 0) & (new_points[:, 0] < w) &
        (new_points[:, 1] >= 0) & (new_points[:, 1] < h)
    )
    return new_points.astype(np.float32), in_bounds.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# 1.  CAMERA
# ══════════════════════════════════════════════════════════════════════

def detect_points(gray: np.ndarray,
                  cfg: FlowConfig | None = None) -> np.ndarray:
    cfg = cfg or _FLOW_CFG
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners   = cfg.max_points,
        qualityLevel = cfg.seed_quality,
        minDistance  = cfg.seed_min_distance,
        blockSize    = 7,
        mask         = _ROAD_MASK,   # only seed inside road trapezoid
    )
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts


def run_lk(prev_gray: np.ndarray, curr_gray: np.ndarray,
           points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_2d = points.reshape(-1, 2).astype(np.float32)
    if len(pts_2d) == 0:
        return points, np.zeros((len(points), 1), dtype=np.uint8)
    new_pts, status = pyramidal_lk(prev_gray, curr_gray, pts_2d, _FLOW_CFG)
    return new_pts.reshape(-1, 1, 2), status.reshape(-1, 1)


def capture_frame(car_id: int) -> np.ndarray:
    pos, orn = p.getBasePositionAndOrientation(car_id)
    R   = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
    fwd = R[:, 0]
    up  = R[:, 2]
    eye = np.array(pos) + R @ np.array([_CAM_FWD, 0.0, _CAM_UP])
    view = p.computeViewMatrix(eye.tolist(), (eye + fwd).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(CAM_FOV, CAM_W / CAM_H, CAM_NEAR, CAM_FAR)
    _, _, rgba, _, _ = p.getCameraImage(CAM_W, CAM_H, view, proj,
                                        renderer=p.ER_TINY_RENDERER)
    rgb = np.array(rgba, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════
# 2.  VEHICLE STATE
# ══════════════════════════════════════════════════════════════════════

class VehicleState:
    def __init__(self, car_id: int) -> None:
        self.car_id = car_id
        self.x = self.y = self.psi = self.v = 0.0

    def update(self) -> None:
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        lin, _   = p.getBaseVelocity(self.car_id)
        euler    = p.getEulerFromQuaternion(orn)
        self.x   = float(pos[0])
        self.y   = float(pos[1])
        self.psi = float(euler[2])
        self.v   = float(np.linalg.norm(lin[:2]))


# ══════════════════════════════════════════════════════════════════════
# 3.  FOE
# ══════════════════════════════════════════════════════════════════════

def compute_foe(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float] | None:
    vecs = curr - prev
    mag  = np.linalg.norm(vecs, axis=1)
    ok   = mag > 0.30
    if ok.sum() < 3:
        return None
    p_ok, v_ok = prev[ok], vecs[ok]
    A = np.column_stack([v_ok[:, 1], -v_ok[:, 0]])
    b = v_ok[:, 1] * p_ok[:, 0] - v_ok[:, 0] * p_ok[:, 1]
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        return None


# ══════════════════════════════════════════════════════════════════════
# 4.  TTC
# ══════════════════════════════════════════════════════════════════════

def compute_ttc(prev: np.ndarray, curr: np.ndarray,
                foe: tuple[float, float]) -> np.ndarray:
    mag  = np.linalg.norm(curr - prev, axis=1)
    dist = np.linalg.norm(prev - np.array(foe, dtype=np.float32), axis=1)
    return np.where(mag > TTC_EPS, dist / (mag + 1e-9), TTC_LARGE).clip(0, TTC_LARGE)


# ══════════════════════════════════════════════════════════════════════
# 5.  OBSTACLE MAP + GRADIENT FIELD
# ══════════════════════════════════════════════════════════════════════

def build_obstacle_map(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    canvas = np.zeros((CAM_H, CAM_W), dtype=np.float32)
    mag    = np.linalg.norm(curr - prev, axis=1)
    for pt, m in zip(prev, mag):
        x = int(np.clip(pt[0], 0, CAM_W - 1))
        y = int(np.clip(pt[1], 0, CAM_H - 1))
        canvas[y, x] = max(canvas[y, x], float(m))
    u8 = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if u8.max() < 1:
        return np.zeros((CAM_H, CAM_W), dtype=np.float32)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.float32) / 255.0


def build_gradient_field(obs_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sm     = gaussian_filter(obs_map, sigma=GAUSS_SIGMA)
    gy, gx = np.gradient(sm)
    return gx.astype(np.float32), gy.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 6A.  WORLD OBSTACLE MEMORY
# ══════════════════════════════════════════════════════════════════════

class WorldObstacleMemory:
    def __init__(self) -> None:
        self._records: list[dict] = []
        self._fx = (CAM_W / 2.0) / math.tan(math.radians(CAM_FOV / 2.0))

    def _camera_axes(self, car_id: int):
        try:
            pos, orn = p.getBasePositionAndOrientation(car_id)
        except p.error as e:
            raise RuntimeError(f"PyBullet disconnected in _camera_axes: {e}")
        R   = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
        eye = np.array(pos) + R @ np.array([_CAM_FWD, 0.0, _CAM_UP])
        fwd   = R[:, 0]
        right = -R[:, 1]
        down  = -R[:, 2]
        return eye, fwd, right, down

    def back_project(self, pixel_pts: np.ndarray, car_id: int) -> list[tuple[float, float]]:
        """Project image-space obstacle points to world XY floor plane."""
        if len(pixel_pts) == 0:
            return []
        eye, fwd, right, down = self._camera_axes(car_id)
        world_pts = []
        for px in pixel_pts:
            u, v = float(px[0]), float(px[1])
            ray = fwd + ((u - CAM_W / 2.0) / self._fx) * right \
                      + ((v - CAM_H / 2.0) / self._fx) * down
            ray_z = ray[2]
            if abs(ray_z) < 1e-4:
                continue
            t = (OBS_REPROJECT_Z - eye[2]) / ray_z
            if t <= 0.0:
                continue
            world_pts.append((float(eye[0] + t * ray[0]),
                               float(eye[1] + t * ray[1])))
        return world_pts

    def tick(self) -> None:
        """Advance TTLs without adding new points (called when flow_reliable=False)."""
        next_records = []
        for rec in self._records:
            rec['ttl'] -= 1
            if rec['ttl'] > 0:
                next_records.append(rec)
        self._records = next_records

    def update(self, new_world_pts: list[tuple[float, float]]) -> None:
        """Merge new points and advance TTLs (called when flow_reliable=True)."""
        for nwx, nwy in new_world_pts:
            merged = False
            for rec in self._records:
                if math.hypot(nwx - rec['wx'], nwy - rec['wy']) < OBS_MERGE_RADIUS:
                    rec['wx']   = 0.7 * rec['wx'] + 0.3 * nwx
                    rec['wy']   = 0.7 * rec['wy'] + 0.3 * nwy
                    rec['ttl']  = OBS_MEMORY_TTL
                    rec['hits'] = rec.get('hits', 1) + 1
                    merged = True
                    break
            if not merged:
                self._records.append({'wx': nwx, 'wy': nwy,
                                      'ttl': OBS_MEMORY_TTL, 'hits': 1})
        next_records = []
        for rec in self._records:
            rec['ttl'] -= 1
            if rec['ttl'] > 0:
                next_records.append(rec)
        self._records = next_records[-OBS_MEMORY_CAP:]

    def reproject(self, car_id: int, max_dist: float = 3.0) -> np.ndarray:
        if not self._records:
            return np.empty((0, 2), dtype=np.float32)
        try:
            eye, fwd, right, down = self._camera_axes(car_id)
            pos, orn = p.getBasePositionAndOrientation(car_id)
            car_xy   = np.array([pos[0], pos[1]])
        except RuntimeError:
            return np.empty((0, 2), dtype=np.float32)

        # ── Find closest confirmed record ─────────────────────────────
        min_dist = float('inf')
        for rec in self._records:
            if rec.get('hits', 1) < OBS_HITS_MIN:
                continue
            d = math.hypot(rec['wx'] - car_xy[0], rec['wy'] - car_xy[1])
            if d < min_dist:
                min_dist = d

        if min_dist == float('inf'):
            return np.empty((0, 2), dtype=np.float32)

        # ── Only reproject records within one cluster radius of closest ─
        # Tight enough to exclude the next obstacle; wide enough to cover
        # all points belonging to the same physical object.
        cluster_radius = max(OBS_MERGE_RADIUS * 3.0, 1.0)

        px_list = []
        for rec in self._records:
            if rec.get('hits', 1) < OBS_HITS_MIN:
                continue
            d = math.hypot(rec['wx'] - car_xy[0], rec['wy'] - car_xy[1])
            if d > min_dist + cluster_radius:   # farther obstacle — skip
                continue
            if d > max_dist:                    # even closest too far — skip
                continue
            w_pt  = np.array([rec['wx'], rec['wy'], OBS_REPROJECT_Z])
            delta = w_pt - eye
            depth = float(fwd @ delta)
            if depth <= 0.10:
                continue
            u = CAM_W / 2.0 + self._fx * float(right @ delta) / depth
            v = CAM_H / 2.0 + self._fx * float(down  @ delta) / depth
            if 0 <= u < CAM_W and 0 <= v < CAM_H:
                px_list.append([u, v])
        return (np.array(px_list, dtype=np.float32)
                if px_list else np.empty((0, 2), dtype=np.float32))

    def count(self) -> int:
        return len(self._records)

    def has_obstacle_ahead(self, car_id: int, max_dist: float = 3.5) -> bool:
        if not self._records:
            return False
        try:
            pos, orn = p.getBasePositionAndOrientation(car_id)
            R      = np.array(p.getMatrixFromQuaternion(orn),
                               dtype=np.float64).reshape(3, 3)
            fwd    = R[:, 0]
            car_xy = np.array([pos[0], pos[1]])
        except Exception:
            return False

        for rec in self._records:
            if rec.get('hits', 1) < OBS_HITS_MIN:
                continue
            delta = np.array([rec['wx'], rec['wy']]) - car_xy
            if np.linalg.norm(delta) > max_dist:
                continue
            if fwd[0] * delta[0] + fwd[1] * delta[1] > 0.0:
                return True
        return False

    def purge_behind(self, car_id: int, margin: float = 0.5) -> None:
        try:
            pos, orn = p.getBasePositionAndOrientation(car_id)
            R      = np.array(p.getMatrixFromQuaternion(orn),
                               dtype=np.float64).reshape(3, 3)
            fwd    = R[:, 0]
            car_xy = np.array([pos[0], pos[1]])
        except Exception:
            return

        self._records = [
            rec for rec in self._records
            if (fwd[0] * (rec['wx'] - car_xy[0])
                + fwd[1] * (rec['wy'] - car_xy[1])) > -margin
        ]


# ══════════════════════════════════════════════════════════════════════
# 6B.  STEERING COMMITMENT FILTER
# ══════════════════════════════════════════════════════════════════════

class CommitmentFilter:
    def __init__(self) -> None:
        self.direction      = 0
        self.hold_frames    = 0
        self.commit_force   = 0.0
        self.cooldown       = 0
        self.confirm_frames = 0
        self._vote_dir      = 0
        self._clear_frames  = 0

    def _start_cooldown(self) -> None:
        self.cooldown       = WALL_COOLDOWN_FRAMES
        self.direction      = 0
        self.commit_force   = 0.0
        self.confirm_frames = 0
        self._vote_dir      = 0

    def gate_frac(self) -> float:
        return SPATIAL_GATE_FLOOR if self.cooldown > 0 else 0.0

    # ── Called every danger frame ──────────────────────────────────────
    def vote(self, raw: float,
             min_ttc_live: float,
             n_danger_live: int,
             obs_centroid_x: float | None = None,
             speed: float = 0.0) -> float:

        # Change 1 — centroid-based vote direction (geometrically stable)
        if obs_centroid_x is not None:
            vote_dir = 1 if obs_centroid_x > CAM_W / 2.0 else -1
        else:
            vote_dir = 1 if raw >= 0 else -1

        # Change 3 — speed-scaled hold duration
        def _speed_hold(v: float) -> int:
            return int(np.clip(
                COMMIT_MIN_FRAMES + int(14 * v),
                COMMIT_MIN_FRAMES,
                COMMIT_MIN_FRAMES + 40,
            ))

        # ── Already committed ─────────────────────────────────────────
        if self.direction != 0:
            self._clear_frames = 0
            if vote_dir == self.direction:
                self.hold_frames = _speed_hold(speed)
                if abs(raw) > self.commit_force:
                    self.commit_force = abs(raw)
                return raw
            else:
                # No hold burn on opposite vote — hold is unconditional mid-dodge
                return self.commit_force * self.direction

        # ── Pre-commitment: accumulate streak ─────────────────────────
        if vote_dir == self._vote_dir:
            self.confirm_frames += 1
        else:
            self.confirm_frames = max(0, self.confirm_frames - 1)
            self._vote_dir = vote_dir if self.confirm_frames == 0 else self._vote_dir

        if self.confirm_frames >= COMMIT_CONFIRM_FRAMES:
            self.direction      = self._vote_dir
            self.hold_frames    = _speed_hold(speed)
            self.commit_force   = abs(raw)
            self.cooldown       = 0
            self.confirm_frames = 0

        return raw

    # ── Called every non-danger frame ─────────────────────────────────
    def decay(self) -> None:
        self.confirm_frames = max(0, self.confirm_frames - 1)
        if self.confirm_frames == 0:
            self._vote_dir = 0

        if self.direction != 0:
            self._clear_frames += 1
            if self._clear_frames > COMMIT_CLEAR_FRAMES:
                self.hold_frames = max(0, self.hold_frames - 1)
                if self.hold_frames == 0:
                    self._start_cooldown()
        else:
            self._clear_frames = 0

        if self.cooldown > 0:
            self.cooldown -= 1

    # ── Shim for existing call-sites ──────────────────────────────────
    def filter(self, raw: float,
               min_ttc_live: float,
               n_danger_live: int,
               obs_centroid_x: float | None = None,
               speed: float = 0.0) -> float:
        return self.vote(raw, min_ttc_live, n_danger_live, obs_centroid_x, speed)


# ══════════════════════════════════════════════════════════════════════
# 7.  POTENTIAL FIELD FORCES
# ══════════════════════════════════════════════════════════════════════

def attractive_force(state: VehicleState,
                     goal: tuple[float, float]) -> tuple[float, float]:
    dx   = goal[0] - state.x
    dy   = goal[1] - state.y
    dist = math.hypot(dx, dy) + 1e-8
    return ALPHA * dx / dist, ALPHA * dy / dist


def _spatial_gate_threshold(commit_dir, commit_hold, cooldown_frac=0.0):
    if commit_dir == 0:
        return cooldown_frac * CAM_W

    # HARD LOCK: ignore entire opposite half of image during commitment
    return CAM_W * 0.5


def repulsive_force(
    obs_pts     : np.ndarray,
    gx_field    : np.ndarray,
    gy_field    : np.ndarray,
    ttc_vals    : np.ndarray,
    commit_dir  : int   = 0,
    commit_hold : int   = 0,
    cooldown_frac: float = 0.0,
) -> tuple[float, float]:
    if len(obs_pts) == 0:
        return 0.0, 0.0

    cx      = CAM_W / 2.0
    gate_px = _spatial_gate_threshold(commit_dir, commit_hold, cooldown_frac)

    ttc_sum = float(ttc_vals.sum()) + 1e-8
    rx = 0.0
    for pt, ttc in zip(obs_pts, ttc_vals):
        dx_from_cx = float(pt[0]) - cx
        if commit_dir > 0 and dx_from_cx < gate_px:
            continue
        if commit_dir < 0 and dx_from_cx > -gate_px:
            continue
        ix = int(np.clip(pt[0], 0, CAM_W - 1))
        iy = int(np.clip(pt[1], 0, CAM_H - 1))
        rx += float(gx_field[iy, ix])
    grad_vehicle_y = -GAMMA * rx / ttc_sum

    lat_fy_vehicle = 0.0
    n_danger = 0
    min_ttc = 1e9
    for pt, ttc in zip(obs_pts, ttc_vals):
        if ttc < min_ttc:
            min_ttc = ttc
        if ttc >= TTC_DANGER:
            continue
        dx_from_cx = float(pt[0]) - cx
        if commit_dir > 0 and dx_from_cx < gate_px:
            continue
        if commit_dir < 0 and dx_from_cx > -gate_px:
            continue
        n_danger += 1
        dx_norm = float(np.clip(dx_from_cx / (CAM_W * 0.5), -1.0, 1.0))
        # Change 2 — 1/TTC² urgency: front-loads correction, reduces late oscillation.
        # Normalised so urgency=1.0 at ttc=1 s.
        urgency = 1.0 / (ttc * ttc + 1e-8)
        lat_fy_vehicle += -TTC_LATERAL_K * urgency * dx_norm

    if n_danger > 0:
        lat_fy_vehicle /= math.sqrt(n_danger)
    if min_ttc < 3.0:
        fy = lat_fy_vehicle
    else:
        fy = grad_vehicle_y
    return 0.0, fy


def _morse_grad(y_img: float, y_bnd: float, sign: float) -> float:
    e = math.exp(float(np.clip(sign * MORSE_B * (y_img - y_bnd), -30.0, 30.0)))
    return 2.0 * MORSE_A * MORSE_B * sign * (1.0 - e) * e


def road_force(foe: tuple[float, float] | None,
               qx: float = CAM_W / 2.0,
               qy: float = CAM_H / 2.0) -> tuple[float, float]:
    cx = CAM_W / 2.0
    cy = CAM_H / 2.0
    c2 = (0.002 if foe is None or abs(foe[0] - cx) < CAM_W * 0.15
          else 0.015 * float(np.sign(foe[0] - cx)))
    yr = qy + LANE_HALF * (1.0 + c2 * (qx - cx))
    yl = qy - LANE_HALF * (1.0 + c2 * (qx - cx))
    morse_fy  = _morse_grad(qy, yr, 1.0) + _morse_grad(qy, yl, -1.0)
    fx        = -c2 * abs(morse_fy) * 0.5
    center_fy = CENTER_K * (cy - qy)
    return fx, morse_fy + center_fy


def total_force_world(f_att_img: tuple[float, float],
                      f_rep    : tuple[float, float],
                      f_road   : tuple[float, float],
                      psi      : float) -> tuple[float, float]:
    fx = f_att_img[0] - f_rep[0] - LAMBDA_X * f_road[0]
    fy = f_att_img[1] - f_rep[1] - LAMBDA_Y * f_road[1]
    cp, sp = math.cos(psi), math.sin(psi)
    return fx * cp - fy * sp, fx * sp + fy * cp


# ══════════════════════════════════════════════════════════════════════
# 8.  SLIDING-MODE CONTROLLER
# ══════════════════════════════════════════════════════════════════════

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SlidingModeController:
    def __init__(self) -> None:
        self.delta        = 0.0
        self.psi_err_prev = 0.0
        self.steer_out    = 0.0
        self.speed_out    = BASE_SPEED

    @staticmethod
    def _sat(x: float, phi: float = 0.02) -> float:
        return float(np.clip(x / (phi + 1e-9), -1.0, 1.0))

    def step(self, psi: float, psi_des: float, v: float) -> tuple[float, float]:
        err   = _wrap(psi - psi_des)
        d_err = (err - self.psi_err_prev) / DT
        self.psi_err_prev = err
        s_r        = C_R * err + d_err
        delta_dot  = -U0 * self._sat(s_r)
        self.delta = float(np.clip(self.delta + delta_dot * DT, -MAX_STEER, MAX_STEER))
        s_l       = C_L * v - V_DESIRED * C_L
        accel     = -A0 * self._sat(s_l)
        wheel_vel = BASE_SPEED + accel * 2.0
        wheel_vel = max(wheel_vel, BASE_SPEED * 0.55)
        self.steer_out = SMC_EMA * self.delta  + (1.0 - SMC_EMA) * self.steer_out
        self.speed_out = SMC_EMA * wheel_vel   + (1.0 - SMC_EMA) * self.speed_out
        return self.steer_out, float(np.clip(self.speed_out, 0.0, BASE_SPEED * 1.3))


# ══════════════════════════════════════════════════════════════════════
# 9.  APPLY CONTROL
# ══════════════════════════════════════════════════════════════════════

def apply_control(car_id, steer_j, motor_j, steer, wheel_vel):
    for j in steer_j:
        p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL,
                                targetPosition=steer, force=STEER_FORCE)
    for j in motor_j:
        p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL,
                                targetVelocity=wheel_vel, force=MOTOR_FORCE)


# ══════════════════════════════════════════════════════════════════════
# 10.  OVERLAY
# ══════════════════════════════════════════════════════════════════════

def draw_overlay(frame, prev_pts, curr_pts, obs_mask, ttc_vals,
                 mem_px, foe, f_rep, f_road, steer, state,
                 commit_dir, commit_hold, cooldown, confirm_frames,
                 escape_active):
    vis = frame.copy()

    cooldown_frac = commitment_gate_frac_for_overlay(commit_dir, cooldown)
    gate_px = _spatial_gate_threshold(commit_dir, commit_hold, cooldown_frac)
    if gate_px > 0:
        cx_line = int(CAM_W / 2.0)
        if commit_dir > 0:
            gx = int(np.clip(cx_line + gate_px, 0, CAM_W - 1))
            cv2.line(vis, (gx, 0), (gx, CAM_H), (255, 100, 0), 1, cv2.LINE_AA)
        elif commit_dir < 0:
            gx = int(np.clip(cx_line - gate_px, 0, CAM_W - 1))
            cv2.line(vis, (gx, 0), (gx, CAM_H), (255, 100, 0), 1, cv2.LINE_AA)
        elif cooldown > 0:
            cv2.line(vis, (cx_line, 0), (cx_line, CAM_H), (180, 80, 0), 1, cv2.LINE_AA)

    for i, (p0, p1) in enumerate(zip(prev_pts, curr_pts)):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        if not (0 <= x0 < CAM_W and 0 <= y0 < CAM_H):
            continue
        if len(obs_mask) > i and obs_mask[i]:
            t     = float(np.clip(ttc_vals[i] / 20.0, 0.0, 1.0)) if len(ttc_vals) > i else 1.0
            color = (0, int(200 * t), 220)
        else:
            color = (0, 180, 0)
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA, tipLength=0.5)

    for mp in mem_px:
        mx, my = int(mp[0]), int(mp[1])
        if 0 <= mx < CAM_W and 0 <= my < CAM_H:
            cv2.circle(vis, (mx, my), 3, (200, 0, 200), -1)

    if foe is not None:
        fx = int(np.clip(foe[0], 0, CAM_W - 1))
        fy = int(np.clip(foe[1], 0, CAM_H - 1))
        cv2.drawMarker(vis, (fx, fy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    yr_px = int(CAM_H / 2 + LANE_HALF)
    yl_px = int(CAM_H / 2 - LANE_HALF)
    cv2.line(vis, (0, yr_px), (CAM_W, yr_px), (180, 70, 0), 1)
    cv2.line(vis, (0, yl_px), (CAM_W, yl_px), (180, 70, 0), 1)

    cx_i, cy_i = CAM_W // 2, CAM_H // 2
    sc = 14.0

    def _arr(fx, fy, col):
        ex = int(np.clip(cx_i + fx * sc, 0, CAM_W - 1))
        ey = int(np.clip(cy_i + fy * sc, 0, CAM_H - 1))
        cv2.arrowedLine(vis, (cx_i, cy_i), (ex, ey), col, 2, cv2.LINE_AA, tipLength=0.25)

    _arr(0.0, f_rep[1], (0, 0, 200))
    _arr(*f_road, (180, 70, 0))

    bw, bx, by = 70, CAM_W // 2, CAM_H - 10
    fill = int(steer / MAX_STEER * bw // 2)
    cv2.rectangle(vis, (bx - bw // 2, by - 5), (bx + bw // 2, by + 5), (40, 40, 40), -1)
    cv2.rectangle(vis, (bx + min(0, fill), by - 4), (bx + max(0, fill), by + 4),
                  (0, 200, 255), -1)
    cv2.line(vis, (bx, by - 6), (bx, by + 6), (255, 255, 255), 1)

    if escape_active:
        commit_str = "ESCAPE (REV)"
        commit_col = (0, 60, 255)
    elif commit_dir != 0:
        commit_str = f"COMMIT {'R' if commit_dir > 0 else 'L'} [{commit_hold}] cf={confirm_frames}"
        commit_col = (0, 200, 255)
    elif cooldown > 0:
        commit_str = f"COOLDOWN [{cooldown}]"
        commit_col = (80, 160, 255)
    else:
        commit_str = f"free cf={confirm_frames}"
        commit_col = (150, 150, 150)
    cv2.putText(vis, commit_str, (CAM_W - 130, CAM_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, commit_col, 1)

    min_ttc  = float(ttc_vals.min()) if len(ttc_vals) > 0 else float("inf")
    n_danger = int((ttc_vals < TTC_DANGER).sum()) if len(ttc_vals) > 0 else 0
    hud = [
        f"X     : {state.x:5.1f} m",
        f"Y     : {state.y:+4.2f} m",
        f"Yaw   : {math.degrees(state.psi):+5.1f} deg",
        f"Speed : {state.v:.2f} m/s",
        f"Steer : {steer:+.3f} rad",
        f"TTC   : {min_ttc:5.1f} s",
        f"DANGER: {n_danger}  Mem:{len(mem_px)}",
    ]
    for k, line in enumerate(hud):
        cv2.putText(vis, line, (4, 13 + k * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


def commitment_gate_frac_for_overlay(commit_dir: int, cooldown: int) -> float:
    if commit_dir != 0:
        return 0.0
    return SPATIAL_GATE_FLOOR if cooldown > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# 11.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def run(gui=True, save_video=True, max_frames=5000,
        output_path="navigation_output.mp4"):

    car_id, steer_j, motor_j = setup_simulation(dt=DT, settle_frames=80, gui=gui)

    state      = VehicleState(car_id)
    smc        = SlidingModeController()
    obs_memory = WorldObstacleMemory()
    commitment = CommitmentFilter()
    goal       = (GOAL_X, 0.0)

    stall_frames  = 0
    escape_frames = 0

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (CAM_W * 2, CAM_H))
        print(f"[Video] → {output_path}")

    prev_frame = capture_frame(car_id)
    prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    points = detect_points(prev_gray).reshape(-1, 2)

    print(f"\n[Nav v20] Goal x ≥ {GOAL_X} m  |  v_desired={V_DESIRED} m/s")
    print(f"[Commit]  confirm={COMMIT_CONFIRM_FRAMES}fr  "
          f"TTC_COMMIT={TTC_COMMIT}s  N_DANGER_MIN={N_DANGER_MIN}  "
          f"TTC_MIN_SPEED={TTC_MIN_SPEED}  "
          f"hold={COMMIT_MIN_FRAMES}fr  cooldown={WALL_COOLDOWN_FRAMES}fr")
    print(f"[Stall]   escape_thresh={STALL_ESCAPE_THRESH}fr  "
          f"escape_dur={STALL_ESCAPE_FRAMES}fr")
    print(f"[Memory]  TTL={OBS_MEMORY_TTL}fr  hits_min={OBS_HITS_MIN}  "
          f"cap={OBS_MEMORY_CAP}\n")

    _e2 = np.empty((0, 2), dtype=np.float32)

    frame_idx = 0
    try:
        while frame_idx < max_frames:
            p.stepSimulation()
            if gui:
                time.sleep(DT * 0.4)

            state.update()
            # Purge records the car has physically passed — this is the correct
            # mechanism to prevent passed obstacles affecting steering.
            obs_memory.purge_behind(car_id)

            flow_reliable = state.v > FLOW_MIN_SPEED

            # ── Stall / escape state machine ──────────────────────────────
            if state.v < STALL_SPEED_THRESH:
                stall_frames += 1
            else:
                stall_frames = 0
                if escape_frames > 0:
                    escape_frames = 0

            escape_active = False
            if stall_frames >= STALL_ESCAPE_THRESH or escape_frames > 0:
                if escape_frames == 0:
                    stall_frames = 0
                    commitment._start_cooldown()
                    commitment.hold_frames = 0
                    print(f"[Nav] ↩ Escape reverse at frame {frame_idx}  "
                          f"y={state.y:+.2f}  ψ={math.degrees(state.psi):+.1f}°")
                escape_frames += 1
                escape_active  = True
                if escape_frames > STALL_ESCAPE_FRAMES:
                    escape_frames = 0

            # ── Capture current frame ─────────────────────────────────────
            curr_frame = capture_frame(car_id)
            curr_gray  = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            pending_pts = _e2.copy()
            if len(points) < LK_RESEED_BELOW:
                new_pts = detect_points(curr_gray).reshape(-1, 2)
                if len(new_pts) > 0:
                    pending_pts = new_pts

            prev_pts      = _e2.copy()
            curr_pts      = _e2.copy()
            foe           = None
            ttc_vals      = np.array([], dtype=np.float32)
            obs_mask      = np.zeros(0, dtype=bool)
            fallback_mask = np.zeros(0, dtype=bool)
            fallback_ttc  = np.array([], dtype=np.float32)
            f_rep         = (0.0, 0.0)
            f_road        = road_force(None)
            mem_px        = _e2.copy()
            gx_field      = np.zeros((CAM_H, CAM_W), dtype=np.float32)
            gy_field      = np.zeros((CAM_H, CAM_W), dtype=np.float32)

            if len(points) >= 4:
                pts_cv                = points.reshape(-1, 1, 2).astype(np.float32)
                tracked_cv, status_cv = run_lk(prev_gray, curr_gray, pts_cv)
                alive    = status_cv.reshape(-1) == 1
                prev_pts = points[alive]
                curr_pts = tracked_cv.reshape(-1, 2)[alive]

                # ── ROI filter — discard points outside road trapezoid ────
                if len(prev_pts) > 0:
                    in_roi = np.array([
                        _ROAD_MASK[int(np.clip(pt[1], 0, CAM_H-1)),
                                   int(np.clip(pt[0], 0, CAM_W-1))] > 0
                        for pt in prev_pts
                    ], dtype=bool)
                    prev_pts = prev_pts[in_roi]
                    curr_pts = curr_pts[in_roi]
                # ─────────────────────────────────────────────────────────

                if len(pending_pts) > 0:
                    points = (np.vstack([curr_pts, pending_pts])
                              if len(curr_pts) > 0 else pending_pts)
                else:
                    points = curr_pts.copy()

                if len(prev_pts) >= 4:
                    foe = compute_foe(prev_pts, curr_pts)

                ttc_vals = (compute_ttc(prev_pts, curr_pts, foe)
                            if foe is not None
                            else np.full(len(prev_pts), TTC_LARGE, dtype=np.float32))

                obs_map            = build_obstacle_map(prev_pts, curr_pts)
                gx_field, gy_field = build_gradient_field(obs_map)

                obs_mask = np.array(
                    [obs_map[int(np.clip(pt[1], 0, CAM_H-1)),
                             int(np.clip(pt[0], 0, CAM_W-1))] > 0.5
                     for pt in prev_pts], dtype=bool
                )

                fallback_mask = np.zeros(len(prev_pts), dtype=bool)
                fallback_ttc  = np.full(len(prev_pts), TTC_LARGE, dtype=np.float32)
                if foe is None and len(prev_pts) >= 2:
                    flow_mags = np.linalg.norm(curr_pts - prev_pts, axis=1)
                    n_fwd = 0; mag_sum = 0.0
                    for pt, mag in zip(prev_pts, flow_mags):
                        if (mag > 0.25
                                and abs(float(pt[0]) - CAM_W / 2.0) < CAM_W * 0.35
                                and float(pt[1]) > CAM_H * 0.3):
                            n_fwd += 1; mag_sum += mag
                    if n_fwd >= 2:
                        avg_mag    = mag_sum / n_fwd
                        pseudo_ttc = float(np.clip(10.0 / (avg_mag + 1e-6),
                                                   0.5, TTC_DANGER - 0.1))
                        for i, (pt, mag) in enumerate(zip(prev_pts, flow_mags)):
                            if (mag > 0.25
                                    and abs(float(pt[0]) - CAM_W/2.0) < CAM_W * 0.35
                                    and float(pt[1]) > CAM_H * 0.3):
                                fallback_mask[i] = True
                                fallback_ttc[i]  = pseudo_ttc

                if flow_reliable:
                    fresh_obs = prev_pts[obs_mask] if obs_mask.any() else _e2
                    new_world = obs_memory.back_project(fresh_obs, car_id)
                    obs_memory.update(new_world)
                else:
                    obs_memory.tick()

                mem_px = obs_memory.reproject(car_id)

                if flow_reliable:
                    # ── Nearest-cluster filter on live flow points ────────
                    # Only feed the closest obstacle cluster to the force
                    # calculation — prevents the next obstacle from contributing
                    # before the current one is cleared.
                    if obs_mask.any():
                        obs_pts_all = prev_pts[obs_mask]
                        obs_ttc_all = ttc_vals[obs_mask]
                        ref   = np.array([CAM_W / 2.0, CAM_H], dtype=np.float32)
                        dists = np.linalg.norm(obs_pts_all - ref, axis=1)
                        nearest_dist = dists.min()
                        near_mask = dists < nearest_dist + CAM_W * 0.25
                        aug_obs = obs_pts_all[near_mask]
                        aug_ttc = obs_ttc_all[near_mask]
                    else:
                        aug_obs = _e2.copy()
                        aug_ttc = np.array([], dtype=np.float32)
                    # ─────────────────────────────────────────────────────

                    if fallback_mask.any():
                        aug_obs = (np.vstack([aug_obs, prev_pts[fallback_mask]])
                                   if len(aug_obs) > 0
                                   else prev_pts[fallback_mask])
                        aug_ttc = (np.concatenate([aug_ttc, fallback_ttc[fallback_mask]])
                                   if len(aug_ttc) > 0
                                   else fallback_ttc[fallback_mask])

                    if len(mem_px) > 0:
                        mem_ttc = np.full(len(mem_px), TTC_DANGER * 4.0, dtype=np.float32)
                        aug_obs = (np.vstack([aug_obs, mem_px])
                                   if len(aug_obs) > 0 else mem_px)
                        aug_ttc = (np.concatenate([aug_ttc, mem_ttc])
                                   if len(aug_ttc) > 0 else mem_ttc)

                    f_rep = repulsive_force(
                        aug_obs, gx_field, gy_field, aug_ttc,
                        commit_dir   = commitment.direction,
                        commit_hold  = commitment.hold_frames,
                        cooldown_frac= commitment.gate_frac(),
                    )
                else:
                    if len(mem_px) > 0:
                        mem_ttc = np.full(len(mem_px), TTC_DANGER * 2.0, dtype=np.float32)
                        f_rep = repulsive_force(
                            mem_px, gx_field, gy_field, mem_ttc,
                            commit_dir   = commitment.direction,
                            commit_hold  = commitment.hold_frames,
                            cooldown_frac= commitment.gate_frac(),
                        )

                f_road = road_force(foe)

            else:
                if len(pending_pts) > 0:
                    points = (np.vstack([points, pending_pts])
                              if len(points) > 0 else pending_pts)
                if flow_reliable:
                    obs_memory.update([])
                else:
                    obs_memory.tick()
                mem_px = obs_memory.reproject(car_id)
                if len(mem_px) > 0:
                    mem_ttc = np.full(len(mem_px), TTC_DANGER * 2.0, dtype=np.float32)
                    f_rep = repulsive_force(
                        mem_px, gx_field, gy_field, mem_ttc,
                        commit_dir   = commitment.direction,
                        commit_hold  = commitment.hold_frames,
                        cooldown_frac= commitment.gate_frac(),
                    )

            # ── Danger metrics ────────────────────────────────────────────
            _danger_ttc_list: list[float] = []
            if obs_mask.any():
                _danger_ttc_list.extend(ttc_vals[obs_mask].tolist())
            if fallback_mask.any():
                _danger_ttc_list.extend(fallback_ttc[fallback_mask].tolist())
            if _danger_ttc_list:
                min_ttc_live  = float(min(_danger_ttc_list))
                n_danger_live = int(sum(1 for t in _danger_ttc_list if t < TTC_DANGER))
            else:
                min_ttc_live  = float("inf")
                n_danger_live = 0

            edge_prox = float(np.clip(1.0 - abs(state.y) / ROAD_HALF_W, 0.0, 1.0))
            f_rep = (f_rep[0] * edge_prox, f_rep[1] * edge_prox)

            f_att_w = attractive_force(state, goal)
            cp, sp  = math.cos(state.psi), math.sin(state.psi)
            f_att_img = ( f_att_w[0] * cp + f_att_w[1] * sp,
                         -f_att_w[0] * sp + f_att_w[1] * cp)

            if abs(state.y) > ROAD_HALF_W * 0.90:
                if commitment.direction != 0 or commitment.hold_frames > 0:
                    commitment._start_cooldown()
                commitment.hold_frames = 0

            fw_x, fw_y = total_force_world(f_att_img, f_rep, f_road, state.psi)

            # ── Obstacle centroid for vote direction ──────────────────────
            _centroid_pts: list[float] = []
            if obs_mask.any():
                _centroid_pts.extend(prev_pts[obs_mask, 0].tolist())
            if fallback_mask.any():
                _centroid_pts.extend(prev_pts[fallback_mask, 0].tolist())
            obs_centroid_x = float(np.mean(_centroid_pts)) if _centroid_pts else None

            # ── Commitment filter ─────────────────────────────────────────
            effective_danger = (
                min_ttc_live  < TTC_COMMIT
                and state.v   >= TTC_MIN_SPEED
                and n_danger_live >= N_DANGER_MIN
            )

            if flow_reliable and effective_danger:
                commitment.filter(
                    1.0, min_ttc_live, n_danger_live,
                    obs_centroid_x=obs_centroid_x,
                    speed=state.v,
                )
                if commitment.direction != 0:
                    fw_y = max(abs(fw_y), 0.35) * commitment.direction
            else:
                # Suppress decay when committed and obstacle is still ahead in
                # world space — car turned it out of FOV, hasn't cleared it yet.
                if commitment.direction != 0 and ( obs_memory.has_obstacle_ahead(car_id) or min_ttc_live < 4.0 ):
                    commitment.hold_frames = max(
                        commitment.hold_frames, COMMIT_CONFIRM_FRAMES * 4)
                else:
                    commitment.decay()

            # ── Wall spring (PD) ──────────────────────────────────────────
            wall_spring_active = (
                min_ttc_live >= TTC_DANGER
                and commitment.cooldown == 0
                and not escape_active
            )
            if wall_spring_active:
                lane_center   = 0.0
                lat_vel       = state.v * math.sin(state.psi)
                lateral_error = state.y - lane_center
                wall_p = float(np.clip(-WALL_K * lateral_error / ROAD_HALF_W, -0.8, 0.8))
                wall_d = -WALL_D * lat_vel
                fw_y  += wall_p + wall_d

            # ── Escape override ───────────────────────────────────────────
            if escape_active:
                center_steer = float(np.clip(
                    -state.y / ROAD_HALF_W, -MAX_STEER, MAX_STEER))
                rep_steer = float(np.clip(
                    f_rep[1] * 0.15, -MAX_STEER, MAX_STEER))
                escape_steer = float(np.clip(
                    (1.0 - ESCAPE_REP_BLEND) * center_steer
                    + ESCAPE_REP_BLEND * rep_steer,
                    -MAX_STEER, MAX_STEER))
                apply_control(car_id, steer_j, motor_j,
                              escape_steer, -BASE_SPEED * 0.6)
                steer = escape_steer

            else:
                yaw_cap = (math.radians(10) if state.v < STALL_YAW_CAP_SPEED
                           else math.radians(20))
                psi_desired = math.atan2(fw_y, fw_x + 1e-9)
                psi_desired = float(np.clip(psi_desired, -yaw_cap, yaw_cap))

                steer, wheel_vel = smc.step(state.psi, psi_desired, state.v)

                if min_ttc_live < TTC_DANGER:
                    ttc_scale = float(np.clip(min_ttc_live / TTC_DANGER, 0.35, 1.0))
                    steer = smc.delta
                else:
                    ttc_scale = 1.0

                steer_fraction = abs(steer) / MAX_STEER
                turn_scale = (1.0 - 0.30 * steer_fraction) if state.v > 0.8 else 1.0
                wheel_vel  = wheel_vel * min(ttc_scale, turn_scale)
                wheel_vel  = max(wheel_vel, BASE_SPEED * 0.65)

                apply_control(car_id, steer_j, motor_j, steer, wheel_vel)

            overlay = draw_overlay(
                curr_frame, prev_pts, curr_pts, obs_mask, ttc_vals,
                mem_px, foe, f_rep, f_road, steer, state,
                commitment.direction, commitment.hold_frames,
                commitment.cooldown, commitment.confirm_frames,
                escape_active,
            )
            if writer:
                writer.write(np.hstack([curr_frame, overlay]))
            if gui:
                cv2.imshow("Camera", curr_frame)
                cv2.imshow("Flow + SMC", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_idx % 60 == 0:
                min_ttc  = float(ttc_vals.min()) if len(ttc_vals) else float("inf")
                n_danger = int((ttc_vals < TTC_DANGER).sum()) if len(ttc_vals) else 0
                gate_px  = _spatial_gate_threshold(
                    commitment.direction,
                    commitment.hold_frames,
                    commitment.gate_frac(),
                )
                esc_tag = "  ESC" if escape_active else ""
                print(
                    f"[F{frame_idx:4d}]  x={state.x:5.1f}  y={state.y:+4.2f}  "
                    f"ψ={math.degrees(state.psi):+5.1f}°  v={state.v:.2f}  "
                    f"δ={steer:+.3f}  TTC={min_ttc:5.1f}  "
                    f"danger={n_danger}  mem={obs_memory.count()}  "
                    f"commit={'R' if commitment.direction>0 else 'L' if commitment.direction<0 else '-'}"
                    f"[{commitment.hold_frames}]  "
                    f"cd={commitment.cooldown}  cf={commitment.confirm_frames}  "
                    f"gate={gate_px:.0f}px{esc_tag}"
                )

            if state.x >= GOAL_X:
                print(f"\n[Nav] ✓ Goal at frame {frame_idx}! "
                      f"pos=({state.x:.2f},{state.y:.2f}) v={state.v:.2f}")
                break

            prev_gray  = curr_gray
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[Nav] Ctrl-C")
    except RuntimeError as e:
        print(f"\n[Nav] Simulation ended unexpectedly: {e}")
    finally:
        if writer:
            writer.release()
            print(f"[Video] Saved → {output_path}")
        cv2.destroyAllWindows()
        try:
            p.disconnect()
        except Exception:
            pass
        print(f"[Nav] Done. Frames: {frame_idx}")


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui",     dest="gui",        action="store_false")
    ap.add_argument("--no-video",   dest="save_video", action="store_false")
    ap.add_argument("--max-frames", type=int, default=5000)
    ap.add_argument("--output",     type=str, default="navigation_output.mp4")
    args = ap.parse_args()
    run(gui=args.gui, save_video=args.save_video,
        max_frames=args.max_frames, output_path=args.output)