"""
Tracking and motion estimation utilities for Visual Odometry.
Handles Lucas-Kanade tracking, motion estimation, and adaptive parameters.
"""

import math
import numpy as np
import cv2


def track_lk(prev_gray, gray, prev_pts, win, levels):
    """Track features using Lucas-Kanade optical flow."""
    nxt, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None,
        winSize=(win, win), maxLevel=levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
    )
    st = st.reshape(-1).astype(bool)
    return prev_pts.reshape(-1, 2)[st], nxt.reshape(-1, 2)[st], int(st.sum())


def rigid_from_similarity(A):
    """Extract rigid rotation from similarity transformation."""
    M = A[:, :2]
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def fit_rigid_t(R, p1, p2, mask=None):
    """Fit translation for rigid transformation."""
    if mask is not None:
        m = mask.reshape(-1).astype(bool)
        if m.sum() >= 3:
            p1 = p1[m]
            p2 = p2[m]
    c1 = p1.mean(axis=0)
    c2 = p2.mean(axis=0)
    t = (c2 - (R @ c1)).reshape(2, 1)
    return t


def compose_affine(A_prev0, A_step_inv):
    """Compose affine transformations."""
    R1, t1 = A_prev0[:, :2], A_prev0[:, 2:3]
    R2, t2 = A_step_inv[:, :2], A_step_inv[:, 2:3]
    R = R1 @ R2
    t = R1 @ t2 + t1
    return np.hstack([R, t])


def invert_affine(A):
    """Invert affine transformation."""
    R = A[:, :2]
    t = A[:, 2:3]
    Rinv = np.linalg.inv(R)
    tinv = -Rinv @ t
    return np.hstack([Rinv, tinv])


def corners(w, h):
    """Get corner points of rectangle."""
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def to33(A23):
    """Convert 2x3 affine to 3x3 homogeneous."""
    A33 = np.eye(3, dtype=np.float32)
    A33[:2, :2] = A23[:, :2]
    A33[:2, 2] = A23[:, 2]
    return A33


def to23(A33):
    """Convert 3x3 homogeneous to 2x3 affine."""
    return np.hstack([A33[:2, :2], A33[:2, 2:3]])


class AdaptiveParams:
    """Adaptive parameter controller for tracking robustness."""
    
    def __init__(self, cfg, det):
        self.cfg = cfg
        self.det = det
        self.fast_thr = float(cfg["fast_threshold"])
        self.lk_win = int(cfg["lk_win"])
        self.lk_levels = int(cfg["lk_levels"])
        self.ransac_thr = float(cfg["ransac_thresh_px"])
    
    def _clamp_fast(self):
        """Clamp FAST threshold to valid range."""
        mn, mx = self.cfg["fast_threshold_min"], self.cfg["fast_threshold_max"]
        self.fast_thr = max(mn, min(mx, self.fast_thr))
        self.det.setThreshold(int(self.fast_thr))
    
    def soften(self):
        """Soften parameters for difficult tracking."""
        self.lk_win = int(self.cfg["lk_win_hard"])
        self.lk_levels = int(self.cfg["lk_levels_hard"])
        self.ransac_thr = float(self.cfg["ransac_thresh_hard"])
    
    def harden(self):
        """Harden parameters for easy tracking."""
        self.lk_win = int(self.cfg["lk_win_soft"])
        self.lk_levels = int(self.cfg["lk_levels_soft"])
        self.ransac_thr = float(self.cfg["ransac_thresh_easy"])
    
    def adapt_after(self, inliers):
        """Adapt parameters based on tracking quality."""
        target = int(self.cfg["adapt_inliers_target"])
        if inliers < 0.7 * target:
            self.fast_thr *= float(self.cfg["adapt_down_factor"])
            self._clamp_fast()
            self.soften()
            print(f"    adapting: FAST={int(self.fast_thr)}, LKwin={self.lk_win}, "
                  f"levels={self.lk_levels}, RANSAC={self.ransac_thr:.2f}")
        elif inliers > 1.3 * target:
            self.fast_thr *= float(self.cfg["adapt_up_factor"])
            self._clamp_fast()
            self.harden()
            print(f"    relaxing: FAST={int(self.fast_thr)}, LKwin={self.lk_win}, "
                  f"levels={self.lk_levels}, RANSAC={self.ransac_thr:.2f}")
        else:
            self.harden()
