"""
Main Visual Odometry algorithm implementation.
Handles the core VO pipeline, motion estimation, and trajectory building.
"""

import os
import time
import csv
import math
import numpy as np
import cv2
from pathlib import Path

from image_utils import (
    load_image_paths, parse_resize, to_gray, read_image_any, 
    apply_clahe, unsharp, DistortionCorrector
)
from features import (
    build_fast_detector, ensure_pts_fmt, detect_points_FAST, 
    detect_points_tile_FAST, build_orb, orb_kp_desc, try_relocalize
)
from tracking import (
    track_lk, rigid_from_similarity, fit_rigid_t, compose_affine, 
    invert_affine, AdaptiveParams
)
from csv_utils import load_frame_csv, load_imu_csv, en_to_latlon
from visualization import (
    plot_trajectory_xy, plot_gps_track, plot_imu_angles,
    plot_vo_metric_no_imu, plot_vo_metric_with_imu
)


class VisualOdometry:
    """Main Visual Odometry algorithm class."""
    
    def __init__(self, config):
        self.config = config
        self.paths = None
        self.det = None
        self.adapt = None
        self.orb = None
        self.bf = None
        self.dc = None
        self.misses = 0
        self.pairs = []
        self.global_A = []
        self.pose_frame_ids = []
        self.traj_xy = []
        self.last_motion_dxdy = np.array([0.0, 0.0], dtype=np.float32)
        self.last_motion_rot = 0.0
        
    def initialize(self):
        """Initialize the VO system."""
        # Load image paths
        self.paths = load_image_paths(
            str(Path(self.config["images_dir"]).expanduser()), 
            self.config["pattern"]
        )
        
        # Build detectors
        self.det = build_fast_detector(
            self.config["fast_threshold"], 
            self.config["fast_nonmax"], 
            self.config["fast_type"]
        )
        self.adapt = AdaptiveParams(self.config, self.det)
        
        # ORB fallback
        if self.config["use_orb_fallback"]:
            self.orb, self.bf = build_orb(self.config)
        
        # Read first frame and setup processing
        self._setup_first_frame()
        
    def _setup_first_frame(self):
        """Setup first frame and initialize processing pipeline."""
        # Read first frame (native) + sizes
        img0_full = read_image_any(self.paths[0], as_bgr=False)
        if img0_full is None:
            raise FileNotFoundError(f"Cannot read first image: {self.paths[0]}")
        H_full, W_full = img0_full.shape[:2]
        
        # Determine processing size
        resize_tuple = parse_resize(self.config["resize_to"]) if self.config["resize_to"] else None
        if resize_tuple:
            W0, H0 = resize_tuple
        else:
            s = float(self.config["PROCESS_SCALE"])
            if s <= 0 or s > 1:
                raise ValueError("PROCESS_SCALE must be in (0,1]")
            W0, H0 = (int(W_full * s), int(H_full * s)) if s < 1.0 else (W_full, H_full)
        
        # Undistorter (native)
        if self.config.get("use_undistort", False):
            self.dc = DistortionCorrector(
                W_full, H_full, self.config["fx"], self.config["fy"], 
                self.config["cx"], self.config["cy"], self.config["k1"], 
                self.config["k2"], self.config["p1"], self.config["p2"], 
                self.config.get("k3", 0.0)
            )
        
        # Store dimensions for later use
        self.W_full, self.H_full = W_full, H_full
        self.W0, self.H0 = W0, H0
        
        # Process first frame
        gray0_native = to_gray(img0_full).astype(np.uint8)
        if self.dc is not None:
            gray0_native = self.dc.apply_gray(gray0_native)
        gray_prev = cv2.resize(gray0_native, (W0, H0), interpolation=cv2.INTER_AREA) if (W0 != W_full or H0 != H_full) else gray0_native
        
        if self.config["use_clahe"]:
            gray_prev = apply_clahe(gray_prev, self.config["clahe_clip"], self.config["clahe_tilesize"])
        if self.config["use_unsharp"]:
            gray_prev = to_gray(unsharp(cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2BGR),
                                       self.config["unsharp_amount"], self.config["unsharp_sigma"]))
        
        # Initial features
        pts_prev = (detect_points_tile_FAST(self.det, gray_prev, self.config) if self.config["use_tile_fast"]
                    else detect_points_FAST(self.det, gray_prev, self.config))
        pts_prev = ensure_pts_fmt(pts_prev)
        if pts_prev is None or len(pts_prev) < self.config["min_tracks_for_solve"]:
            raise RuntimeError("Not enough FAST features in first frame.")
        
        # Initialize trajectory
        self.global_A = [np.hstack([np.eye(2, dtype=np.float32), np.zeros((2, 1), dtype=np.float32)])]
        self.pose_frame_ids = [0]
        cx0, cy0 = W0 / 2.0, H0 / 2.0
        center_pt = np.array([[[cx0, cy0]]], dtype=np.float32)
        pt0 = cv2.transform(center_pt, self.global_A[0])
        self.traj_xy = [(float(pt0[0, 0, 0]), float(pt0[0, 0, 1]))]
        
        # Store for main loop
        self.gray_prev = gray_prev
        self.pts_prev = pts_prev
        self.kf_idx = 0
        self.kf_gray = gray_prev.copy()
        self.kf_kp = self.kf_des = None
        if self.config["use_orb_fallback"]:
            self.kf_kp, self.kf_des = orb_kp_desc(self.orb, self.kf_gray)
    
    def process_frame(self, i):
        """Process a single frame."""
        t_start = time.perf_counter()
        
        # Load and preprocess image
        img_full = read_image_any(self.paths[i], as_bgr=False)
        if img_full is None:
            print(f"⚠️ Skip unreadable: {self.paths[i]}")
            return False
            
        gray_native = to_gray(img_full).astype(np.uint8)
        if self.dc is not None:
            gray_native = self.dc.apply_gray(gray_native)
        gray = cv2.resize(gray_native, (self.W0, self.H0), interpolation=cv2.INTER_AREA) if (self.W0 != self.W_full or self.H0 != self.H_full) else gray_native
        
        if self.config["use_clahe"]:
            gray = apply_clahe(gray, self.config["clahe_clip"], self.config["clahe_tilesize"])
        if self.config["use_unsharp"]:
            gray = to_gray(unsharp(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                                  self.config["unsharp_amount"], self.config["unsharp_sigma"]))
        
        # Track features
        success = self._track_features(gray, i)
        if not success:
            return False
        
        # Estimate motion
        motion_data = self._estimate_motion()
        if motion_data is None:
            return False
        
        inl, dx, dy, rot_deg, A_rigid = motion_data
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        
        print(f"[{i:04d}] inliers={inl:4d}  dpx=({dx:+.2f},{dy:+.2f})  rot={rot_deg:+.2f}°  {dt_ms:.1f} ms")
        self.pairs.append((i, inl, dx, dy, rot_deg, dt_ms))
        self.misses = 0
        self.adapt.adapt_after(inl)
        
        # Update global trajectory
        A_step_inv = invert_affine(A_rigid).astype(np.float32)
        A_k0 = compose_affine(self.global_A[-1], A_step_inv).astype(np.float32)
        self.global_A.append(A_k0)
        self.pose_frame_ids.append(i)
        
        cx0, cy0 = self.W0 / 2.0, self.H0 / 2.0
        center_pt = np.array([[[cx0, cy0]]], dtype=np.float32)
        pt_k = cv2.transform(center_pt, A_k0)
        self.traj_xy.append((float(pt_k[0, 0, 0]), float(pt_k[0, 0, 1])))
        
        self.last_motion_dxdy[:] = [dx, dy]
        self.last_motion_rot = rot_deg
        
        # Check for keyframe
        self._update_keyframe(i, gray)
        
        # Update for next iteration
        if self.p2 is not None and len(self.p2) >= 3:
            self.pts_prev = self.p2.reshape(-1, 1, 2).astype(np.float32)
        else:
            self.pts_prev = (detect_points_tile_FAST(self.det, gray, self.config) if self.config["use_tile_fast"]
                             else detect_points_FAST(self.det, gray, self.config))
            self.pts_prev = ensure_pts_fmt(self.pts_prev)
        self.gray_prev = gray
        
        return True
    
    def _track_features(self, gray, i):
        """Track features between frames."""
        self.pts_prev = ensure_pts_fmt(self.pts_prev)
        if self.pts_prev is None or len(self.pts_prev) < self.config["min_tracks_for_solve"]:
            self.pts_prev = (detect_points_tile_FAST(self.det, self.gray_prev, self.config) if self.config["use_tile_fast"]
                            else detect_points_FAST(self.det, self.gray_prev, self.config))
            self.pts_prev = ensure_pts_fmt(self.pts_prev)
            if self.pts_prev is None or len(self.pts_prev) < self.config["min_tracks_for_solve"]:
                old_thr = self.det.getThreshold()
                tmp_thr = max(self.config["fast_threshold_min"], int(old_thr * 0.8))
                try:
                    self.det.setThreshold(tmp_thr)
                    self.pts_prev = (detect_points_tile_FAST(self.det, self.gray_prev, self.config) if self.config["use_tile_fast"]
                                    else detect_points_FAST(self.det, self.gray_prev, self.config))
                    self.pts_prev = ensure_pts_fmt(self.pts_prev)
                finally:
                    self.det.setThreshold(old_thr)
                if self.pts_prev is None or len(self.pts_prev) < 3:
                    self.misses += 1
                    print(f"    ⚠️ no features on prev; miss#{self.misses} skip {i-1}->{i}")
                    self.gray_prev = gray
                    self.pts_prev = (detect_points_tile_FAST(self.det, gray, self.config) if self.config["use_tile_fast"]
                                     else detect_points_FAST(self.det, gray, self.config))
                    self.pts_prev = ensure_pts_fmt(self.pts_prev)
                    if self.config["use_orb_fallback"] and self.misses >= self.config["relocalize_after_misses"] and self.kf_gray is not None:
                        Hkf, inl = try_relocalize(self.orb, self.bf, self.kf_gray, self.kf_kp, self.kf_des, gray, self.config)
                        if Hkf is not None:
                            A_kf0 = self.global_A[self.kf_idx]
                            A_cur_kf = Hkf.astype(np.float32)
                            A_cur0 = compose_affine(A_kf0, invert_affine(A_cur_kf))
                            self.global_A.append(A_cur0)
                            self.pose_frame_ids.append(i)
                            cx0, cy0 = self.W0 / 2.0, self.H0 / 2.0
                            center_pt = np.array([[[cx0, cy0]]], dtype=np.float32)
                            pt_k = cv2.transform(center_pt, A_cur0)
                            self.traj_xy.append((float(pt_k[0, 0, 0]), float(pt_k[0, 0, 1])))
                            print(f"    ✅ relocalized to keyframe #{self.kf_idx} with {inl} inliers (frame {i})")
                            self.misses = 0
                            self.gray_prev = gray
                            self.pts_prev = (detect_points_tile_FAST(self.det, gray, self.config) if self.config["use_tile_fast"]
                                             else detect_points_FAST(self.det, gray, self.config))
                            self.pts_prev = ensure_pts_fmt(self.pts_prev)
                            return True
                    return False
        
        try:
            self.p1, self.p2, ntrk = track_lk(self.gray_prev, gray, self.pts_prev, self.adapt.lk_win, self.adapt.lk_levels)
        except cv2.error as e:
            print(f"    ⚠️ LK failed: {e}. Reinit on current.")
            self.misses += 1
            self.gray_prev = gray
            self.pts_prev = (detect_points_tile_FAST(self.det, gray, self.config) if self.config["use_tile_fast"]
                            else detect_points_FAST(self.det, gray, self.config))
            self.pts_prev = ensure_pts_fmt(self.pts_prev)
            return False
        
        if ntrk < self.config["min_tracks_for_solve"]:
            self.pts_prev = (detect_points_tile_FAST(self.det, self.gray_prev, self.config) if self.config["use_tile_fast"]
                            else detect_points_FAST(self.det, self.gray_prev, self.config))
            self.pts_prev = ensure_pts_fmt(self.pts_prev)
            if self.pts_prev is None or len(self.pts_prev) < 3:
                self.misses += 1
                print(f"    ⚠️ re-detect failed (prev); miss#{self.misses} skip {i-1}->{i}")
                self.gray_prev = gray
                self.pts_prev = (detect_points_tile_FAST(self.det, gray, self.config) if self.config["use_tile_fast"]
                                 else detect_points_FAST(self.det, gray, self.config))
                self.pts_prev = ensure_pts_fmt(self.pts_prev)
                return False
            self.p1, self.p2, ntrk = track_lk(self.gray_prev, gray, self.pts_prev,
                                              win=max(self.adapt.lk_win, self.config["lk_win_hard"]),
                                              levels=max(self.adapt.lk_levels, self.config["lk_levels_hard"]))
            if ntrk < 3:
                self.misses += 1
                print(f"    ⚠️ LK retry low ({ntrk}); miss#{self.misses} skip {i-1}->{i}")
                self.gray_prev = gray
                self.pts_prev = (detect_points_tile_FAST(self.det, gray, self.config) if self.config["use_tile_fast"]
                                 else detect_points_FAST(self.det, gray, self.config))
                self.pts_prev = ensure_pts_fmt(self.pts_prev)
                return False
        
        return True
    
    def _estimate_motion(self):
        """Estimate motion between tracked features."""
        inl = 0
        dx = dy = rot_deg = 0.0
        A_rigid = np.hstack([np.eye(2, dtype=np.float32), np.zeros((2, 1), dtype=np.float32)])
        tries = 0
        
        while True:
            A_sim, inliers_mask = cv2.estimateAffinePartial2D(
                self.p1, self.p2, method=cv2.RANSAC,
                ransacReprojThreshold=float(self.adapt.ransac_thr), confidence=0.999
            )
            
            if A_sim is not None:
                R2 = rigid_from_similarity(A_sim).astype(np.float32)
                t2 = fit_rigid_t(R2, self.p1, self.p2, inliers_mask).astype(np.float32)
                inl = int(inliers_mask.sum()) if inliers_mask is not None else 0
                A_rigid = np.hstack([R2, t2]).astype(np.float32)
                dx, dy = float(t2[0, 0]), float(t2[1, 0])
                rot_deg = math.degrees(math.atan2(R2[1, 0], R2[0, 0]))
            
            if inl < self.config["adapt_retrack_min_inliers"] and tries < int(self.config["max_retries_per_pair"]):
                tries += 1
                self.adapt.fast_thr *= float(self.config["adapt_down_factor"])
                self.adapt._clamp_fast()
                self.adapt.soften()
                print(f"    ↺ retry {tries}: FAST={int(self.adapt.fast_thr)}, LKwin={self.adapt.lk_win}, "
                      f"levels={self.adapt.lk_levels}, RANSAC={self.adapt.ransac_thr:.2f}")
                self.pts_prev = (detect_points_tile_FAST(self.det, self.gray_prev, self.config) if self.config["use_tile_fast"]
                                 else detect_points_FAST(self.det, self.gray_prev, self.config))
                self.pts_prev = ensure_pts_fmt(self.pts_prev)
                if self.pts_prev is None or len(self.pts_prev) < 3:
                    break
                self.p1, self.p2, ntrk = track_lk(self.gray_prev, self.gray_prev, self.pts_prev, 
                                                  self.adapt.lk_win, self.adapt.lk_levels)
                continue
            break
        
        if inl < self.config["adapt_retrack_min_inliers"]:
            return None
        
        return (inl, dx, dy, rot_deg, A_rigid)
    
    def _update_keyframe(self, i, gray):
        """Update keyframe if needed."""
        need_kf = ((i - self.kf_idx) >= int(self.config["kf_every_n"]) or
                   abs(self.last_motion_rot) >= float(self.config["kf_min_rot_deg"]) or
                   np.hypot(*self.last_motion_dxdy) >= float(self.config["kf_min_trans_px"]))
        
        if need_kf:
            self.kf_idx = i
            self.kf_gray = gray.copy()
            if self.config["use_orb_fallback"]:
                self.kf_kp, self.kf_des = orb_kp_desc(self.orb, self.kf_gray)
    
    def run(self):
        """Run the complete VO algorithm."""
        t0_all = time.perf_counter()
        
        for i in range(1, len(self.paths)):
            success = self.process_frame(i)
            if not success:
                continue
        
        print(f"Done: {len(self.paths)} frames in {time.perf_counter() - t0_all:.2f}s")
        return self.pairs, self.global_A, self.pose_frame_ids, self.traj_xy
