"""
Feature detection and matching utilities for Visual Odometry.
Handles FAST, ORB, and adaptive feature detection.
"""

import numpy as np
import cv2


def build_fast_detector(threshold, nonmax, type_str):
    """Build FAST feature detector with specified parameters."""
    type_map = {
        "TYPE_5_8": cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
        "TYPE_7_12": cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
        "TYPE_9_16": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    }
    det = cv2.FastFeatureDetector_create(
        threshold=int(threshold),
        nonmaxSuppression=bool(nonmax),
        type=type_map.get(type_str.upper(), cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    )
    return det


def set_fast_threshold(det, thr):
    """Set FAST detector threshold."""
    det.setThreshold(int(thr))


def ensure_pts_fmt(pts):
    """Ensure points are in correct format for OpenCV."""
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 2 and pts.shape[1] == 2:
        pts = pts.reshape(-1, 1, 2)
    elif pts.ndim != 3 or pts.shape[1:] != (1, 2):
        return None
    return pts if len(pts) >= 3 else None


def detect_points_FAST(det, gray, cfg):
    """Detect FAST features in image."""
    kps = det.detect(gray, None)
    if not kps or len(kps) < 3:
        return None
    pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 1, 2)
    
    if cfg["refine_subpix"]:
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, cfg["subpix_iter"], cfg["subpix_eps"])
        win = (cfg["subpix_win"], cfg["subpix_win"])
        cv2.cornerSubPix(gray, pts, win, (-1, -1), term)
    
    return pts


def detect_points_tile_FAST(det, gray, cfg):
    """Detect FAST features using tiled approach for better distribution."""
    H, W = gray.shape[:2]
    gx, gy = cfg["tiles_xy"]
    quota = int(cfg["tile_quota"])
    relax = float(cfg["tile_fast_relax"])
    pts_all = []
    base_thr = det.getThreshold()
    
    for yi in range(gy):
        for xi in range(gx):
            x0 = int(round(W * xi / gx))
            x1 = int(round(W * (xi + 1) / gx))
            y0 = int(round(H * yi / gy))
            y1 = int(round(H * (yi + 1) / gy))
            roi = gray[y0:y1, x0:x1]
            
            thr = base_thr
            det.setThreshold(int(thr))
            kps = det.detect(roi, None)
            
            if (not kps or len(kps) < quota // 2) and base_thr > cfg["fast_threshold_min"]:
                thr = max(cfg["fast_threshold_min"], int(base_thr * relax))
                det.setThreshold(int(thr))
                kps = det.detect(roi, None)
            
            if kps:
                kps = sorted(kps, key=lambda k: -k.response)[:quota]
                pts = np.array([[[k.pt[0] + x0, k.pt[1] + y0]] for k in kps], dtype=np.float32)
                pts_all.append(pts)
    
    det.setThreshold(int(base_thr))
    if not pts_all:
        return None
    
    pts = np.vstack(pts_all)
    if cfg["refine_subpix"]:
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, cfg["subpix_iter"], cfg["subpix_eps"])
        win = (cfg["subpix_win"], cfg["subpix_win"])
        cv2.cornerSubPix(gray, pts, win, (-1, -1), term)
    
    return pts


def build_orb(cfg):
    """Build ORB detector and matcher for fallback/relocalization."""
    orb = cv2.ORB_create(
        nfeatures=int(cfg["orb_nfeatures"]),
        scaleFactor=float(cfg["orb_scaleFactor"]),
        nlevels=int(cfg["orb_nlevels"]),
        edgeThreshold=31,
        patchSize=31,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE
    )
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return orb, bf


def orb_kp_desc(orb, gray):
    """Extract ORB keypoints and descriptors."""
    return orb.detectAndCompute(gray, None)


def orb_match_and_estimate(bf, des_ref, des_cur, kp_ref, kp_cur, ransac_thresh_px):
    """Match ORB descriptors and estimate transformation."""
    if des_ref is None or des_cur is None or len(des_ref) == 0 or len(des_cur) == 0:
        return None, 0
    
    knn = bf.knnMatch(des_ref, des_cur, k=2)
    good = []
    for m, n in knn:
        if m.distance < 0.8 * n.distance:  # bf_max_ratio
            good.append(m)
    
    if len(good) < 8:
        return None, 0
    
    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, mask = cv2.estimateAffinePartial2D(
        pts_cur, pts_ref, method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh_px), confidence=0.999
    )
    
    inl = int(mask.sum()) if mask is not None else 0
    return H, inl


def try_relocalize(orb, bf, kf_gray, kf_kp, kf_des, cur_gray, cfg):
    """Attempt relocalization using ORB features."""
    if orb is None or bf is None:
        return (None, 0)
    
    if kf_kp is None or kf_des is None:
        kf_kp, kf_des = orb_kp_desc(orb, kf_gray)
        if kf_kp is None:
            return (None, 0)
    
    kp_cur, des_cur = orb_kp_desc(orb, cur_gray)
    if kp_cur is None:
        return (None, 0)
    
    H, inl = orb_match_and_estimate(bf, kf_des, des_cur, kf_kp, kp_cur, float(cfg["relocalize_ransac_thresh"]))
    return (H.astype(np.float32) if H is not None else None, inl)
