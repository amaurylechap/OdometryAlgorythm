"""
Image processing utilities for Visual Odometry.
Handles image loading, preprocessing, and distortion correction.
"""

import os
import re
import glob
import numpy as np
import cv2


def natural_key(s):
    """Natural sorting key for file names."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def load_image_paths(folder, pattern):
    """Load image paths from folder with natural sorting."""
    exts = [pattern] if pattern else ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.pgm", "*.ppm"]
    paths = []
    [paths.extend(glob.glob(os.path.join(folder, ext))) for ext in exts]
    paths = sorted(paths, key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    return paths


def parse_resize(resize_str):
    """Parse resize string like '960x540' into tuple."""
    if resize_str is None:
        return None
    m = re.match(r"^(\d+)x(\d+)$", resize_str)
    if not m:
        raise ValueError("resize_to must be 'WxH' or None")
    return (int(m.group(1)), int(m.group(2)))


def to_gray(img):
    """Convert image to grayscale if needed."""
    return img if (img.ndim == 2) else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def read_image_any(path, as_bgr=False):
    """Robust image reader that handles various formats including PGM."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    # Handle 16-bit images
    if img.dtype == np.uint16:
        maxv = int(img.max())
        img = cv2.convertScaleAbs(img, alpha=255.0/max(1, maxv)) if maxv > 0 else np.zeros_like(img, dtype=np.uint8)
    
    if as_bgr:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    return img


def ensure_bgr(img):
    """Ensure image is in BGR format."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def apply_clahe(gray, clip, tilesize):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tilesize, tilesize))
    return clahe.apply(gray)


def unsharp(img, amount=1.0, sigma=1.0):
    """Apply unsharp mask filter."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(sigma))
    return cv2.addWeighted(img, 1.0 + float(amount), blur, -float(amount), 0)


class DistortionCorrector:
    """Handles camera distortion correction using precomputed maps."""
    
    def __init__(self, cam_w, cam_h, fx, fy, cx, cy, k1, k2, p1, p2, k3=0.0,
                 interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
        self.W, self.H = int(cam_w), int(cam_h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=self.K, distCoeffs=self.dist, R=np.eye(3, dtype=np.float32),
            newCameraMatrix=self.K, size=(self.W, self.H), m1type=cv2.CV_32FC1
        )
    
    def apply_gray(self, gray_u8):
        """Apply distortion correction to grayscale image."""
        return cv2.remap(gray_u8, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def apply_bgr(self, bgr_u8):
        """Apply distortion correction to BGR image."""
        return cv2.remap(bgr_u8, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
