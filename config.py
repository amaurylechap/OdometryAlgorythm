"""
Configuration module for Visual Odometry algorithm.
Contains all user-configurable parameters and settings.
"""

# =========================
# ===== USER CONFIG =======
# =========================
CONFIG = {
    # ---- I/O ----
    "images_dir": r"C:\Users\amaur\Desktop\frames_flight\Flight_26_09_2025\frames_20250926_135944",
    "pattern": "*.pgm",  # set to "*.png" or "" to auto-match many

    # ---- CSV paths (point these to your actual files) ----
    "PATH_IMU_CSV": "inputs/IMU_Timestamp.csv",        # epoch or parseable datetime; includes GPS
    "PATH_FRAME_CSV": "inputs/FrameId_Timestamp.csv",  # frame id (or filename), timestamp
    "TIME_OFFSET_USER_S": -6,  # user tweak added to the auto offset

    # ---- Frame/filename handling ----
    # Regex to extract frame index from filenames like 'frame_000001.pgm'
    "FRAME_FILENAME_REGEX": r"frame_(\d+)\.",
    # If True, filter the frame CSV down to only frames that exist in images_dir/pattern
    "CLIP_FRAME_CSV_TO_IMAGES": True,
    
    # ---- VO trajectory rotation ----
    "VO_ROTATION_DEG": 5,  # User-defined rotation angle in degrees (0 = no rotation)
    
    # ---- Heading source (camera == VO in-plane rotation) ----
    "HEADING_SOURCE": "camera",   # only use camera heading (derived from VO rotations)
        # Body-frame sign conventions (aircraft axes: +Right, +Forward)
        "COMP_SIGN_ROLL_TO_RIGHT": -1.0,   # +roll (right wing down) => footprint shifts to +Right
        "COMP_SIGN_PITCH_TO_FWD":  1,    # +pitch (nose up) => footprint shifts aft (disabled)
    "ENABLE_TILT_COMP": True,

    # ---- Processing size for VO ----
    "PROCESS_SCALE": 0.5,        # 0.4–0.6 is fast & accurate
    "resize_to": None,           # e.g. "960x540"; overrides PROCESS_SCALE

    # ---- Undistortion (put your intrinsics at NATIVE sensor size) ----
    "use_undistort": True,
    "fx": 800.0, "fy": 800.0,
    "cx": 640.0, "cy": 360.0,
    "k1": -0.13, "k2": 0.012, "p1": 0.0, "p2": 0.0, "k3": 0.0,

    # ---- PREPROC ----
    "use_clahe": True, "clahe_clip": 2.0, "clahe_tilesize": 8,
    "use_unsharp": True, "unsharp_amount": 1.0, "unsharp_sigma": 1.0,

    # ---- FAST / LK / RANSAC ----
    "fast_threshold": 6, "fast_nonmax": True, "fast_type": "TYPE_5_8",
    "min_tracks_for_solve": 80,
    "use_tile_fast": True, "tiles_xy": (4, 3), "tile_quota": 150, "tile_fast_relax": 0.75,
    "lk_win": 21, "lk_levels": 3, "ransac_thresh_px": 1.6,

    # ---- Subpixel ----
    "refine_subpix": True, "subpix_win": 5, "subpix_iter": 20, "subpix_eps": 0.01,

    # ---- Adaptive ----
    "adapt_inliers_target": 140, "adapt_retrack_min_inliers": 50, "max_retries_per_pair": 2,
    "fast_threshold_min": 5, "fast_threshold_max": 60, "adapt_down_factor": 0.8, "adapt_up_factor": 1.25,
    "lk_win_soft": 21, "lk_win_hard": 31, "lk_levels_soft": 3, "lk_levels_hard": 4,
    "ransac_thresh_easy": 1.3, "ransac_thresh_hard": 2.0,

    # ---- ORB fallback ----
    "use_orb_fallback": True, "orb_nfeatures": 4000, "orb_scaleFactor": 1.2, "orb_nlevels": 8,
    "bf_max_ratio": 0.8, "relocalize_after_misses": 5, "relocalize_ransac_thresh": 3.0,

    # ---- Keyframes ----
    "kf_every_n": 15, "kf_min_rot_deg": 2.0, "kf_min_trans_px": 25.0,

    # ---- Mosaic (disabled by default per your request) ----
    "ENABLE_MOSAIC": True,
    "MOSAIC_MODE": "thumbnail",   # "outline" | "thumbnail" | "image"
    "MOSAIC_SRC_WARP_SCALE": 0.5,
    "MOSAIC_RENDER_SCALE": 0.6,
    "MOSAIC_STRIDE": 100,
    "alpha": 0.40,
    "canvas_margin": 200,
    "mosaic_png": "outputs/mosaic.png",
    "mosaic_with_traj_png": "outputs/mosaic_traj.png",

    # ---- Plots / logs ----
    "save_csv":   "outputs/pairs_log.csv",
    "traj_png_xy": "outputs/traj_xy.png",
    "plot_flip_y": True,   # for traj_xy plot only
    "plot_center": True,

    # ---- VO→meters scale & geography ----
    # You wanted a simple user factor. Use this directly:
    "SCALE_M_PER_PX": None,   # meters per pixel (set this!)
    # If None, we can compute ALTITUDE / LENS_FACTOR instead:
    "ALTITUDE_M": 205,
    "LENS_FACTOR": 500,    # arbitrary number so that ALTITUDE/LENS ~ meters/pixel
    # Image→ENU axis mapping (typical image y is downward)
    "IMAGE_X_TO_EAST": +1.0,
    "IMAGE_Y_TO_NORTH": -1.0,   # -1 flips y

    # ---- Outputs for VO metric plots ----
    "vo_metric_no_imu_png": "outputs/vo_metric_no_imu.png",
    "vo_metric_with_imu_png": "outputs/vo_metric_with_imu.png",
    "gps_plot_png": "outputs/gps_track.png",
    "imu_plot_png": "outputs/imu_angles.png",
    "vo_vs_gps_png": "outputs/vo_vs_gps.png",  # NEW: combined plot

    # ---- Optional CSV outputs ----
    "save_vo_no_imu_latlon_csv": "outputs/vo_no_imu_latlon.csv",
    "save_vo_with_imu_latlon_csv": "outputs/vo_with_imu_latlon.csv",

    # ---- Debug sync CSV ----
    "WRITE_SYNC_CSV": True,
    "SYNC_CSV_PATH": "outputs/debug/vo_gps_synced_new.csv",
    
    # ---- Plane-frame series outputs ----
    "PLANE_OUTPUTS_ENABLED": True,
    "PLANE_SERIES_CSV": "outputs/plane_series.csv",
    "PLANE_SERIES_PNG": "outputs/plane_series.png",
    
    # ---- Lateral analysis options ----
    "LATERAL_ANALYSIS_SHOW_MPS": False,  # Show m/s alongside m/frame in lateral analysis
    
    # ---- VO heading alignment (at t=0) ----
    # Align the VO/camera heading so that VO_heading(t0) == GPS_heading(t0) + USER_OFFSET
    "HEADING_ALIGN_MODE": "gps0",      # "gps0" or "none"
    "HEADING_OFFSET_USER_DEG": 0.0,    # extra user tweak (degrees), applied after gps0 alignment
    
    # ===== Minimal outputs (only mosaic + plane series) =====
    "MINIMAL_OUTPUTS": True,                 # master switch to hide everything else
    "PLANE_OUTPUTS_ENABLED": True,
    "PLANE_SERIES_PNG": "outputs/plane_series.png",
    "PLANE_SERIES_CSV": None,               # set to path if you still want the CSV, else None

        # Mosaic on (choose your mode/stride as you like)
        "ENABLE_MOSAIC": True,
    "MOSAIC_MODE": "thumbnail",              # "outline" | "thumbnail" | "image"
    "MOSAIC_RENDER_SCALE": 0.5,
    "MOSAIC_STRIDE": 5,
    "mosaic_png": "outputs/mosaic.png",      # (used by generate_mosaic)
    "mosaic_with_traj_png": "outputs/mosaic_traj.png",

    # Optional: prevent saving legacy CSV
    "save_csv": None,                        # disable pairs_log.csv
        
    # VO → plane-frame mapping (which image axis is "forward" for your aircraft)
    # If your forward is image +Y (typical nadir cam, y down), set:
    "VO_FORWARD_AXIS": "y",           # "y" or "x"
    "SIGN_Y_TO_FORWARD": +1.0,        # multiply VO Δy (in meters) by this to get +Forward
    "SIGN_X_TO_LATERAL": +1.0,        # multiply VO Δx (in meters) by this to get +Lateral (right)
    
    # Plane-frame axis signs (Right/Forward). Only use these, remove ad-hoc *-1.
    "PLANE_SIGN_FWD": +1.0,           # flip if your forward is inverted
    "PLANE_SIGN_LAT": -1.0,           # flip if your lateral (Right) is inverted
    
    # Align VO heading to GPS at t0 + user tweak
    "HEADING_ALIGN_MODE": "gps0",     # "gps0" | "none"
    "HEADING_OFFSET_USER_DEG": 0.0,   # user tweak
}
