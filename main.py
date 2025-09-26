#!/usr/bin/env python3
"""
Main entry point for Visual Odometry algorithm.
Orchestrates the VO pipeline, CSV loading, time synchronization, and plotting.
"""

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from vo_algorithm import VisualOdometry
from visualization import (
    plot_trajectory_xy, plot_gps_track, plot_imu_angles,
    plot_vo_metric_no_imu, plot_vo_metric_with_imu, generate_mosaic
)
from csv_utils import load_frame_csv, load_imu_csv, en_to_latlon, latlon_to_en
from imu_compensation import compensate_positions_body_to_en
from config import CONFIG

def en_steps_to_body(dE_step, dN_step, yaw_rad_step):
    """
    Project EN step vectors to aircraft body axes (Forward, Right)
    using the camera/VO heading at that step (yaw about vertical).
    Mapping (Fwd, Right) -> (N,E):
        [N;E] = [[cosψ, -sinψ],[sinψ, cosψ]] @ [Fwd; Right]
    Inverse to get (Fwd, Right) from (N,E):
        [Fwd; Right] = [[cosψ, sinψ],[-sinψ, cosψ]] @ [N;E]
    """
    c = np.cos(yaw_rad_step); s = np.sin(yaw_rad_step)
    fwd  = c * dN_step + s * dE_step
    right = -s * dN_step + c * dE_step
    return fwd, right

def angle_deg_from_steps(dE_step, dN_step):
    """Heading/course (deg) of each EN step (atan2(E, N))."""
    ang = np.degrees(np.arctan2(dE_step, dN_step))
    return ang

def _unwrap_deg(a_deg):
    """Unwrap degrees to avoid ±180° jumps."""
    import numpy as np
    return np.degrees(np.unwrap(np.deg2rad(a_deg)))

def _course_deg_from_steps(dE_step, dN_step):
    """Course angle of each EN step (atan2(E, N)) in degrees."""
    import numpy as np
    return np.degrees(np.arctan2(dE_step, dN_step))

def main():
    cfg = CONFIG

    # ---- 1) Run VO exactly as-is ----
    vo = VisualOdometry(cfg)
    vo.initialize()
    pairs, global_A, pose_frame_ids, traj_xy = vo.run()
    
    # Only plot trajectory if not in minimal mode
    if not cfg.get("MINIMAL_OUTPUTS", False):
        plot_trajectory_xy(traj_xy, cfg)
    
    # ---- Minimal outputs: Mosaic + Plane series only ----
    if cfg.get("MINIMAL_OUTPUTS", False):
        # Generate mosaic if enabled
        if cfg.get("ENABLE_MOSAIC", False):
            generate_mosaic(vo.global_A, vo.pose_frame_ids, vo.paths, 
                           vo.W0, vo.H0, vo.W_full, vo.H_full, vo.dc, cfg, traj_xy)
        
        # Generate plane series plot if enabled
        if cfg.get("PLANE_OUTPUTS_ENABLED", False):
            # Compute plane-frame series using proper aircraft reference system
            # Use basic trajectory data for minimal computation
            scale = float(cfg["SCALE_M_PER_PX"] if cfg.get("SCALE_M_PER_PX") is not None
                          else float(cfg["ALTITUDE_M"]) / float(cfg["LENS_FACTOR"]))
            Tpix = np.array(traj_xy, dtype=np.float32)
            Trel = Tpix - Tpix[0]
            sx = float(cfg["IMAGE_X_TO_EAST"]) * scale
            sy = float(cfg["IMAGE_Y_TO_NORTH"]) * scale
            pos_raw_m = np.column_stack([Trel[:, 0] * sx, Trel[:, 1] * sy])
            
            # VO step rotations (deg) → cumulative heading (rad)
            yaw_steps_deg = np.array([row[4] for row in pairs], dtype=float)
            yaw_steps_rad = np.deg2rad(yaw_steps_deg)
            yaw_pose = np.zeros(len(pos_raw_m), dtype=float)
            yaw_pose[1:] = np.cumsum(yaw_steps_rad)
            
            # Apply constant offset (simplified for minimal - no GPS alignment)
            user_off_rad = np.deg2rad(float(cfg.get("HEADING_OFFSET_USER_DEG", 0.0)))
            yaw_pose_aligned = yaw_pose + user_off_rad
            yaw_step_used = yaw_pose_aligned[:-1]  # heading at start of each step
            
            # Compute EN steps
            dE_vo = np.diff(pos_raw_m[:, 0])
            dN_vo = np.diff(pos_raw_m[:, 1])
            
            # Project EN→body using aligned heading
            fwd_vo_step, right_vo_step = en_steps_to_body(dE_vo, dN_vo, yaw_step_used)
            
            # SWAP forward and lateral - the axes are inverted!
            fwd_vo_step, right_vo_step = right_vo_step, fwd_vo_step
            
            # Apply only explicit config signs (if needed)
            fwd_vo_step = cfg.get("PLANE_SIGN_FWD", +1.0) * fwd_vo_step
            right_vo_step = cfg.get("PLANE_SIGN_LAT", +1.0) * right_vo_step
            
            # Cumulative forward and lateral
            fwd_vo_cum = np.concatenate([[0.0], np.cumsum(fwd_vo_step)])
            lat_vo_cum = np.concatenate([[0.0], np.cumsum(right_vo_step)])
            
            # Rotation per step for VO (deg). Use raw step rotations
            rot_vo_step_deg = yaw_steps_deg.copy()
            
            # Cumulative heading for the plot
            vo_heading_cum_deg = np.concatenate([[0.0], np.cumsum(rot_vo_step_deg)])
            
            # Time arrays - match the actual data dimensions
            time_rel = np.arange(len(yaw_pose)) * 0.1  # Assume 10fps, match yaw_pose length
            t_step_time = time_rel[1:]
            frame_nums = np.arange(len(yaw_pose))
            step_frame_nums = frame_nums[1:]
            
            # Plot plane series
            from visualization import plot_plane_series
            plot_plane_series(time_rel, frame_nums,
                              fwd_vo_cum, lat_vo_cum, vo_heading_cum_deg,
                              t_step_time, step_frame_nums,
                              out_png=cfg["PLANE_SERIES_PNG"])
        
        print("✅ Minimal outputs complete!")
        return

    # ---- 2) Load CSVs (if provided) ----
    have_frame_csv = isinstance(cfg.get("PATH_FRAME_CSV"), str) and os.path.isfile(cfg["PATH_FRAME_CSV"])
    have_imu_csv   = isinstance(cfg.get("PATH_IMU_CSV"), str) and os.path.isfile(cfg["PATH_IMU_CSV"])
    if not (have_frame_csv and have_imu_csv):
        print("ℹ️ CSVs not fully available — will plot VO only (no IMU/GPS).")
        # still save VO in meters relative
        scale = float(cfg["SCALE_M_PER_PX"] if cfg.get("SCALE_M_PER_PX") is not None
                      else float(cfg["ALTITUDE_M"]) / float(cfg["LENS_FACTOR"]))
        Tpix = np.array(traj_xy, dtype=np.float32)
        Trel = Tpix - Tpix[0]
        sx = float(cfg["IMAGE_X_TO_EAST"]) * scale
        sy = float(cfg["IMAGE_Y_TO_NORTH"]) * scale
        pos_noimu_m = np.column_stack([Trel[:, 0] * sx, Trel[:, 1] * sy])
        if not cfg.get("MINIMAL_OUTPUTS", False):
            plot_vo_metric_no_imu(pos_noimu_m, None, None, {}, cfg)
        print("✅ Visual Odometry processing complete!")
        return

    # ---- 3) Frame CSV: keep only the frames we actually processed ----
    frame_df_all = load_frame_csv(cfg["PATH_FRAME_CSV"],
                                  filename_regex=cfg.get("FRAME_FILENAME_REGEX", None),
                                  allow_filename_column=True)

    # Extract image-set frame indices from VO (by filenames)
    import re
    rx = re.compile(cfg.get("FRAME_FILENAME_REGEX", r"frame_(\d+)\."))
    img_paths = vo.paths
    img_frames = []
    for p in img_paths:
        m = rx.search(os.path.basename(p))
        if not m:
            raise ValueError(f"Cannot extract frame id from filename: {p}")
        img_frames.append(int(m.group(1)))
    img_frames = np.array(img_frames, dtype=int)

    if cfg.get("CLIP_FRAME_CSV_TO_IMAGES", True):
        frame_df = frame_df_all[frame_df_all["frame"].isin(img_frames)].copy()
    else:
        frame_df = frame_df_all.copy()

    frame_df = frame_df.sort_values("frame").reset_index(drop=True)
    if frame_df.empty:
        raise RuntimeError("After clipping, frame CSV has no rows matching the images subset.")

    # ---- 4) IMU CSV (with GPS), sorted ----
    imu_df = load_imu_csv(cfg["PATH_IMU_CSV"])
    
    # Fix: Convert IMU timestamps from milliseconds to seconds
    imu_df["t_s"] = imu_df["t_s"] * 0.001  # ms → s
    print(f"🔧 IMU timestamps converted: {imu_df['t_s'].iloc[0]:.6f} .. {imu_df['t_s'].iloc[-1]:.6f} s (span: {imu_df['t_s'].iloc[-1] - imu_df['t_s'].iloc[0]:.1f} s)")

    # ---- 5) Pose times by interpolating the per-frame timestamps ----
    # frames_all: from frame_df["frame"] (e.g., 1001..1600)
    # times_all : from frame_df["t_s"]   (epoch seconds)
    frames_all = frame_df["frame"].values.astype(float)
    times_all  = frame_df["t_s"].values.astype(float)

    # pose_frame_ids are indices into vo.paths (0..N-1), NOT filename numbers
    pose_indices = np.array(pose_frame_ids, dtype=int)       # 0..N-1
    pose_frame_nums = img_frames[pose_indices].astype(float) # map to e.g. 1001..1600

    # Interpolate timestamps AT those filename frame numbers
    pose_t = np.interp(pose_frame_nums, frames_all, times_all)
    if np.any(np.isnan(pose_t)):
        print("⚠️ Some pose timestamps are NaN after interpolation (check frame CSV coverage).")

    # ---- 6) Compute time offset (auto + user) ----
    # auto: align IMU start to first pose timestamp => offset to apply to IMU so that imu_t + off ~ pose_t
    auto_off = float(pose_t[0] - imu_df["t_s"].iloc[0])
    user_off = float(cfg.get("TIME_OFFSET_USER_S", 0.0))
    final_off = auto_off + user_off
    print(f"⏱  Auto time offset: {auto_off:.6f} s; +user={user_off:+.6f} => USING {final_off:.6f} s")

    # ---- 7) Query IMU/GPS at the pose times ----
    tq = pose_t - final_off  # IMU time axis
    imu_t = imu_df["t_s"].values

    # Optional sanity prints
    print(f"Pose frames: idx[0]={pose_indices[0]} -> file_frame={pose_frame_nums[0]} -> t={pose_t[0]:.6f}")
    print(f"IMU t range: {imu_df['t_s'].iloc[0]:.6f} .. {imu_df['t_s'].iloc[-1]:.6f}")
    print(f"Query tq    : {tq[0]:.6f} .. {tq[-1]:.6f} (IMU domain after offset)")

    # Interpolate roll/pitch (fallback to edges)
    roll_pose  = np.interp(tq, imu_t, imu_df["roll_rad"].values,
                           left=imu_df["roll_rad"].values[0], right=imu_df["roll_rad"].values[-1])
    pitch_pose = np.interp(tq, imu_t, imu_df["pitch_rad"].values,
                           left=imu_df["pitch_rad"].values[0], right=imu_df["pitch_rad"].values[-1])

    # GPS: prefer lat/lon if available
    lat_pose = lon_pose = None
    gps_lat0 = gps_lon0 = None
    if "lat" in imu_df.columns and "lon" in imu_df.columns:
        lat_pose = np.interp(tq, imu_t, imu_df["lat"].values,
                             left=imu_df["lat"].values[0], right=imu_df["lat"].values[-1])
        lon_pose = np.interp(tq, imu_t, imu_df["lon"].values,
                             left=imu_df["lon"].values[0], right=imu_df["lon"].values[-1])
        gps_lat0 = float(lat_pose[0]); gps_lon0 = float(lon_pose[0])
    elif "E" in imu_df.columns and "N" in imu_df.columns:
        # If only EN, use (0,0) as origin and compare in meters
        gps_lat0 = gps_lon0 = None  # handled in plots

    # ---- 8) VO → meters (simple user scale or ALTITUDE/LENS fallback) ----
    scale = cfg.get("SCALE_M_PER_PX", None)
    if scale is None:
        scale = float(cfg["ALTITUDE_M"]) / float(cfg["LENS_FACTOR"])
    scale = float(scale)
    sx = float(cfg["IMAGE_X_TO_EAST"]) * scale
    sy = float(cfg["IMAGE_Y_TO_NORTH"]) * scale

    Tpix = np.array(traj_xy, dtype=np.float32)
    Trel = Tpix - Tpix[0]
    pos_raw_m = np.column_stack([Trel[:, 0] * sx, Trel[:, 1] * sy])

    # ---- 8.5) Apply user-defined VO rotation (if specified) ----
    vo_rotation_deg = cfg.get("VO_ROTATION_DEG", 0.0)
    if vo_rotation_deg != 0.0:
        print(f"🔄 Applying VO rotation: {vo_rotation_deg:.2f}°")
        theta = np.deg2rad(vo_rotation_deg)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        # Apply rotation to VO trajectory
        vo_rotated = np.column_stack([
            cos_theta * pos_raw_m[:, 0] - sin_theta * pos_raw_m[:, 1],
            sin_theta * pos_raw_m[:, 0] + cos_theta * pos_raw_m[:, 1]
        ])
        pos_raw_m = vo_rotated

    # ---- 9) Camera heading from VO rotations ----
    # VO provides per-step rotations in degrees in `pairs` (row[4] if that's rot_deg).
    # Accumulate them to get per-pose heading (yaw about camera optical axis).
    yaw_steps_deg = np.array([row[4] for row in pairs], dtype=float)  # shape (N-1,)
    yaw_steps_rad = np.deg2rad(yaw_steps_deg)

    # Build per-pose yaw (length N): yaw[0] = 0; yaw[k+1] = yaw[k] + yaw_steps[k]
    yaw_pose = np.zeros(len(pos_raw_m), dtype=float)
    if len(yaw_steps_rad) == len(yaw_pose) - 1:
        yaw_pose[1:] = np.cumsum(yaw_steps_rad)
    else:
        print("⚠️ VO rotation/pose length mismatch; heading set to zeros")

    # Optional: if your image axes aren't aligned to aircraft axes at yaw=0,
    # you can add a fixed bias here (e.g., camera mounted rotated by 90°):
    # yaw_pose += np.deg2rad(cfg.get("CAM_YAW_BIAS_DEG", 0.0))

    # ---- Derive GPS heading at t=0 (course of the first segment) ----
    gps_heading0_rad = None
    if (cfg.get("HEADING_ALIGN_MODE", "gps0").lower() == "gps0") and \
       (lat_pose is not None) and (lon_pose is not None) and (len(lat_pose) >= 2):
        # Convert GPS to local EN meters around the first GPS fix (pose times already synced)
        E_gps, N_gps = latlon_to_en(lat_pose[0], lon_pose[0], lat_pose, lon_pose)
        dE0 = E_gps[1] - E_gps[0]
        dN0 = N_gps[1] - N_gps[0]
        if (abs(dE0) + abs(dN0)) > 1e-6:
            gps_heading0_rad = np.arctan2(dE0, dN0)   # course: atan2(E, N)
        else:
            print("⚠️ GPS first step is too small to define a heading; skipping auto alignment.")

    # ---- Compute and apply VO heading offset ----
    user_off_deg = float(cfg.get("HEADING_OFFSET_USER_DEG", 0.0))
    user_off_rad = np.deg2rad(user_off_deg)

    if gps_heading0_rad is not None:
        # VO initial heading is yaw_pose[0] (currently 0 by construction)
        # We want: yaw_pose_aligned[0] = gps_heading0 + user_offset
        # Apply -1 factor to match the heading display
        delta_rad = (gps_heading0_rad + user_off_rad) * -1.0 - yaw_pose[0]
    else:
        # No GPS heading available -> just use user offset
        delta_rad = user_off_rad * -1.0

    # Apply constant offset to ALL poses
    yaw_pose_aligned = yaw_pose + delta_rad

    # For per-step projections, use heading at the start of each step
    yaw_step_used = yaw_pose_aligned[:-1]    # (N-1,)
    print(f"🧭 Heading alignment: mode={cfg.get('HEADING_ALIGN_MODE')} | "
          f"gps0_deg={np.degrees(gps_heading0_rad) if gps_heading0_rad is not None else 'None'} | "
          f"user_off_deg={user_off_deg:+.3f} | applied_offset_deg={np.degrees(delta_rad):+.3f}")

    # ---- 10) Unified reference frame: rebase all trajectories to VO(IMU) origin ----
    
    # Keep VO(no-IMU) at (0,0) - already done via Trel = traj_xy - traj_xy[0]
    # pos_noimu_m: (N,2) EN meters, with pos_noimu_m[0] == [0,0]
    
    # Compute VO(IMU) in the same meter space (per-pose, non-integrating)
    pos_comp_pose_m = None
    if cfg.get("ENABLE_TILT_COMP", True):
        # Apply IMU compensation to raw trajectory
        pos_comp_pose_m = compensate_positions_body_to_en(
            pos_noimu_EN_m=pos_raw_m,               # same meter space as baseline
            roll_rad_series=roll_pose,             # radians
            pitch_rad_series=pitch_pose,           # radians
            heading_rad_series=yaw_pose,           # radians, from VO
            altitude_m=float(cfg.get("ALTITUDE_M", 100.0)),
            sign_roll_to_right=float(cfg.get("COMP_SIGN_ROLL_TO_RIGHT", +1.0)),
            sign_pitch_to_fwd=float(cfg.get("COMP_SIGN_PITCH_TO_FWD", -1.0)),
        )
    else:
        pos_comp_pose_m = pos_raw_m.copy()

    # Define the new global origin = first VO(IMU) point; rebase everything
    origin_voimu_E = float(pos_comp_pose_m[0, 0])
    origin_voimu_N = float(pos_comp_pose_m[0, 1])
    
    # Rebase VO(no-IMU) and VO(IMU) to this origin, so VO(IMU)[0] = (0,0)
    pos_noimu_m_reb = pos_raw_m - np.array([origin_voimu_E, origin_voimu_N], dtype=float)
    pos_comp_m_reb  = pos_comp_pose_m - np.array([origin_voimu_E, origin_voimu_N], dtype=float)
    
    # Convert GPS to meters with origin at VO(IMU)[0]
    gps_E_m = gps_N_m = None
    if (gps_lat0 is not None) and (gps_lon0 is not None):
        # Convert VO(IMU) EN meters to lat/lon using the original GPS origin
        # to discover the lat/lon of VO(IMU)[0]
        from csv_utils import en_to_latlon
        lat_comp, lon_comp = en_to_latlon(
            gps_lat0, gps_lon0,
            dE=pos_comp_pose_m[:, 0],
            dN=pos_comp_pose_m[:, 1],
        )
        lat_ref = float(lat_comp[0])   # lat of VO(IMU) first pose
        lon_ref = float(lon_comp[0])   # lon of VO(IMU) first pose
        
        # Now convert the synchronized GPS (lat_pose, lon_pose) to meters using this new reference
        gps_E_m, gps_N_m = latlon_to_en(lat_ref, lon_ref, lat_pose, lon_pose)
        
        # Move GPS first point to (0,0) by subtracting the first GPS point
        gps_E_m = gps_E_m - gps_E_m[0]
        gps_N_m = gps_N_m - gps_N_m[0]
        
        print(f"🎯 GPS rebased to VO(IMU) origin: {lat_ref:.6f}, {lon_ref:.6f}")
        print(f"🎯 GPS first point moved to (0,0): E={gps_E_m[0]:.3f}, N={gps_N_m[0]:.3f}")
    
    # Legacy compatibility - keep pos_withimu_m for existing code
    pos_withimu_m = pos_comp_m_reb
    pos_noimu_m = pos_noimu_m_reb

    # ---- 11) Unified meter frame plot ----
    if not cfg.get("MINIMAL_OUTPUTS", False):
        # Plot everything in meters (same axes)
        plt.figure(figsize=(12, 8))
        plt.grid(True, alpha=0.3)
        plt.title("Unified meter frame — VO(no-IMU), VO(IMU), GPS@VO(IMU)[0]", fontsize=14)
        plt.xlabel("East (m)", fontsize=12)
        plt.ylabel("North (m)", fontsize=12)
        
        # Plot VO trajectories
        plt.plot(pos_noimu_m_reb[:, 0], pos_noimu_m_reb[:, 1], '-', linewidth=2, 
                 label="VO (no IMU)", alpha=0.8, color='blue')
        plt.plot(pos_comp_m_reb[:, 0], pos_comp_m_reb[:, 1], '-', linewidth=2, 
                 label="VO (IMU)", alpha=0.8, color='green')
        
        # Plot GPS if available
        if gps_E_m is not None and gps_N_m is not None:
            plt.plot(gps_E_m, gps_N_m, '.', ms=2.5, alpha=0.9, 
                    label="GPS (synced, meters)", color='red')
        
        # Add origin marker
        plt.plot(0, 0, '*', ms=10, color='black', label="VO(IMU) start")
        
        plt.axis('equal')
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()
        plt.savefig(cfg.get("vo_vs_gps_png", "outputs/vo_vs_gps.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("🧭 Plotted unified meter frame with GPS rebased to VO(IMU)[0].")
    
    
    # Keep existing individual plots for compatibility (unless minimal mode)
    if not cfg.get("MINIMAL_OUTPUTS", False):
        plot_gps_track(imu_df, cfg)
        plot_imu_angles(imu_df, cfg)
        plot_vo_metric_no_imu(pos_noimu_m_reb, gps_lat0, gps_lon0, imu_df, cfg,
                              lat_pose=lat_pose, lon_pose=lon_pose)
        if pos_withimu_m is not None:
            plot_vo_metric_with_imu(pos_withimu_m, gps_lat0, gps_lon0, imu_df, cfg,
                                    lat_pose=lat_pose, lon_pose=lon_pose)

    # Save pairs CSV (unchanged)
    if cfg["save_csv"] and pairs:
        with open(cfg["save_csv"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "inliers", "dx_px", "dy_px", "rot_deg", "dt_ms"])
            for row in pairs:
                w.writerow([row[0], row[1], f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}", f"{row[5]:.2f}"])
        print(f"💾 CSV saved: {cfg['save_csv']}")

    # --- BEGIN: write VO↔GPS synchronized CSV for debugging ---
    if cfg.get("WRITE_SYNC_CSV", False):
        import pandas as pd

        os.makedirs(os.path.dirname(cfg["SYNC_CSV_PATH"]), exist_ok=True)

        # 1) VO → lat/lon at each pose sample (if GPS origin available)
        if (gps_lat0 is not None) and (gps_lon0 is not None):
            from csv_utils import en_to_latlon
            vo_lat, vo_lon = en_to_latlon(
                gps_lat0, gps_lon0,
                dE=pos_noimu_m[:, 0],  # East meters
                dN=pos_noimu_m[:, 1],  # North meters
            )
        else:
            # no lat/lon in IMU CSV -> fill with NaNs so CSV shape stays consistent
            vo_lat = np.full(len(pos_noimu_m), np.nan, dtype=float)
            vo_lon = np.full(len(pos_noimu_m), np.nan, dtype=float)

        # 2) GPS synced track at the same pose times (already computed above)
        # Ensure arrays exist; if not, create NaNs of the right length
        if 'lat_pose' not in locals() or lat_pose is None:
            gps_lat_sync = np.full(len(pos_noimu_m), np.nan, dtype=float)
        else:
            gps_lat_sync = np.asarray(lat_pose, dtype=float)

        if 'lon_pose' not in locals() or lon_pose is None:
            gps_lon_sync = np.full(len(pos_noimu_m), np.nan, dtype=float)
        else:
            gps_lon_sync = np.asarray(lon_pose, dtype=float)

        # 3) Sync time column
        # We export the *pose_t* (absolute Unix seconds) as the synchronization time.
        sync_time = np.asarray(pose_t, dtype=float)

        # 4) Build the exact columns you requested (names preserved verbatim)
        df_sync = pd.DataFrame({
            "Sync time": sync_time,
            "VO_lat":    np.asarray(vo_lat, dtype=float),
            "Vo_long":   np.asarray(vo_lon, dtype=float),
            "GPS_lat":   np.asarray(gps_lat_sync, dtype=float),
            "GPS_LONG":  np.asarray(gps_lon_sync, dtype=float),
        })

        df_sync.to_csv(cfg["SYNC_CSV_PATH"], index=False)
        print(f"💾 Sync CSV written: {cfg['SYNC_CSV_PATH']}  rows={len(df_sync)}")

        # Optional: a few high-signal prints to verify sync at a glance
        imu_t0, imu_t1 = float(imu_df['t_s'].iloc[0]), float(imu_df['t_s'].iloc[-1])
        print(f"⏱  pose_t range: {sync_time[0]:.6f} .. {sync_time[-1]:.6f} (Unix s)")
        print(f"⏱  IMU t range : {imu_t0:.6f} .. {imu_t1:.6f} (Unix s)")
        print(f"⏱  Offset used : auto={auto_off:+.6f}  user={cfg.get('TIME_OFFSET_USER_S', 0.0):+.6f}  final={final_off:+.6f}")
    # --- END: write VO↔GPS synchronized CSV for debugging ---

    # --- Export unified meter frame CSV for debugging ---
    if cfg.get("WRITE_SYNC_CSV", False):
        import pandas as pd
        os.makedirs(os.path.dirname(cfg["SYNC_CSV_PATH"]), exist_ok=True)
        df_dbg = pd.DataFrame({
            "Sync time": pose_t,
            "VO_NoIMU_E": pos_noimu_m_reb[:, 0],
            "VO_NoIMU_N": pos_noimu_m_reb[:, 1],
            "VO_IMU_E": pos_comp_m_reb[:, 0],
            "VO_IMU_N": pos_comp_m_reb[:, 1],
            "GPS_E": (gps_E_m if gps_E_m is not None else np.full(len(pos_comp_m_reb), np.nan)),
            "GPS_N": (gps_N_m if gps_N_m is not None else np.full(len(pos_comp_m_reb), np.nan)),
        })
        df_dbg.to_csv(cfg["SYNC_CSV_PATH"], index=False)
        print(f"💾 Sync CSV written: {cfg['SYNC_CSV_PATH']} (VO origin=VO(IMU)[0])")
    # --- END: unified meter frame CSV export ---

        # ============================================================
        # A) VO → plane-frame series by FRAME (no timestamps yet)
        # ============================================================
        if cfg.get("PLANE_OUTPUTS_ENABLED", True):
            import pandas as pd
            os.makedirs(os.path.dirname(cfg["PLANE_SERIES_CSV"]), exist_ok=True)

            # Pose count
            N = len(vo.pose_frame_ids)
            if N < 2:
                print("⚠️ Not enough poses for plane-frame series.")
            else:
                # Frame numbers (e.g., 1001..)
                import re, os as _os
                rx = re.compile(cfg.get("FRAME_FILENAME_REGEX", r"frame_(\d+)\."))
                img_frames = []
                for p in vo.paths:
                    m = rx.search(_os.path.basename(p))
                    if not m: raise ValueError(f"Cannot extract frame id: {p}")
                    img_frames.append(int(m.group(1)))
                img_frames = np.asarray(img_frames, dtype=int)
                pose_indices    = np.asarray(vo.pose_frame_ids, dtype=int)      # 0..N-1
                frame_nums      = img_frames[pose_indices]                    # (N,)
                step_frame_nums = frame_nums[1:]                              # (N-1,)

                # VO per-step pixel motions
                dpx = np.array([[row[2], row[3]] for row in pairs], dtype=np.float32)   # (N-1, 2) [dx, dy] px
                rot_vo_step_deg = np.array([row[4] for row in pairs], dtype=float)      # (N-1,)

                # Scale to meters along image axes
                dE_vo = dpx[:, 0] * sx   # East meters per step (from image X)
                dN_vo = dpx[:, 1] * sy   # North meters per step (from image Y)
                
                # Use aligned heading for cumulative heading (convert to degrees) with -1 factor
                vo_heading_cum_deg = np.degrees(yaw_pose_aligned) * -1.0  # (N,) aligned cumulative heading with -1 factor

                # Your rule: in VO we want the raw along-x / along-y components in aircraft frame.
                # Define which image axis is "forward" (vs "lateral") via config:
                if str(cfg.get("VO_FORWARD_AXIS", "y")).lower() == "y":
                    fwd_vo_step = cfg.get("SIGN_Y_TO_FORWARD", +1.0) * dN_vo * -1.0  # Apply -1 factor
                    lat_vo_step = cfg.get("SIGN_X_TO_LATERAL", +1.0) * dE_vo
                else:  # forward = image X
                    fwd_vo_step = cfg.get("SIGN_X_TO_LATERAL", +1.0) * dE_vo * -1.0  # Apply -1 factor
                    lat_vo_step = cfg.get("SIGN_Y_TO_FORWARD", +1.0) * dN_vo

                # Integrate to cumulative series (start at 0 at first frame)
                fwd_vo_cum = np.concatenate([[0.0], np.cumsum(fwd_vo_step)])   # (N,)
                lat_vo_cum = np.concatenate([[0.0], np.cumsum(lat_vo_step)])   # (N,)
                
                # Compute VO(IMU) lateral from the compensated trajectory
                lat_vo_imu_cum = pos_comp_m_reb[:, 1]  # North component (Y) for IMU-compensated

                # ============================================================
                # B) Map FRAME → TIME using pose_t (from your CSV sync)
                # ============================================================
                time_rel = pose_t - pose_t[0]           # (N,) seconds since first frame time
                t_step_time = time_rel[1:]              # (N-1,) time at steps

                # ============================================================
                # C) GPS → plane-frame series (same aircraft-frame projection as VO)
                # ============================================================
                fwd_gps_cum = lat_gps_cum = rot_gps_step_deg = None
                if (lat_pose is not None) and (lon_pose is not None):
                    # Convert GPS to EN meters around the first GPS fix at pose times
                    E_gps, N_gps = latlon_to_en(lat_pose[0], lon_pose[0], lat_pose, lon_pose)
                    # Per-step EN deltas
                    dE_gps = np.diff(E_gps)                            # (N-1,)
                    dN_gps = np.diff(N_gps)                            # (N-1,)
                    
                    # Project EN→body using the SAME aligned heading series as VO
                    fwd_gps_step, right_gps_step = en_steps_to_body(dE_gps, dN_gps, yaw_step_used)
                    
                    # SWAP forward and lateral - same as VO (the axes are inverted!)
                    fwd_gps_step, right_gps_step = right_gps_step, fwd_gps_step
                    
                    # Apply only explicit config signs (if needed)
                    fwd_gps_step = cfg.get("PLANE_SIGN_FWD", +1.0) * fwd_gps_step
                    right_gps_step = cfg.get("PLANE_SIGN_LAT", +1.0) * right_gps_step
                    
                    # Cumulative (start at 0)
                    fwd_gps_cum = np.concatenate([[0.0], np.cumsum(fwd_gps_step)])
                    lat_gps_cum = np.concatenate([[0.0], np.cumsum(right_gps_step)])
                    
                    # GPS rotation per step = course change between segments
                    course_deg = np.degrees(np.arctan2(dE_gps, dN_gps))
                    course_deg = np.degrees(np.unwrap(np.deg2rad(course_deg)))
                    rot_gps_step_deg = np.diff(np.concatenate([[course_deg[0]], course_deg]))

                # ============================================================
                # D) Save one compact CSV aligned by frame, with time columns
                # ============================================================
                out = {
                    "frame_num": frame_nums,
                    "time_rel_s": time_rel,
                    "fwd_vo_cum_m": fwd_vo_cum,
                    "lat_vo_cum_m": lat_vo_cum,
                    "lat_vo_imu_cum_m": lat_vo_imu_cum,
                    "vo_heading_cum_deg": vo_heading_cum_deg,
                }
                # steps table aligned to next frame index
                out_steps = {
                    "step_frame_num": step_frame_nums,
                    "t_step_s": t_step_time,
                    "fwd_vo_step_m": fwd_vo_step,
                    "lat_vo_step_m": lat_vo_step,
                    "rot_vo_step_deg": rot_vo_step_deg,
                }
                # add GPS if present
                if fwd_gps_cum is not None:
                    out["fwd_gps_cum_m"] = fwd_gps_cum
                    out["lat_gps_cum_m"] = lat_gps_cum
                    out["gps_heading_cum_deg"] = gps_heading_cum_deg
                    out_steps["fwd_gps_step_m"] = fwd_gps_step
                    out_steps["lat_gps_step_m"] = lat_gps_step
                    out_steps["rot_gps_step_deg"] = rot_gps_step_deg

                # Add heading info (degrees) to debug CSV
                yaw_vo_deg         = np.degrees(yaw_pose)
                yaw_vo_aligned_deg = np.degrees(yaw_pose_aligned)
                out["yaw_vo_deg"] = yaw_vo_deg
                out["yaw_vo_aligned_deg"] = yaw_vo_aligned_deg
                
                # merge and write
                df_pose = pd.DataFrame(out)
                df_step = pd.DataFrame(out_steps)
                merged = df_pose.merge(df_step, left_on="frame_num", right_on="step_frame_num", how="left")
                merged.to_csv(cfg["PLANE_SERIES_CSV"], index=False)
                print(f"💾 Plane-frame series CSV saved: {cfg['PLANE_SERIES_CSV']}")

                # ============================================================
                # E) Plot the three panels (VO vs GPS in same units)
                # ============================================================
                from visualization import plot_plane_series
                plot_plane_series(time_rel, frame_nums,
                                  fwd_vo_cum, lat_vo_cum, vo_heading_cum_deg,
                                  t_step_time, step_frame_nums,
                                  fwd_gps_cum, lat_gps_cum, gps_heading_cum_deg,
                                  lat_vo_imu_cum,
                                  out_png=cfg["PLANE_SERIES_PNG"])

        # ---- 13) Generate mosaic (if enabled) ----
        if cfg.get("ENABLE_MOSAIC", False):
            # Get trajectory data for mosaic
            traj_xy = vo.traj_xy if hasattr(vo, 'traj_xy') else np.array([])
            generate_mosaic(vo.global_A, vo.pose_frame_ids, vo.paths, 
                           vo.W0, vo.H0, vo.W_full, vo.H_full, vo.dc, cfg, traj_xy)

        # ---- 14) Lateral analysis: VO lateral rate (body Right) vs IMU lateral comp Δ ----
        if not cfg.get("MINIMAL_OUTPUTS", False):
            plt.figure(figsize=(14, 8))

            # Time axis for steps (one value per frame-to-frame step)
            time_steps = (pose_t - pose_t[0])[1:]  # (N-1,)

            # --- VO lateral translation rate (meters per frame) in AIRCRAFT frame ---
            # Use the *uncompensated* VO path to measure what the camera sees
            dE_vo = np.diff(pos_noimu_m[:, 0])   # EN steps (E)
            dN_vo = np.diff(pos_noimu_m[:, 1])   # EN steps (N)
            # Use aligned heading for consistent aircraft frame projection
            _, right_vo_step = en_steps_to_body(dE_vo, dN_vo, yaw_step_used)  # meters/frame

            # --- IMU lateral compensation per step (meters per frame) in BODY Right axis ---
            altitude = float(cfg.get("ALTITUDE_M", 100.0))
            sign_r = float(cfg.get("COMP_SIGN_ROLL_TO_RIGHT", +1.0))
            comp_right_pose = sign_r * altitude * np.tan(roll_pose)       # meters (pose-wise)
            comp_right_step = np.diff(comp_right_pose)                     # meters/frame

            # 1) Roll angle (for context)
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(pose_t - pose_t[0], np.degrees(roll_pose), '-', lw=2, label="Roll [deg]")
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Roll (deg)")
            ax1.set_title("Roll angle over time")
            ax1.legend()

            # 2) Lateral rate vs IMU compensation Δ — same units: meters per frame
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(time_steps, right_vo_step, '-', lw=2, label="VO lateral rate [m/frame]")
            ax2.plot(time_steps, comp_right_step, '-', lw=2, label="IMU lateral comp Δ [m/frame]")
            
            # Optional: Show m/s alongside m/frame if enabled
            if cfg.get("LATERAL_ANALYSIS_SHOW_MPS", False):
                dt_steps = np.diff(pose_t)  # Time between frames
                right_vo_mps = right_vo_step / dt_steps  # Convert to m/s
                comp_right_mps = comp_right_step / dt_steps  # Convert to m/s
                ax2.plot(time_steps, right_vo_mps, '--', lw=1.5, alpha=0.7, label="VO lateral rate [m/s]")
                ax2.plot(time_steps, comp_right_mps, '--', lw=1.5, alpha=0.7, label="IMU lateral comp Δ [m/s]")
            
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Meters per frame" + (" / Meters per second" if cfg.get("LATERAL_ANALYSIS_SHOW_MPS", False) else ""))
            ax2.set_title("Lateral translation rate (aircraft Right) vs IMU compensation Δ")
            ax2.legend()

            plt.tight_layout()
            plt.savefig("outputs/lateral_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("📊 Plotted lateral analysis (aircraft-frame): VO lateral rate vs IMU Δ.")

        print("✅ Visual Odometry processing complete!")

if __name__ == "__main__":
    main()