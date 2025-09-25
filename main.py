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
    plot_vo_metric_no_imu, plot_vo_metric_with_imu
)
from csv_utils import load_frame_csv, load_imu_csv, en_to_latlon
from imu_compensation import compensate_positions_absolute
from config import CONFIG

def main():
    cfg = CONFIG

    # ---- 1) Run VO exactly as-is ----
    vo = VisualOdometry(cfg)
    vo.initialize()
    pairs, global_A, pose_frame_ids, traj_xy = vo.run()
    plot_trajectory_xy(traj_xy, cfg)

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
    pos_noimu_m = np.column_stack([Trel[:, 0] * sx, Trel[:, 1] * sy])

    # ---- 8.5) Apply user-defined VO rotation (if specified) ----
    vo_rotation_deg = cfg.get("VO_ROTATION_DEG", 0.0)
    if vo_rotation_deg != 0.0:
        print(f"🔄 Applying VO rotation: {vo_rotation_deg:.2f}°")
        theta = np.deg2rad(vo_rotation_deg)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        # Apply rotation to VO trajectory
        vo_rotated = np.column_stack([
            cos_theta * pos_noimu_m[:, 0] - sin_theta * pos_noimu_m[:, 1],
            sin_theta * pos_noimu_m[:, 0] + cos_theta * pos_noimu_m[:, 1]
        ])
        pos_noimu_m = vo_rotated

    # ---- 9) VO per-pose tilt compensation (absolute 0 reference) ----
    pos_comp_pose_m = None
    if cfg.get("ENABLE_TILT_COMP", True):
        pos_comp_pose_m = compensate_positions_absolute(
            pos_noimu_EN_m=pos_noimu_m,
            roll_rad_series=roll_pose,
            pitch_rad_series=pitch_pose,
            altitude_m=float(cfg.get("ALTITUDE_M", 100.0)),
            sign_roll_to_E=float(cfg.get("COMP_SIGN_ROLL_TO_E", +1.0)),
            sign_pitch_to_N=float(cfg.get("COMP_SIGN_PITCH_TO_N", +1.0)),
        )

        # Convert to lat/lon using same origin
        if (gps_lat0 is not None) and (gps_lon0 is not None):
            lat_comp, lon_comp = en_to_latlon(
                gps_lat0, gps_lon0,
                dE=pos_comp_pose_m[:, 0],
                dN=pos_comp_pose_m[:, 1],
            )
        else:
            lat_comp = lon_comp = None

        # Plot overlay
        import matplotlib.pyplot as plt
        plt.figure(); plt.grid(True)
        plt.title("VO (IMU-compensated, abs 0 ref) vs GPS")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        if lat_comp is not None and lon_comp is not None:
            plt.plot(lon_comp, lat_comp, '-', label="VO (IMU-comp)")
        else:
            plt.xlabel("East (m)"); plt.ylabel("North (m)")
            plt.plot(pos_comp_pose_m[:, 0], pos_comp_pose_m[:, 1], '-', label="VO (IMU-comp)")
        if (lat_pose is not None) and (lon_pose is not None):
            plt.plot(lon_pose, lat_pose, '.', ms=3, alpha=0.9, label="GPS (synced)")
        plt.axis('equal'); plt.legend(); plt.tight_layout()
        plt.savefig(cfg["vo_metric_with_imu_png"], dpi=150); plt.close()
        print(f"🧭 VO (IMU-compensated, abs 0 ref) saved: {cfg['vo_metric_with_imu_png']}")

    # Legacy compatibility - keep pos_withimu_m for existing code
    pos_withimu_m = pos_comp_pose_m

    # ---- 10) Plots ----
    # quick sanity IMU plots
    plot_gps_track(imu_df, cfg)
    plot_imu_angles(imu_df, cfg)

    # VO vs GPS (no-IMU)
    plot_vo_metric_no_imu(pos_noimu_m, gps_lat0, gps_lon0, imu_df, cfg,
                          lat_pose=lat_pose, lon_pose=lon_pose)

    # VO vs GPS (IMU-compensated), if available
    if pos_withimu_m is not None:
        plot_vo_metric_with_imu(pos_withimu_m, gps_lat0, gps_lon0, imu_df, cfg,
                                lat_pose=lat_pose, lon_pose=lon_pose)

    # Combined quicklook plot (VO no-IMU + GPS synced), always equal aspect
    try:
        from csv_utils import en_to_latlon
        plt.figure()
        plt.grid(True)
        if gps_lat0 is not None:
            lat_noimu, lon_noimu = en_to_latlon(gps_lat0, gps_lon0,
                                                dE=pos_noimu_m[:, 0], dN=pos_noimu_m[:, 1])
            plt.title("VO (no IMU) vs GPS — synced")
            plt.xlabel("Longitude"); plt.ylabel("Latitude")
            plt.plot(lon_noimu, lat_noimu, '-', label="VO (no IMU)")
            if lat_pose is not None and lon_pose is not None:
                plt.plot(lon_pose, lat_pose, '.', ms=3, alpha=0.9, label="GPS (synced)")
            plt.axis('equal'); plt.legend(); plt.tight_layout()
        else:
            plt.title("VO (no IMU) — EN meters (no lat/lon in CSV)")
            plt.xlabel("East (m)"); plt.ylabel("North (m)")
            plt.plot(pos_noimu_m[:, 0], pos_noimu_m[:, 1], '-', label="VO (no IMU)")
            if "E" in imu_df.columns and "N" in imu_df.columns:
                # Clip IMU EN to tq range for fair visual
                E_syn = np.interp(tq, imu_t, imu_df["E"].values,
                                  left=imu_df["E"].values[0], right=imu_df["E"].values[-1])
                N_syn = np.interp(tq, imu_t, imu_df["N"].values,
                                  left=imu_df["N"].values[0], right=imu_df["N"].values[-1])
                plt.plot(E_syn - E_syn[0], N_syn - N_syn[0], '.', ms=2, alpha=0.8, label="GPS EN (synced)")
            plt.axis('equal'); plt.legend(); plt.tight_layout()
        plt.savefig(cfg["vo_vs_gps_png"], dpi=150); plt.close()
        print(f"🧭 Combined VO vs GPS saved: {cfg['vo_vs_gps_png']}")
    except Exception as e:
        print(f"⚠️ Combined quicklook plot skipped: {e}")

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

    print("✅ Visual Odometry processing complete!")

if __name__ == "__main__":
    main()