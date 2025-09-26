"""
Visualization utilities for Visual Odometry.
Handles plotting, progress bars, and mosaic generation.
"""

import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_utils import read_image_any, ensure_bgr, DistortionCorrector
from tracking import corners, to33, to23


def progress_wrapper(iterable, total=None, desc=""):
    """Progress bar wrapper with fallback for systems without tqdm."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="img", leave=False)
    except Exception:
        class _SimpleProg:
            def __init__(self, it, total, desc):
                self.it = iter(it)
                self.total = total
                self.count = 0
                self.desc = desc
            
            def __iter__(self):
                return self
            
            def __next__(self):
                item = next(self.it)
                self.count += 1
                if self.total:
                    pct = int(100 * self.count / max(1, self.total))
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(f"\r{self.desc} [{bar}] {self.count}/{self.total} ({pct}%)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r{self.desc} {self.count}")
                    sys.stdout.flush()
                return item
            
            def close(self):
                sys.stdout.write("\n")
                sys.stdout.flush()
        
        return _SimpleProg(iterable, total, desc)


def plot_trajectory_xy(traj_xy, config):
    """Plot XY pixel trajectory."""
    T = np.array(traj_xy, dtype=np.float32)
    Tplot = T.copy()
    
    if config["plot_center"]:
        Tplot = Tplot - Tplot[0]
    if config["plot_flip_y"]:
        Tplot[:, 1] = -Tplot[:, 1]
    
    plt.figure()
    plt.axis('equal')
    plt.grid(True)
    plt.title("Trajectory: X–Y (pixel frame)")
    plt.plot(Tplot[:, 0], Tplot[:, 1])
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.tight_layout()
    plt.savefig(config["traj_png_xy"], dpi=150)
    plt.close()
    print(f"📈 Trajectory XY saved: {config['traj_png_xy']}")


def plot_gps_track(imu_df, config):
    """Plot GPS track if available."""
    if "lat" in imu_df.columns and "lon" in imu_df.columns:
        plt.figure()
        plt.title("GPS track")
        plt.xlabel("Lon")
        plt.ylabel("Lat")
        plt.grid(True)
        plt.plot(imu_df["lon"], imu_df["lat"], '.', ms=1)
        plt.tight_layout()
        plt.savefig(config["gps_plot_png"], dpi=150)
        plt.close()
        print(f"🗺️  GPS plot saved: {config['gps_plot_png']}")


def plot_imu_angles(imu_df, config):
    """Plot IMU angles over time."""
    plt.figure()
    plt.title("IMU angles")
    plt.xlabel("time (s)")
    plt.ylabel("rad")
    plt.grid(True)
    plt.plot(imu_df["t_s"], imu_df["roll_rad"], label="roll")
    plt.plot(imu_df["t_s"], imu_df["pitch_rad"], label="pitch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config["imu_plot_png"], dpi=150)
    plt.close()
    print(f"📐 IMU angles plot saved: {config['imu_plot_png']}")


def plot_vo_metric_no_imu(pos_noimu_m, gps_lat0, gps_lon0, imu_df, config, lat_pose=None, lon_pose=None):
    """
    Plot VO metric without IMU compensation vs GPS (if available).
    Uses arrays passed in; does not read CSV files.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.grid(True)

    if gps_lat0 is not None and gps_lon0 is not None:
        from csv_utils import en_to_latlon
        lat_noimu, lon_noimu = en_to_latlon(gps_lat0, gps_lon0,
                                            dE=pos_noimu_m[:, 0], dN=pos_noimu_m[:, 1])
        plt.title("VO (no IMU) vs GPS — synced")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.plot(lon_noimu, lat_noimu, '-', label="VO (no IMU)")
        if lat_pose is not None and lon_pose is not None:
            plt.plot(lon_pose, lat_pose, '.', ms=3, alpha=0.9, label="GPS (synced)")
        elif isinstance(imu_df, dict) is False and "lat" in imu_df.columns and "lon" in imu_df.columns:
            plt.plot(imu_df["lon"], imu_df["lat"], '.', ms=2, alpha=0.6, label="GPS (raw)")
        plt.axis('equal')
    else:
        plt.title("VO (no IMU) — EN meters")
        plt.xlabel("East (m)"); plt.ylabel("North (m)")
        plt.plot(pos_noimu_m[:, 0], pos_noimu_m[:, 1], '-', label="VO (no IMU)")
        if isinstance(imu_df, dict) is False and "E" in imu_df.columns and "N" in imu_df.columns:
            plt.plot(imu_df["E"] - imu_df["E"].iloc[0], imu_df["N"] - imu_df["N"].iloc[0],
                     '.', ms=2, alpha=0.6, label="GPS EN (raw)")
        plt.axis('equal')

    plt.legend()
    plt.tight_layout()
    plt.savefig(config["vo_metric_no_imu_png"], dpi=150)
    plt.close()


def plot_vo_metric_with_imu(pos_withimu_m, gps_lat0, gps_lon0, imu_df, config, lat_pose=None, lon_pose=None):
    """
    Plot VO metric with IMU compensation vs GPS (if available).
    Uses arrays passed in; does not read CSV files.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.grid(True)

    if gps_lat0 is not None and gps_lon0 is not None:
        from csv_utils import en_to_latlon
        lat_wimu, lon_wimu = en_to_latlon(gps_lat0, gps_lon0,
                                          dE=pos_withimu_m[:, 0], dN=pos_withimu_m[:, 1])
        plt.title("VO (IMU) vs GPS — synced")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.plot(lon_wimu, lat_wimu, '-', label="VO (IMU)")
        if lat_pose is not None and lon_pose is not None:
            plt.plot(lon_pose, lat_pose, '.', ms=3, alpha=0.9, label="GPS (synced)")
        elif isinstance(imu_df, dict) is False and "lat" in imu_df.columns and "lon" in imu_df.columns:
            plt.plot(imu_df["lon"], imu_df["lat"], '.', ms=2, alpha=0.6, label="GPS (raw)")
        plt.axis('equal')
    else:
        plt.title("VO (IMU) — EN meters")
        plt.xlabel("East (m)"); plt.ylabel("North (m)")
        plt.plot(pos_withimu_m[:, 0], pos_withimu_m[:, 1], '-', label="VO (IMU)")
        if isinstance(imu_df, dict) is False and "E" in imu_df.columns and "N" in imu_df.columns:
            plt.plot(imu_df["E"] - imu_df["E"].iloc[0], imu_df["N"] - imu_df["N"].iloc[0],
                     '.', ms=2, alpha=0.6, label="GPS EN (raw)")
        plt.axis('equal')

    plt.legend()
    plt.tight_layout()
    plt.savefig(config["vo_metric_with_imu_png"], dpi=150)
    plt.close()


def generate_mosaic(global_A, pose_frame_ids, paths, W0, H0, W_full, H_full, 
                   dc, config, traj_xy=None):
    """Generate mosaic from all frames."""
    if not config.get("ENABLE_MOSAIC", False):
        print("ℹ️ Mosaic disabled (ENABLE_MOSAIC=False).")
        return
    
    # Canvas bounds in I0 coords
    W0_plot, H0_plot = W0, H0
    cs = corners(W0_plot, H0_plot)
    warped_corners = [cv2.transform(cs, A).reshape(-1, 2) for A in global_A]
    allc = np.vstack(warped_corners)
    minx, miny = np.floor(allc.min(axis=0)).astype(int)
    maxx, maxy = np.ceil(allc.max(axis=0)).astype(int)
    margin = int(config["canvas_margin"])
    offset = np.array([margin - minx, margin - miny], dtype=np.float32)
    CW = int((maxx - minx) + 2 * margin)
    CH = int((maxy - miny) + 2 * margin)
    r = float(config.get("MOSAIC_RENDER_SCALE", 1.0))
    r = max(0.1, min(1.0, r))
    CW_r = int(max(1, round(CW * r)))
    CH_r = int(max(1, round(CH * r)))
    S = np.array([[r, 0, 0], [0, r, 0], [0, 0, 1]], dtype=np.float32)
    mode = config.get("MOSAIC_MODE", "image").lower().strip()
    stride = max(1, int(config.get("MOSAIC_STRIDE", 1)))
    alpha = float(config["alpha"])
    canvas = np.zeros((CH_r, CW_r, 3), dtype=np.uint8)
    
    render_indices = list(range(0, len(global_A), stride))
    prog = progress_wrapper(render_indices, total=len(render_indices), desc="Mosaic")
    
    try:
        for j in prog:
            idx_path = pose_frame_ids[j]
            if idx_path < 0 or idx_path >= len(paths):
                print(f"⚠️ Mosaic skip: invalid path index {idx_path} for pose {j}")
                continue
            
            p = paths[idx_path]
            A = global_A[j].astype(np.float32)
            A_can = A.copy()
            A_can[0, 2] += offset[0]
            A_can[1, 2] += offset[1]
            A_can_r = to23(S @ to33(A_can))
            
            if mode == "outline":
                rect = corners(W0_plot, H0_plot)
                rect[:, 0, 0] += offset[0]
                rect[:, 0, 1] += offset[1]
                rect33 = np.concatenate([rect.reshape(-1, 2), np.ones((4, 1), np.float32)], axis=1).T
                rect_r = (S @ rect33).T[:, :2].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [rect_r], True, (80, 80, 80), 1, cv2.LINE_AA)
                continue
            
            img_full = read_image_any(p, as_bgr=True)
            if img_full is None:
                print(f"⚠️ Mosaic skip unreadable: {p}")
                continue
            if dc is not None:
                img_full = dc.apply_bgr(img_full)
            img_proc = cv2.resize(img_full, (W0, H0), interpolation=cv2.INTER_AREA) if (W0 != W_full or H0 != H_full) else img_full
            img_proc = ensure_bgr(img_proc)
            
            if mode == "thumbnail":
                src_scale = max(0.1, min(1.0, float(config.get("MOSAIC_SRC_WARP_SCALE", 0.5))))
                if src_scale < 1.0:
                    h_s, w_s = img_proc.shape[:2]
                    img_src = cv2.resize(img_proc, (int(w_s * src_scale), int(h_s * src_scale)), 
                                        interpolation=cv2.INTER_AREA)
                    A_can_r_adj = A_can_r.copy()
                    A_can_r_adj[:, :2] /= src_scale
                else:
                    img_src = img_proc
                    A_can_r_adj = A_can_r
                warped = cv2.warpAffine(img_src, A_can_r_adj, (CW_r, CH_r),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            else:
                warped = cv2.warpAffine(img_proc, A_can_r, (CW_r, CH_r),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            
            m = (warped.sum(axis=2) > 0)
            if not np.any(m):
                continue
            if alpha >= 1.0:
                canvas[m] = warped[m]
            else:
                canvas[m] = (alpha * warped[m] + (1.0 - alpha) * canvas[m]).astype(np.uint8)
    finally:
        try:
            prog.close()
        except Exception:
            pass
    
    cv2.imwrite(config["mosaic_png"], canvas)
    print(f"\n🖼️ Mosaic saved: {config['mosaic_png']}")
    
    # overlay simple traj
    if traj_xy is not None and len(traj_xy) > 0:
        traj_xy = np.array(traj_xy, dtype=np.float32)
        traj_px = (traj_xy + offset) * r
        pts = traj_px.astype(np.int32).reshape(-1, 1, 2)
        canvas_traj = canvas.copy()
        cv2.polylines(canvas_traj, [pts], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite(config["mosaic_with_traj_png"], canvas_traj)
        print(f"🖼️ Mosaic + Trajectory saved: {config['mosaic_with_traj_png']}")
    else:
        print("⚠️ No trajectory data available for mosaic overlay")


def plot_plane_series(time_rel, frame_nums,
                      fwd_vo_cum, lat_vo_cum, vo_heading_cum_deg,
                      t_step_time, step_frame_nums,
                      fwd_gps_cum=None, lat_gps_cum=None, gps_heading_cum_deg=None,
                      lat_vo_imu_cum=None,
                      out_png="outputs/plane_series.png"):
    """
    Plot aircraft-frame trajectories:
      - Forward (cumulative, meters)
      - Lateral (cumulative, meters)
      - Heading (cumulative, degrees)
    VO is required; GPS overlays if provided.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 9))

    # 1) Forward cumulative
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_rel, fwd_vo_cum, label="VO forward [m]", linewidth=2)
    if fwd_gps_cum is not None:
        ax1.plot(time_rel, fwd_gps_cum, label="GPS forward [m]", linewidth=2, alpha=0.9)
    ax1.set_ylabel("Forward [m]"); ax1.grid(True); ax1.legend(loc="best")

    # 2) Lateral cumulative
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(time_rel, lat_vo_cum, label="VO (no IMU) lateral [m]", linewidth=2)
    if lat_vo_imu_cum is not None:
        ax2.plot(time_rel, lat_vo_imu_cum, label="VO (IMU) lateral [m]", linewidth=2, alpha=0.8, color='green')
    if lat_gps_cum is not None:
        ax2.plot(time_rel, lat_gps_cum, label="GPS lateral [m]", linewidth=2, alpha=0.9)
    ax2.set_ylabel("Lateral [m]"); ax2.grid(True); ax2.legend(loc="best")

    # 3) Cumulative heading
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(time_rel, vo_heading_cum_deg, label="VO heading [deg]", linewidth=2)
    if gps_heading_cum_deg is not None:
        ax3.plot(time_rel, gps_heading_cum_deg, label="GPS heading [deg]", linewidth=2, alpha=0.9)
    ax3.set_xlabel("Time since start [s]"); ax3.set_ylabel("Heading [deg]")
    ax3.grid(True); ax3.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Plane-frame series plot saved: {out_png}")
