#!/usr/bin/env python3
"""
Fast analysis using pre-computed VO data.
Loads raw VO data and performs analysis without re-running VO.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIG
from csv_utils import load_frame_csv, load_imu_csv, en_to_latlon, latlon_to_en
from imu_compensation import compensate_positions_body_to_en
from visualization import plot_plane_series

def en_steps_to_body(dE_step, dN_step, yaw_rad_step):
    """Project EN step vectors to aircraft body axes."""
    c = np.cos(yaw_rad_step); s = np.sin(yaw_rad_step)
    fwd  = c * dN_step + s * dE_step
    right = -s * dN_step + c * dE_step
    return fwd, right

def main():
    """Fast analysis using pre-computed VO data."""
    cfg = CONFIG
    
    # Check if raw data exists
    if not os.path.exists("outputs/raw_data/vo_raw_data.csv"):
        print("❌ No raw VO data found. Run 'python run_vo_export.py' first.")
        return
    
    print("Loading pre-computed VO data...")
    
    # Load raw VO data
    vo_data = pd.read_csv("outputs/raw_data/vo_raw_data.csv")
    traj_data = pd.read_csv("outputs/raw_data/vo_trajectory.csv")
    
    print(f"Loaded VO data: {len(vo_data)} frames")
    
    # Extract data
    frame_ids = vo_data['frame_id'].values
    dx_px = vo_data['dx_px'].values
    dy_px = vo_data['dy_px'].values
    rotations_rad = vo_data['rotation_rad'].values
    x_px = traj_data['x_px'].values
    y_px = traj_data['y_px'].values
    
    # Convert to meters
    scale = float(cfg["SCALE_M_PER_PX"] if cfg.get("SCALE_M_PER_PX") is not None
                  else float(cfg["ALTITUDE_M"]) / float(cfg["LENS_FACTOR"]))
    sx = float(cfg["IMAGE_X_TO_EAST"]) * scale
    sy = float(cfg["IMAGE_Y_TO_NORTH"]) * scale
    
    # Convert trajectory to meters in world coordinates (East/North)
    Tpix = np.column_stack([x_px, y_px])
    Trel = Tpix - Tpix[0]
    pos_raw_m = np.column_stack([Trel[:, 0] * sx, Trel[:, 1] * sy])
    
    # Compute EN steps from trajectory
    dE_vo = np.diff(pos_raw_m[:, 0])  # East steps
    dN_vo = np.diff(pos_raw_m[:, 1])  # North steps
    
    print("Computing aircraft-frame analysis...")
    
    # VO step rotations (deg) → cumulative heading (rad)
    yaw_steps_deg = rotations_rad  # These are actually degrees from the CSV
    yaw_steps_rad = np.deg2rad(yaw_steps_deg)
    yaw_pose = np.zeros(len(pos_raw_m), dtype=float)
    yaw_pose[1:] = np.cumsum(yaw_steps_rad)
    
    # Apply constant offset (simplified for fast analysis - no GPS alignment)
    user_off_rad = np.deg2rad(float(cfg.get("HEADING_OFFSET_USER_DEG", 0.0)))
    yaw_pose_aligned = yaw_pose + user_off_rad
    yaw_step_used = yaw_pose_aligned[:-1]  # heading at start of each step
    
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
    
    # GPS forward/lateral/rotation per step (simplified for fast analysis)
    fwd_gps_cum = lat_gps_cum = rot_gps_step_deg = None
    # Note: GPS data would need to be loaded from CSV files for full implementation
    # For now, we'll just use VO data
    
    # Time arrays - match the cumulative arrays (599 steps + 1 initial = 600)
    time_rel = np.arange(len(rotations_rad) + 1) * 0.1  # Assume 10fps
    frame_nums = np.arange(len(rotations_rad) + 1)
    
    print("Generating plane series plot...")
    
    # Plot plane series
    plot_plane_series(time_rel, frame_nums,
                      fwd_vo_cum, lat_vo_cum, vo_heading_cum_deg,
                      time_rel[1:], frame_nums[1:],
                      out_png=cfg["PLANE_SERIES_PNG"])
    
    print("Fast analysis complete!")
    print(f"   - Forward range: [{fwd_vo_cum.min():.2f}, {fwd_vo_cum.max():.2f}] m")
    print(f"   - Lateral range: [{lat_vo_cum.min():.2f}, {lat_vo_cum.max():.2f}] m")
    print(f"   - Rotation range: [{rot_vo_step_deg.min():.2f}, {rot_vo_step_deg.max():.2f}] deg")

if __name__ == "__main__":
    main()
