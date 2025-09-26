#!/usr/bin/env python3
"""
Run VO once and export raw translation/rotation data for fast analysis.
This separates the slow VO computation from the fast analysis iterations.
"""

import os
import numpy as np
import pandas as pd
from vo_algorithm import VisualOdometry
from config import CONFIG

def main():
    """Run VO and export raw data."""
    cfg = CONFIG
    
    print("Running VO pipeline...")
    vo = VisualOdometry(cfg)
    vo.initialize()
    pairs, global_A, pose_frame_ids, traj_xy = vo.run()
    
    print("Exporting raw VO data...")
    
    # Extract raw data from pairs
    # pairs format: (i, inl, dx, dy, rot_deg, dt_ms)
    translations = np.array([[row[2], row[3]] for row in pairs])  # [dx, dy] per frame
    rotations = np.array([row[4] for row in pairs])  # rotation per frame
    
    # Create output directory
    os.makedirs("outputs/raw_data", exist_ok=True)
    
    # Ensure all arrays have the same length
    n_frames = len(pairs)
    print(f"VO data: {n_frames} frames, {len(pose_frame_ids)} pose_frame_ids, {len(translations)} translations")
    
    # Export raw VO data
    vo_data = pd.DataFrame({
        'frame_id': pose_frame_ids[:n_frames],  # Truncate to match pairs length
        'dx_px': translations[:, 0],
        'dy_px': translations[:, 1], 
        'rotation_rad': rotations,
        'rotation_deg': np.degrees(rotations)
    })
    
    vo_data.to_csv("outputs/raw_data/vo_raw_data.csv", index=False)
    print(f"Exported VO data: {len(vo_data)} frames")
    print(f"   - Translation range: dx=[{translations[:, 0].min():.2f}, {translations[:, 0].max():.2f}] px")
    print(f"   - Translation range: dy=[{translations[:, 1].min():.2f}, {translations[:, 1].max():.2f}] px") 
    print(f"   - Rotation range: [{np.degrees(rotations.min()):.2f}, {np.degrees(rotations.max()):.2f}] deg")
    
    # Export trajectory
    traj_xy_array = np.array(traj_xy)
    traj_data = pd.DataFrame({
        'frame_id': range(len(traj_xy_array)),
        'x_px': traj_xy_array[:, 0],
        'y_px': traj_xy_array[:, 1]
    })
    traj_data.to_csv("outputs/raw_data/vo_trajectory.csv", index=False)
    print(f"Exported trajectory: {len(traj_xy)} points")
    
    print("VO export complete! Use fast_analysis.py for iterations.")

if __name__ == "__main__":
    main()
