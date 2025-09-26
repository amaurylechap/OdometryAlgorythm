#!/usr/bin/env python3
"""
Visual Odometry Complete Workflow
Single script for VO analysis and mosaic generation with parameter saving.
"""

import os
import time
import pickle
import config
from vo_algorithm import VisualOdometry
from visualization import generate_mosaic, plot_trajectory_xy, plot_plane_series

class VOWorkflow:
    """Complete VO workflow with parameter saving"""
    
    def __init__(self):
        self.params_file = "outputs/vo_params.pkl"
        self.vo_data = None
    
    def run_analysis(self):
        """Run VO analysis and save parameters"""
        print("🚀 Running VO Analysis...")
        start_time = time.time()
        
        vo = VisualOdometry(config.CONFIG)
        vo.initialize()
        pairs, global_A, pose_frame_ids, traj_xy = vo.run()
        
        # Save parameters
        os.makedirs("outputs", exist_ok=True)
        vo_data = {
            'global_A': vo.global_A, 'pose_frame_ids': vo.pose_frame_ids,
            'paths': vo.paths, 'traj_xy': vo.traj_xy,
            'W0': vo.W0, 'H0': vo.H0, 'W_full': vo.W_full, 'H_full': vo.H_full, 'dc': vo.dc,
            # Per-frame motion data (already available!)
            'pairs': vo.pairs  # Contains (frame, inliers, dx, dy, rot_deg, dt_ms) for each frame
        }
        
        with open(self.params_file, 'wb') as f:
            pickle.dump(vo_data, f)
        
        print(f"✅ Analysis completed in {time.time() - start_time:.1f}s")
        print(f"💾 Parameters saved to: {self.params_file}")
        return vo_data
    
    def _plot_motion_series(self, frame_nums, fwd_vo, lat_vo, rot_vo, out_png):
        """Create integrated motion time series plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate cumulative/integrated motion
        fwd_cum = np.cumsum(fwd_vo)
        lat_cum = np.cumsum(lat_vo)
        rot_cum = np.cumsum(rot_vo)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Forward motion (cumulative)
        ax1.plot(frame_nums, fwd_cum, 'b-', linewidth=1)
        ax1.set_ylabel('Cumulative Forward Motion (pixels)')
        ax1.set_title('Integrated Motion Analysis')
        ax1.grid(True, alpha=0.3)
        
        # Lateral motion (cumulative)
        ax2.plot(frame_nums, lat_cum, 'r-', linewidth=1)
        ax2.set_ylabel('Cumulative Lateral Motion (pixels)')
        ax2.grid(True, alpha=0.3)
        
        # Rotation (cumulative)
        ax3.plot(frame_nums, rot_cum, 'g-', linewidth=1)
        ax3.set_ylabel('Cumulative Rotation (degrees)')
        ax3.set_xlabel('Frame Number')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Integrated motion series saved: {out_png}")
    
    def load_parameters(self):
        """Load saved VO parameters"""
        if not os.path.exists(self.params_file):
            print(f"❌ No saved parameters found. Run analysis first.")
            return None
        
        with open(self.params_file, 'rb') as f:
            self.vo_data = pickle.load(f)
        print(f"✅ Parameters loaded: {len(self.vo_data['traj_xy'])} trajectory points")
        return self.vo_data
    
    def generate_mosaics(self):
        """Generate mosaic and plots from saved parameters"""
        if not self.load_parameters():
            return False
        
        print("🎨 Generating outputs...")
        
        # Generate mosaic
        if config.CONFIG.get("ENABLE_MOSAIC", False):
            print("📸 Creating mosaic...")
            generate_mosaic(self.vo_data['global_A'], self.vo_data['pose_frame_ids'], 
                          self.vo_data['paths'], self.vo_data['W0'], self.vo_data['H0'],
                          self.vo_data['W_full'], self.vo_data['H_full'], self.vo_data['dc'],
                          config.CONFIG, self.vo_data['traj_xy'])
        
        # Generate plots
        print("📊 Creating trajectory plot...")
        plot_trajectory_xy(self.vo_data['traj_xy'], config.CONFIG)
        
        print("📈 Creating plane series plot...")
        # Extract per-frame motion data from pairs
        if 'pairs' in self.vo_data and self.vo_data['pairs']:
            pairs = self.vo_data['pairs']
            frame_nums = [p[0] for p in pairs]  # Frame numbers
            fwd_vo = [p[2] for p in pairs]     # dx (forward translation)
            lat_vo = [p[3] for p in pairs]      # dy (lateral translation) 
            rot_vo = [p[4] for p in pairs]     # rot_deg (rotation)
            
            # Create simple time series plot
            self._plot_motion_series(frame_nums, fwd_vo, lat_vo, rot_vo, 
                                   config.CONFIG.get("plane_series_png", "outputs/plane_series.png"))
        else:
            print("⏭️  Skipping plane series (no motion data available)")
        
        # Show results
        self._show_outputs()
        return True
    
    def _show_outputs(self):
        """Show generated output files"""
        files = [
            ("Mosaic", config.CONFIG["mosaic_png"]),
            ("Mosaic + Trajectory", config.CONFIG["mosaic_with_traj_png"]),
            ("Trajectory Plot", config.CONFIG["traj_png_xy"])
        ]
        
        print(f"\n📁 Generated files:")
        for name, path in files:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   ✅ {name}: {size_mb:.1f} MB")
            else:
                print(f"   ❌ {name}: not found")
    
    def run_complete(self):
        """Run complete workflow: analysis + outputs"""
        print("🎯 Complete VO Workflow")
        print("=" * 50)
        self.run_analysis()
        print()
        self.generate_mosaics()
        print("\n🎉 Workflow completed!")

def main():
    """Main interface"""
    workflow = VOWorkflow()
    
    print("🎯 Visual Odometry Workflow")
    print("=" * 30)
    print("1. Run complete analysis (first time)")
    print("2. Generate outputs (from saved data)")
    print("3. Exit")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        workflow.run_complete()
    elif choice == "2":
        workflow.generate_mosaics()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
