#!/usr/bin/env python3
"""
Workflow manager for VO analysis.
Separates slow VO computation from fast analysis iterations.
"""

import os
import sys
import subprocess

def run_vo_export():
    """Run VO once and export raw data."""
    print("🚀 Step 1: Running VO and exporting raw data...")
    result = subprocess.run([sys.executable, "run_vo_export.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ VO export failed: {result.stderr}")
        return False
    print("✅ VO export complete!")
    return True

def run_fast_analysis():
    """Run fast analysis using pre-computed data."""
    print("📊 Step 2: Running fast analysis...")
    result = subprocess.run([sys.executable, "fast_analysis.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Fast analysis failed: {result.stderr}")
        return False
    print("✅ Fast analysis complete!")
    return True

def main():
    """Main workflow."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python workflow.py export    # Run VO once, export data")
        print("  python workflow.py analyze   # Fast analysis (requires export first)")
        print("  python workflow.py full      # Export + analyze")
        return
    
    command = sys.argv[1].lower()
    
    if command == "export":
        run_vo_export()
    elif command == "analyze":
        if not os.path.exists("outputs/raw_data/vo_raw_data.csv"):
            print("❌ No raw data found. Run 'python workflow.py export' first.")
            return
        run_fast_analysis()
    elif command == "full":
        if run_vo_export():
            run_fast_analysis()
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()

