#!/usr/bin/env python3
"""
Check the actual CSV data to understand the coordinate ranges
"""

import pandas as pd

def check_csv_data():
    """Check the CSV data ranges"""
    
    print("🔍 Checking CSV data ranges...")
    
    # Check VO trajectory CSV
    try:
        vo_df = pd.read_csv('outputs/vo_no_imu_latlon.csv')
        print(f"\n📊 VO trajectory CSV:")
        print(f"  Points: {len(vo_df)}")
        print(f"  Lat range: {vo_df['lat'].min():.6f} to {vo_df['lat'].max():.6f}")
        print(f"  Lon range: {vo_df['lon'].min():.6f} to {vo_df['lon'].max():.6f}")
        print(f"  Lat span: {vo_df['lat'].max() - vo_df['lat'].min():.6f} degrees")
        print(f"  Lon span: {vo_df['lon'].max() - vo_df['lon'].min():.6f} degrees")
        
        # Check if this is reasonable
        lat_span = vo_df['lat'].max() - vo_df['lat'].min()
        lon_span = vo_df['lon'].max() - vo_df['lon'].min()
        
        if lat_span > 0.01 or lon_span > 0.01:  # More than 0.01 degrees
            print(f"⚠️  WARNING: Trajectory span is very large!")
            print(f"   This suggests the VO algorithm is producing a huge trajectory")
        else:
            print(f"✅ Trajectory span looks reasonable")
            
    except FileNotFoundError:
        print("❌ VO trajectory CSV not found")
    
    # Check GPS data
    try:
        from csv_utils import load_imu_csv
        from config import CONFIG
        
        cfg = CONFIG
        imu_df = load_imu_csv(cfg["PATH_IMU_CSV"])
        
        if "lat" in imu_df.columns and "lon" in imu_df.columns:
            print(f"\n📊 GPS data:")
            print(f"  Points: {len(imu_df)}")
            print(f"  Lat range: {imu_df['lat'].min():.6f} to {imu_df['lat'].max():.6f}")
            print(f"  Lon range: {imu_df['lon'].min():.6f} to {imu_df['lon'].max():.6f}")
            print(f"  Lat span: {imu_df['lat'].max() - imu_df['lat'].min():.6f} degrees")
            print(f"  Lon span: {imu_df['lon'].max() - imu_df['lon'].min():.6f} degrees")
        else:
            print("❌ No GPS data found in IMU CSV")
            
    except Exception as e:
        print(f"❌ Error loading GPS data: {e}")

if __name__ == "__main__":
    check_csv_data()
