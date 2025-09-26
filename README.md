# Visual Odometry Algorithm

## 🎯 Quick Start

### Single Command Workflow
```bash
python vo_workflow.py
```

## 📋 Workflow Options

### Option 1: Complete Analysis (First Time)
- Runs full VO analysis on all images
- Saves parameters to `outputs/vo_params.pkl`
- Generates all output files
- **Use once** - takes time for large datasets

### Option 2: Generate Outputs (Fast)
- Uses saved VO parameters
- Generates outputs instantly
- **Use for** changing settings, new plots

## 📁 Output Files

- `outputs/mosaic.png` - Flight path mosaic
- `outputs/mosaic_traj.png` - Mosaic with trajectory
- `outputs/traj_xy.png` - Trajectory plot  
- `outputs/plane_series.png` - Aircraft frame analysis
- `outputs/vo_params.pkl` - Saved VO parameters

## ⚙️ Configuration

Edit `config.py` to adjust:

### Image Settings
```python
"images_dir": r"C:\path\to\your\images",
"FRAME_FILENAME_REGEX": r"frame_(\d+)\.",
```

### Mosaic Settings
```python
"ENABLE_MOSAIC": True,
"MOSAIC_STRIDE": 100,        # Higher = faster, less detail
"MOSAIC_MODE": "thumbnail",   # "outline" | "thumbnail" | "image"
"MOSAIC_RENDER_SCALE": 0.6,
```

### Output Paths
```python
"mosaic_png": "outputs/mosaic.png",
"mosaic_with_traj_png": "outputs/mosaic_traj.png",
"traj_png_xy": "outputs/traj_xy.png",
"plane_series_png": "outputs/plane_series.png",
```

## 🚀 For Large Datasets (10,000+ images)

### Step 1: Run Analysis Once
```bash
python vo_workflow.py
# Choose option 1
```

### Step 2: Generate Outputs Quickly
```bash
python vo_workflow.py  
# Choose option 2
```

### Memory Optimization
- Increase `MOSAIC_STRIDE` (100, 200, 500) for faster processing
- Use `MOSAIC_MODE: "thumbnail"` for smaller files
- Adjust `MOSAIC_RENDER_SCALE` (0.3-0.8) for file size

## 📊 Performance Tips

### Fast Processing
- `MOSAIC_STRIDE: 200` - Every 200th frame
- `MOSAIC_MODE: "thumbnail"` - Smaller images
- `MOSAIC_RENDER_SCALE: 0.4` - 40% size

### High Quality
- `MOSAIC_STRIDE: 50` - Every 50th frame  
- `MOSAIC_MODE: "image"` - Full images
- `MOSAIC_RENDER_SCALE: 0.8` - 80% size

### Memory Issues
- Increase `MOSAIC_STRIDE` to 500+
- Use `MOSAIC_MODE: "outline"` for minimal memory
- Process in smaller batches

## 🔧 Troubleshooting

### "No saved parameters found"
- Run option 1 first to save parameters

### "Memory error" 
- Increase `MOSAIC_STRIDE` in config.py
- Use `MOSAIC_MODE: "outline"`

### "Image directory not found"
- Check `images_dir` path in config.py
- Ensure images exist in the folder

### Slow processing
- Increase `MOSAIC_STRIDE` 
- Use `MOSAIC_MODE: "thumbnail"`
- Reduce `MOSAIC_RENDER_SCALE`

## 📈 Understanding Outputs

### Mosaic
- Visual representation of flight path
- Each frame placed at its trajectory position
- Shows ground coverage and flight pattern

### Trajectory Plot
- 2D path of aircraft movement
- X/Y coordinates over time
- Shows flight direction and distance

### Plane Series
- Aircraft orientation analysis
- Shows pitch, roll, yaw changes
- Useful for flight dynamics analysis

## 🎯 Best Practices

1. **First Run**: Use option 1, let it complete
2. **Subsequent Runs**: Use option 2 for speed
3. **Large Datasets**: Start with high `MOSAIC_STRIDE` (200+)
4. **Quality vs Speed**: Adjust stride based on needs
5. **Memory Issues**: Use outline mode for very large datasets

## 📝 File Structure

```
project/
├── vo_workflow.py          # Main workflow script
├── config.py               # Configuration settings
├── main.py                 # Original VO algorithm
├── vo_algorithm.py         # Core VO implementation
├── visualization.py        # Plot generation
├── outputs/                # Generated files
│   ├── vo_params.pkl       # Saved VO parameters
│   ├── mosaic.png          # Flight mosaic
│   ├── mosaic_traj.png     # Mosaic with trajectory
│   ├── traj_xy.png         # Trajectory plot
│   └── plane_series.png    # Plane analysis
└── README.md               # This file
```