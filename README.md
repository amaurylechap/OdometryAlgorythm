# Visual Odometry Algorithm - Modular Structure

This project has been refactored from a single large script into a modular structure for better maintainability and organization.

## File Structure

### Core Modules

- **`config.py`** - Configuration parameters and settings
- **`main.py`** - Main entry point that orchestrates the complete pipeline
- **`vo_algorithm.py`** - Core Visual Odometry algorithm implementation

### Utility Modules

- **`image_utils.py`** - Image loading, preprocessing, and distortion correction
- **`features.py`** - Feature detection (FAST, ORB) and matching
- **`tracking.py`** - Lucas-Kanade tracking and motion estimation
- **`csv_utils.py`** - CSV data loading and geographic utilities
- **`visualization.py`** - Plotting, progress bars, and mosaic generation

## Usage

### Basic Usage
```bash
python main.py
```

### Output Organization
All generated files are automatically saved to the `outputs/` directory:
- **Plots**: Trajectory, GPS, IMU, and VO metric visualizations
- **Data**: CSV files with tracking data and coordinates
- **Documentation**: README explaining each output file

### Configuration
Edit `config.py` to modify:
- Image paths and processing parameters
- Feature detection settings
- Tracking parameters
- Output file paths (all default to `outputs/` directory)
- GPS/IMU integration settings

## Key Features

### Visual Odometry Pipeline
- FAST feature detection with adaptive parameters
- Lucas-Kanade optical flow tracking
- RANSAC-based motion estimation
- ORB fallback for relocalization
- Keyframe-based trajectory building

### IMU Integration
- Robust CSV reading with auto-detection of column formats
- Automatic time synchronization between frames and IMU data
- Simple roll/pitch compensation for trajectory correction
- GPS coordinate conversion and plotting

### Outputs
- Pixel trajectory plots
- VO metrics with/without IMU compensation
- GPS track visualization
- IMU angle plots
- CSV exports for further analysis
- Optional mosaic generation

## Dependencies

```bash
pip install opencv-python numpy matplotlib pandas
```

Optional for progress bars:
```bash
pip install tqdm
```

## Configuration Examples

### Basic Settings
```python
CONFIG = {
    "images_dir": "/path/to/images",
    "pattern": "*.pgm",
    "PROCESS_SCALE": 0.5,
    "use_undistort": True,
    # ... more settings
}
```

### IMU/GPS Integration
```python
CONFIG = {
    "PATH_IMU_CSV": "imu_data.csv",
    "PATH_FRAME_CSV": "frame_timestamps.csv",
    "TIME_OFFSET_USER_S": 0.0,
    "SCALE_M_PER_PX": 0.1,  # meters per pixel
}
```

## Module Responsibilities

### `config.py`
- Centralized configuration management
- All user-configurable parameters
- Default values and validation

### `main.py`
- Entry point and high-level orchestration
- CSV data loading and processing
- VO metric calculations and plotting
- Output file generation

### `vo_algorithm.py`
- Core VO algorithm implementation
- Frame processing pipeline
- Feature tracking and motion estimation
- Trajectory building

### `image_utils.py`
- Image loading and preprocessing
- Distortion correction
- CLAHE and unsharp mask filtering
- Format conversion utilities

### `features.py`
- FAST feature detection
- ORB fallback and relocalization
- Feature format handling
- Adaptive parameter management

### `tracking.py`
- Lucas-Kanade optical flow
- Motion estimation and RANSAC
- Affine transformation handling
- Adaptive parameter controller

### `csv_utils.py`
- Robust CSV reading with multiple delimiter support
- Time normalization and synchronization
- GPS coordinate conversion
- IMU data processing

### `visualization.py`
- Trajectory and metric plotting
- Progress bar utilities
- Mosaic generation
- Output visualization

## Benefits of Modular Structure

1. **Maintainability** - Each module has a clear, focused responsibility
2. **Testability** - Individual modules can be tested in isolation
3. **Reusability** - Modules can be imported and used independently
4. **Readability** - Code is organized logically with clear interfaces
5. **Extensibility** - New features can be added to appropriate modules
6. **Debugging** - Issues can be isolated to specific modules

## Migration from Original Script

The original monolithic script has been split while preserving all functionality:
- All configuration options remain the same
- Output formats and file names are unchanged
- Performance characteristics are maintained
- The main entry point (`main.py`) provides the same interface

## Future Enhancements

The modular structure makes it easy to:
- Add new feature detectors
- Implement different tracking algorithms
- Add new visualization options
- Integrate additional sensor data
- Create unit tests for individual components
