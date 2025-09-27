"""
Smart Memory-Efficient Mosaic Generator
Processes frames in small batches to avoid memory issues
"""

import cv2
import numpy as np
import os
from config import CONFIG as cfg
from tracking import corners, to33, to23

def smart_mosaic_generator(global_A, pose_frame_ids, paths, W0, H0, W_full, H_full, dc, config, traj_xy=None):
    """Generate mosaic using smart batching to avoid memory issues."""
    
    if not config.get("ENABLE_MOSAIC", False):
        print("ℹ️ Mosaic disabled (ENABLE_MOSAIC=False).")
        return
    
    print("🧠 Smart mosaic generation with memory optimization...")
    
    # Smart parameters
    stride = max(1, int(config.get("MOSAIC_STRIDE", 1000)))
    render_scale = max(0.05, min(1.0, float(config.get("MOSAIC_RENDER_SCALE", 0.1))))
    batch_size = 50  # Process only 50 frames at a time
    max_canvas_size = 20000  # Maximum canvas dimension to prevent memory issues
    
    # Get frame indices to process
    render_indices = list(range(0, len(global_A), stride))
    print(f"📊 Processing {len(render_indices)} frames with stride {stride}")
    
    if len(render_indices) == 0:
        print("⚠️ No frames to process")
        return
    
    # Calculate canvas bounds from trajectory + sample frames
    print(f"🔍 Calculating bounds from trajectory + sample frames...")
    W0_plot, H0_plot = W0, H0
    cs = corners(W0_plot, H0_plot)
    
    # Include trajectory bounds
    if traj_xy is not None and len(traj_xy) > 0:
        traj_array = np.array(traj_xy, dtype=np.float32)
        traj_minx, traj_miny = np.floor(traj_array.min(axis=0)).astype(int)
        traj_maxx, traj_maxy = np.ceil(traj_array.max(axis=0)).astype(int)
        print(f"📊 Trajectory bounds: ({traj_minx}, {traj_miny}) to ({traj_maxx}, {traj_maxy})")
    else:
        traj_minx = traj_miny = traj_maxx = traj_maxy = 0
    
    # Include frame bounds from sample
    sample_size = min(100, len(render_indices))
    sample_indices = render_indices[:sample_size]
    
    warped_corners = []
    for i in sample_indices:
        A = global_A[i]
        corners_warped = cv2.transform(cs, A).reshape(-1, 2)
        warped_corners.append(corners_warped)
    
    if warped_corners:
        allc = np.vstack(warped_corners)
        frame_minx, frame_miny = np.floor(allc.min(axis=0)).astype(int)
        frame_maxx, frame_maxy = np.ceil(allc.max(axis=0)).astype(int)
        print(f"📊 Frame bounds: ({frame_minx}, {frame_miny}) to ({frame_maxx}, {frame_maxy})")
    else:
        frame_minx = frame_miny = frame_maxx = frame_maxy = 0
    
    # Use combined bounds
    minx = min(traj_minx, frame_minx)
    miny = min(traj_miny, frame_miny)
    maxx = max(traj_maxx, frame_maxx)
    maxy = max(traj_maxy, frame_maxy)
    
    # Add margin and calculate final canvas size
    margin = int(config.get("canvas_margin", 100))
    offset = np.array([margin - minx, margin - miny], dtype=np.float32)
    CW = int((maxx - minx) + 2 * margin)
    CH = int((maxy - miny) + 2 * margin)
    
    # Limit canvas size to prevent memory issues
    if CW > max_canvas_size or CH > max_canvas_size:
        scale_factor = min(max_canvas_size / CW, max_canvas_size / CH)
        CW = int(CW * scale_factor)
        CH = int(CH * scale_factor)
        render_scale *= scale_factor
        print(f"⚠️ Canvas too large, scaling down by {scale_factor:.3f}")
    
    # Apply render scale
    CW_r = int(max(1, round(CW * render_scale)))
    CH_r = int(max(1, round(CH * render_scale)))
    
    print(f"📐 Canvas size: {CW_r}x{CH_r} (scale: {render_scale:.3f})")
    
    # Initialize canvas
    canvas = np.zeros((CH_r, CW_r, 3), dtype=np.uint8)
    S = np.array([[render_scale, 0, 0], [0, render_scale, 0], [0, 0, 1]], dtype=np.float32)
    
    # Process frames in batches
    mode = config.get("MOSAIC_MODE", "thumbnail").lower().strip()
    alpha = float(config.get("alpha", 0.5))
    
    print(f"🎨 Rendering in batches of {batch_size} frames...")
    
    for batch_start in range(0, len(render_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(render_indices))
        batch_indices = render_indices[batch_start:batch_end]
        
        print(f"📦 Processing batch {batch_start//batch_size + 1}/{(len(render_indices)-1)//batch_size + 1} ({len(batch_indices)} frames)")
        
        for j in batch_indices:
            try:
                idx_path = pose_frame_ids[j]
                if idx_path < 0 or idx_path >= len(paths):
                    continue
                
                p = paths[idx_path]
                if not os.path.exists(p):
                    continue
                
                A = global_A[j].astype(np.float32)
                A_can = A.copy()
                A_can[0, 2] += offset[0]
                A_can[1, 2] += offset[1]
                A_can_r = to23(S @ to33(A_can))
                
                if mode == "outline":
                    # Draw outline
                    rect = corners(W0_plot, H0_plot)
                    rect_warped = cv2.transform(rect, A_can_r).reshape(-1, 2)
                    rect_warped = rect_warped.astype(int)
                    
                    # Draw rectangle outline
                    for i in range(4):
                        pt1 = tuple(rect_warped[i])
                        pt2 = tuple(rect_warped[(i+1)%4])
                        if (0 <= pt1[0] < CW_r and 0 <= pt1[1] < CH_r and 
                            0 <= pt2[0] < CW_r and 0 <= pt2[1] < CH_r):
                            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)
                
                elif mode == "thumbnail":
                    # Load and warp image (same logic as original)
                    img_full = cv2.imread(p)
                    if img_full is None:
                        continue
                    
                    # Resize to W0, H0 if needed (same as original)
                    img_proc = cv2.resize(img_full, (W0_plot, H0_plot), interpolation=cv2.INTER_AREA) if (W0_plot != W_full or H0_plot != H_full) else img_full
                    
                    # Apply source warp scale if configured (same as original)
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
                    
                    # Warp image with same parameters as original
                    warped = cv2.warpAffine(img_src, A_can_r_adj, (CW_r, CH_r),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                    
                    # Blend with canvas
                    mask = (warped.sum(axis=2) > 0)
                    if np.any(mask):
                        for c in range(3):
                            canvas[:, :, c] = np.where(mask, 
                                                      (canvas[:, :, c] * (1 - alpha) + warped[:, :, c] * alpha).astype(np.uint8),
                                                      canvas[:, :, c])
                
                elif mode == "image":
                    # Full image mode (same logic as original)
                    img_full = cv2.imread(p)
                    if img_full is None:
                        continue
                    
                    # Resize to W0, H0 if needed (same as original)
                    img_proc = cv2.resize(img_full, (W0_plot, H0_plot), interpolation=cv2.INTER_AREA) if (W0_plot != W_full or H0_plot != H_full) else img_full
                    
                    # Warp image with same parameters as original
                    warped = cv2.warpAffine(img_proc, A_can_r, (CW_r, CH_r),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                    
                    # Blend with canvas
                    mask = (warped.sum(axis=2) > 0)
                    if np.any(mask):
                        for c in range(3):
                            canvas[:, :, c] = np.where(mask, 
                                                      (canvas[:, :, c] * (1 - alpha) + warped[:, :, c] * alpha).astype(np.uint8),
                                                      canvas[:, :, c])
                
            except Exception as e:
                print(f"⚠️ Error processing frame {j}: {e}")
                continue
    
    # Save mosaic
    mosaic_path = config.get("mosaic_png", "outputs/mosaic.png")
    os.makedirs(os.path.dirname(mosaic_path), exist_ok=True)
    cv2.imwrite(mosaic_path, canvas)
    print(f"✅ Mosaic saved: {mosaic_path}")
    
    # Generate mosaic with trajectory if requested
    if config.get("mosaic_with_traj_png") and traj_xy is not None and len(traj_xy) > 0:
        print("🎯 Adding trajectory overlay...")
        try:
            # Use the pre-computed trajectory (successive frame center positions)
            traj_xy_array = np.array(traj_xy, dtype=np.float32)
            traj_px = (traj_xy_array + offset) * render_scale
            pts = traj_px.astype(np.int32).reshape(-1, 1, 2)
            
            canvas_traj = canvas.copy()
            cv2.polylines(canvas_traj, [pts], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            
            traj_path_out = config.get("mosaic_with_traj_png", "outputs/mosaic_traj.png")
            cv2.imwrite(traj_path_out, canvas_traj)
            print(f"✅ Mosaic with trajectory saved: {traj_path_out}")
            print(f"📊 Trajectory has {len(traj_xy)} points")
                
        except Exception as e:
            print(f"⚠️ Could not add trajectory: {e}")
    
    print("🎉 Smart mosaic generation complete!")
