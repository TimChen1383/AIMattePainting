import os
import sys

# Add library paths to sys.path for embedded Python environment
script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(os.path.dirname(script_dir))
lib_paths = [
    os.path.join(plugin_dir, 'Libraries', 'Slicing'),
    #os.path.join(plugin_dir, 'Libraries', 'DepthMap')
]

for lib_path in lib_paths:
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

import argparse
import cv2
import numpy as np


def process_depth_map(depth_path, color_path, outdir, section_ranges=None):
    """
    Reads a depth map and color image, divides into 4 sections by normalized depth, and saves RGBA PNGs for each section.
    Args:
        depth_path (str): Path to depth map image (single-channel, e.g. PNG).
        color_path (str): Path to color image (RGB).
        outdir (str): Output directory for sectioned images.
        section_ranges (list of tuple): List of (low, high) tuples for depth sections.
    """
    os.makedirs(outdir, exist_ok=True)
    # Read images
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if depth is None or color is None:
        raise ValueError("Could not read input images.")
    
    print(f"Original depth shape: {depth.shape}")
    print(f"Color shape: {color.shape}")
    
    # Ensure depth is single channel
    if len(depth.shape) == 3:
        # If depth has multiple channels, convert to grayscale
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        print(f"Converted depth to grayscale, new shape: {depth.shape}")
    

    depth = depth.astype(np.float32)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    print(f"Depth value, min: {depth.min()}, max: {depth.max()}")
    # Build section ranges if not provided
    if section_ranges is None:
        section_ranges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    
    for idx, (low, high) in enumerate(section_ranges, 1):
        # Create mask from depth - ensure it's 2D
        if len(depth_norm.shape) == 3:
            # If depth has multiple channels, use only the first one
            depth_for_mask = depth_norm[:, :, 0]
        else:
            depth_for_mask = depth_norm
            
        mask = (depth_for_mask >= low) & (depth_for_mask < high) if high < 1.0 else (depth_for_mask >= low) & (depth_for_mask <= high)
        
        print(f"Section {idx}: depth_for_mask shape: {depth_for_mask.shape}, mask shape: {mask.shape}")
        
        # Create RGBA image
        color_section_img = np.zeros((color.shape[0], color.shape[1], 4), dtype=np.uint8)
        color_section_img[..., 0:3] = color[..., 0:3]
        color_section_img[..., 3] = (mask * 255).astype(np.uint8)
        out_path = os.path.join(outdir, f"section_{idx}.png")
        cv2.imwrite(out_path, color_section_img)
        print(f"Saved section {idx} to: {out_path}")

def run_slicing_pipeline(depth_path, color_path, outdir, section_ranges=None):
    """
    Complete slicing pipeline that can be called from Unreal Engine
    """
    try:
        process_depth_map(depth_path, color_path, outdir, section_ranges)
        print(f"Successfully processed depth slicing to: {outdir}")
        return True
    except Exception as e:
        print(f"Slicing pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Divide depth map into RGBA sections')
    parser.add_argument('--depth-path', type=str, required=True, help='Path to depth map image')
    parser.add_argument('--color-path', type=str, required=True, help='Path to color image')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--section-ranges', type=str, required=False, help='Comma-separated list of section boundaries (e.g. "0.1,0.2,0.6")')
    args = parser.parse_args()

    # Parse section ranges
    section_ranges = None
    if args.section_ranges:
        try:
            boundaries = [float(x) for x in args.section_ranges.split(',') if x.strip()]
            boundaries = sorted(boundaries)
            # Ensure boundaries are within [0,1]
            boundaries = [b for b in boundaries if 0.0 <= b <= 1.0]
            # Build ranges: (0, b1), (b1, b2), ..., (bn, 1)
            section_ranges = []
            prev = 0.0
            for b in boundaries:
                section_ranges.append((prev, b))
                prev = b
            section_ranges.append((prev, 1.0))
        except Exception as e:
            print(f"Error parsing section ranges: {e}")
            section_ranges = None

    process_depth_map(args.depth_path, args.color_path, args.outdir, section_ranges)