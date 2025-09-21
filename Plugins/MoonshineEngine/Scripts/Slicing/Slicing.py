import os
import sys

# Add library paths to sys.path for embedded Python environment
script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(os.path.dirname(script_dir))
lib_paths = [
    os.path.join(plugin_dir, 'Libraries', 'Slicing'),
    os.path.join(plugin_dir, 'Libraries', 'ObjectRemoval'),
    os.path.join(plugin_dir, 'Libraries', 'DepthMap')
]

for lib_path in lib_paths:
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)


# Import cv2 after sys.path is set up, so local cv2 can be found
import argparse
import cv2
import numpy as np
from PIL import Image


def pad_to_mod8(img):
    """Pad image to be divisible by 8 (required for Lama model)"""
    h, w = img.shape[:2]
    pad_h = (8 - h % 8) if h % 8 != 0 else 0
    pad_w = (8 - w % 8) if w % 8 != 0 else 0
    if pad_h or pad_w:
        if img.ndim == 2:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    return img


def run_lama_inpaint_from_arrays(image_np, mask_np):
    """
    Run Lama inpainting using numpy arrays
    Args:
        image_np: RGB image as numpy array (H, W, 3)
        mask_np: Binary mask as numpy array (H, W), 255 = inpaint area, 0 = keep
    Returns:
        Inpainted image as numpy array (H, W, 3)
    """
    try:
        # Import torch and model loading only when needed
        import torch
        from omegaconf import OmegaConf
        import yaml
        
        # Add LaMa source path
        lama_src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Lama_ObjectRemoval', 'LaMa_src')
        if lama_src_path not in sys.path:
            sys.path.insert(0, lama_src_path)
        from saicinpainting.training.trainers import load_checkpoint
        from saicinpainting.evaluation.utils import move_to_device

    # Setup model paths to new location
        model_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            ),
            'Models', 'Lama', 'big-lama'
        )
        config_path = os.path.join(model_dir, 'config.yaml')
        checkpoint_path = os.path.join(model_dir, 'models', 'best.ckpt')
        perceptual_root = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            ),
            'Models', 'Lama', 'LaMa_perceptual_loss_models'
        )
        
        # Check if required files exist
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return image_np
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return image_np

        # Load model config
        with open(config_path, 'r') as f:
            config = OmegaConf.create(yaml.safe_load(f))
        
        # Override perceptual loss weights path
        try:
            if 'losses' in config and 'resnet_pl' in config.losses:
                if os.path.isdir(perceptual_root):
                    config.losses.resnet_pl.weights_path = perceptual_root
                else:
                    # Disable perceptual loss to avoid FileNotFoundError
                    if 'weight' in config.losses.resnet_pl:
                        config.losses.resnet_pl.weight = 0
        except Exception as e:
            print(f"Warning adjusting perceptual loss path: {e}")

        # Load model
        model = load_checkpoint(config, checkpoint_path, map_location='cpu', strict=False)
        model.eval()

        # Prepare input
        image_padded = pad_to_mod8(image_np)
        mask_padded = pad_to_mod8(mask_np)
        
        batch = {
            'image': torch.from_numpy(image_padded.transpose(2, 0, 1)).float().unsqueeze(0) / 255.,
            'mask': torch.from_numpy(mask_padded[None]).float().unsqueeze(0) / 255.
        }
        batch = move_to_device(batch, 'cpu')
        
        # Run inference
        with torch.no_grad():
            result = model(batch)
            result = result['inpainted'].cpu().numpy()[0]
            result = (result.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        # Crop back to original size
        original_h, original_w = image_np.shape[:2]
        result = result[:original_h, :original_w]
        
        print("Successfully completed Lama inpainting")
        return result
        
    except Exception as e:
        print(f"Error during Lama inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original image if inpainting fails
        return image_np
        print(f"Error during Lama inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original image if inpainting fails
        return image_np


def create_bottom_area_mask(img_rgba, boundary):
    """
    Create a mask for inpainting based only on the bottom area that would be filled.
    This replicates the EXACT same logic as the original fill_bottom_area function but creates a mask instead.
    Args:
        img_rgba: RGBA image where alpha=0 are transparent areas
        boundary: Boundary points for bottom area detection (required)
    Returns:
        mask_np: Binary mask (H, W) where 255 = bottom area to inpaint, 0 = keep original
    """
    height, width = img_rgba.shape[:2]
    # Create a copy of the image to work with (same as original function)
    result = img_rgba.copy()
    if not boundary:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Get median color from boundary pixels (same as original, but we don't need it for mask)
    terrain_colors = []
    for x, y in boundary:
        if 0 <= x < width and 0 <= y < height:
            terrain_colors.append(result[y, x, :3])
    if not terrain_colors:
        return np.zeros((height, width), dtype=np.uint8)
    median_color = np.median(terrain_colors, axis=0).astype(np.uint8)
    
    # Create boundary array - find the topmost boundary for each column (EXACT SAME LOGIC)
    boundary_y = np.full(width, height - 1)  # Default to bottom
    
    for x, y in boundary:
        if 0 <= x < width and 0 <= y < height:
            boundary_y[x] = min(boundary_y[x], y)
    
    # Aggressive gap filling using multiple passes (EXACT SAME LOGIC)
    # Pass 1: Forward interpolation
    for x in range(1, width):
        if boundary_y[x] == height - 1 and boundary_y[x-1] < height - 1:
            # Linear interpolation to next known point
            next_known = x
            for xx in range(x, width):
                if boundary_y[xx] < height - 1:
                    next_known = xx
                    break
            
            if next_known > x:
                # Interpolate between x-1 and next_known
                start_y = boundary_y[x-1]
                end_y = boundary_y[next_known]
                for xx in range(x, next_known):
                    progress = (xx - x + 1) / (next_known - x + 1)
                    boundary_y[xx] = int(start_y + (end_y - start_y) * progress)
    
    # Pass 2: Backward interpolation
    for x in range(width - 2, -1, -1):
        if boundary_y[x] == height - 1 and boundary_y[x+1] < height - 1:
            boundary_y[x] = boundary_y[x+1]
    
    # Pass 3: Smooth the boundary with median filter
    window_size = 5
    smoothed = boundary_y.copy()
    for x in range(width):
        start_x = max(0, x - window_size // 2)
        end_x = min(width, x + window_size // 2 + 1)
        smoothed[x] = int(np.median(boundary_y[start_x:end_x]))
    
    # Scanline fill: for each column, fill from bottom to boundary (MODIFY FOR MASK)
    for x in range(width):
        fill_start = smoothed[x]
        # Fill every pixel from boundary to bottom in the RESULT IMAGE (same as original)
        for y in range(fill_start, height):
            result[y, x, 0:3] = median_color
            result[y, x, 3] = 255
    
    # Additional horizontal gap filling (EXACT SAME LOGIC)
    # For each row in the bottom area, fill horizontal gaps
    for y in range(height):
        # Find filled regions in this row
        filled_regions = []
        start = None
        for x in range(width):
            if result[y, x, 3] == 255:  # Filled pixel
                if start is None:
                    start = x
            else:  # Not filled
                if start is not None:
                    filled_regions.append((start, x - 1))
                    start = None
        if start is not None:
            filled_regions.append((start, width - 1))
        
        # Fill gaps between filled regions if they're small
        for i in range(len(filled_regions) - 1):
            gap_start = filled_regions[i][1] + 1
            gap_end = filled_regions[i + 1][0] - 1
            gap_size = gap_end - gap_start + 1
            
            # Fill small gaps (less than 20 pixels)
            if gap_size > 0 and gap_size <= 20:
                for x in range(gap_start, gap_end + 1):
                    result[y, x, 0:3] = median_color
                    result[y, x, 3] = 255
    
    # NOW CREATE THE MASK from the filled result
    # Compare with original input to see what was filled
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create mask where result differs from original (where filling occurred)
    for y in range(height):
        for x in range(width):
            # If the alpha was changed from 0 to 255, or if RGB was changed, this pixel was filled
            original_alpha = img_rgba[y, x, 3]
            result_alpha = result[y, x, 3]
            
            if original_alpha == 0 and result_alpha == 255:
                # This pixel was filled
                mask[y, x] = 255
            elif original_alpha > 0:
                # Check if RGB was changed (for existing pixels that got filled over)
                original_rgb = img_rgba[y, x, :3]
                result_rgb = result[y, x, :3]
                if not np.array_equal(original_rgb, result_rgb):
                    mask[y, x] = 255
    
    return mask


def fallback_median_fill(img_rgba, inpaint_mask):
    """
    Fallback inpainting method using median color fill
    Args:
        img_rgba: RGBA image 
        inpaint_mask: Binary mask where 255 = areas to fill
    Returns:
        Filled image
    """
    result = img_rgba.copy()
    
    # Get median color from visible pixels
    visible_pixels = result[result[..., 3] > 0][..., 0:3]
    if visible_pixels.size > 0:
        median_color = np.median(visible_pixels, axis=0).astype(np.uint8)
    else:
        median_color = np.array([128, 128, 128], dtype=np.uint8)  # Default gray
    
    # Fill masked areas
    fill_areas = inpaint_mask > 0
    result[fill_areas, 0:3] = median_color
    result[fill_areas, 3] = 255
    
    return result


def connect_horizontal_gaps(img, max_gap_size=200):
    """Connect horizontal pixels across transparent gaps to define terrain boundary (bottom-most non-transparent pixel in each column)"""
    height, width = img.shape[:2]
    has_alpha = img.shape[2] == 4
    terrain_boundary = []
    for x in range(width):
        bottom_y = None
        for y in range(height - 1, -1, -1):
            is_transparent = False
            if has_alpha:
                is_transparent = img[y, x, 3] < 128
            else:
                is_transparent = (img[y, x, 0] > 250 and img[y, x, 1] > 250 and img[y, x, 2] > 250)
            if not is_transparent:
                bottom_y = y
                break
        if bottom_y is not None:
            terrain_boundary.append((x, bottom_y))
        else:
            terrain_boundary.append((x, None))
    segments = []
    current_segment = []
    for x, y in terrain_boundary:
        if y is not None:
            current_segment.append((x, y))
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    if current_segment:
        segments.append(current_segment)
    connected_boundary = []
    for i, segment in enumerate(segments):
        connected_boundary.extend(segment)
        if i < len(segments) - 1:
            next_segment = segments[i + 1]
            gap_start_x = segment[-1][0]
            gap_end_x = next_segment[0][0]
            gap_size = gap_end_x - gap_start_x
            if gap_size <= max_gap_size:
                start_y = segment[-1][1]
                end_y = next_segment[0][1]
                for x in range(gap_start_x + 1, gap_end_x):
                    progress = (x - gap_start_x) / (gap_end_x - gap_start_x)
                    interp_y = int(start_y + (end_y - start_y) * progress)
                    connected_boundary.append((x, interp_y))
    return connected_boundary


def process_depth_map(depth_path, color_path, outdir, section_ranges=None, use_lama_inpaint=True):
    """
    Reads a depth map and color image, divides into 4 sections by normalized depth, and saves RGBA PNGs for each section.
    Uses Lama inpainting for missing areas instead of simple color filling.
    Args:
        depth_path (str): Path to depth map image (single-channel, e.g. PNG).
        color_path (str): Path to color image (RGB).
        outdir (str): Output directory for sectioned images.
        section_ranges (list of tuple): List of (low, high) tuples for depth sections.
        use_lama_inpaint (bool): Whether to use Lama inpainting (True) or fallback to median color fill (False).
    """
    os.makedirs(outdir, exist_ok=True)
    # Read images
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if depth is None or color is None:
        raise ValueError("Could not read input images.")

    print(f"Original depth shape: {depth.shape}")
    print(f"Color shape: {color.shape}")

    # Extract base name of color image (without extension)
    color_base = os.path.splitext(os.path.basename(color_path))[0]

    # Ensure depth is single channel
    if len(depth.shape) == 3:
        # If depth has multiple channels, convert to grayscale
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        print(f"Converted depth to grayscale, new shape: {depth.shape}")

    depth = depth.astype(np.float32)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    print(f"Depth value, min: {depth.min()}, max: {depth.max()}")

    # Convert color to RGB for Lama (OpenCV loads as BGR)
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    # Build section ranges if not provided
    if section_ranges is None:
        section_ranges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    total_sections = len(section_ranges)

    for idx, (low, high) in enumerate(section_ranges, 1):
        print(f"Processing section {idx}/{total_sections}: depth range [{low:.2f}, {high:.2f}]")

        # Create mask from depth - ensure it's 2D
        if len(depth_norm.shape) == 3:
            depth_for_mask = depth_norm[:, :, 0]
        else:
            depth_for_mask = depth_norm
        mask = (depth_for_mask >= low) & (depth_for_mask < high) if high < 1.0 else (depth_for_mask >= low) & (depth_for_mask <= high)
        print(f"Section {idx}: depth_for_mask shape: {depth_for_mask.shape}, mask shape: {mask.shape}")

        # Apply morphological erosion to remove thin edge artifacts
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)

        # Create RGBA image
        color_section_img = np.zeros((color.shape[0], color.shape[1], 4), dtype=np.uint8)
        color_section_img[..., 0:3] = color[..., 0:3]
        color_section_img[..., 3] = mask_eroded
        out_path = os.path.join(outdir, f"{color_base}_section_{idx}.png")
        cv2.imwrite(out_path, color_section_img)
        print(f"Saved section {idx} to: {out_path}")

        # Create inpainted version using appropriate strategy based on section position
        img_rgba = color_section_img.copy()

        # Determine inpainting strategy based on section position
        is_first_section = (idx == 1)
        is_last_section = (idx == total_sections)

        if is_first_section:
            # First section (foreground): Inpaint ALL transparent areas
            print(f"Section {idx}: First section - inpainting all transparent areas")
            # Create mask for all transparent areas
            inpaint_mask = (img_rgba[..., 3] == 0).astype(np.uint8) * 255
        elif is_last_section:
            # Last section (background): No inpainting needed
            print(f"Section {idx}: Last section - no inpainting, keeping original as-is")
            # Create empty mask
            inpaint_mask = np.zeros((color.shape[0], color.shape[1]), dtype=np.uint8)
            # For last section, don't do any processing - keep original
            inpainted_img = img_rgba.copy()
        else:
            # Middle sections: Inpaint only bottom areas
            print(f"Section {idx}: Middle section - inpainting bottom areas only")
            boundary = connect_horizontal_gaps(img_rgba)
            inpaint_mask = create_bottom_area_mask(img_rgba, boundary)

        # Expand the inpainting mask using dilation (applies to all but last section)
        if not is_last_section and np.sum(inpaint_mask) > 0:
            dilation_kernel_size = 7  # You can adjust this value for more/less expansion
            kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
            inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

        # Check if there's anything to inpaint (skip for last section)
        if not is_last_section and np.sum(inpaint_mask) > 0:
            print(f"Section {idx}: Found {np.sum(inpaint_mask > 0)} pixels to inpaint")

            if use_lama_inpaint:
                # Try Lama inpainting with the original full image
                print(f"Section {idx}: Running Lama inpainting...")
                inpainted_rgb = run_lama_inpaint_from_arrays(color_rgb, inpaint_mask)

                # Check if Lama inpainting was successful (compare with original)
                if not np.array_equal(inpainted_rgb, color_rgb):
                    # Convert back to BGR for saving
                    inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)

                    # Create final RGBA image with inpainted content
                    inpainted_img = np.zeros((color.shape[0], color.shape[1], 4), dtype=np.uint8)
                    inpainted_img[..., 0:3] = inpainted_bgr[..., 0:3]
                    inpainted_img[..., 3] = mask_eroded  # Keep the original section mask for alpha

                    # For areas that were inpainted, make sure they are opaque
                    inpaint_areas = inpaint_mask > 0
                    inpainted_img[inpaint_areas, 3] = 255

                    print(f"Section {idx}: Successfully used Lama inpainting")
                else:
                    # Lama failed, use fallback
                    print(f"Section {idx}: Lama inpainting failed, using fallback method")
                    inpainted_img = fallback_median_fill(img_rgba, inpaint_mask)
            else:
                # Use fallback method directly
                print(f"Section {idx}: Using fallback median fill")
                inpainted_img = fallback_median_fill(img_rgba, inpaint_mask)
        else:
            if is_last_section:
                print(f"Section {idx}: Last section - no processing needed")
                # Last section already handled above, inpainted_img is already set
            else:
                print(f"Section {idx}: No areas to inpaint")
                # No inpainting needed, but handle any remaining transparent areas with median color
                visible_pixels = img_rgba[img_rgba[..., 3] > 0][..., 0:3]
                if visible_pixels.size > 0:
                    median_color = np.median(visible_pixels, axis=0).astype(np.uint8)
                    transparent_mask = img_rgba[..., 3] == 0
                    img_rgba[transparent_mask, 0:3] = median_color
                    img_rgba[transparent_mask, 3] = 255
                inpainted_img = img_rgba

        # Save final inpainted result
        inpainted_path = os.path.join(outdir, f"{color_base}_section_{idx}_inpainted.png")
        cv2.imwrite(inpainted_path, inpainted_img)
        print(f"Saved inpainted section {idx} to: {inpainted_path}")
        print(f"Section {idx} processing complete\n")

def run_slicing_pipeline(depth_path, color_path, outdir, section_ranges=None, use_lama_inpaint=True):
    """
    Complete slicing pipeline that can be called from Unreal Engine
    Args:
        depth_path (str): Path to depth map image
        color_path (str): Path to color image  
        outdir (str): Output directory
        section_ranges (list): Optional custom section ranges
        use_lama_inpaint (bool): Whether to use Lama inpainting or fallback method
    """
    try:
        process_depth_map(depth_path, color_path, outdir, section_ranges, use_lama_inpaint)
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
    parser.add_argument('--use-lama', action='store_true', default=True, help='Use Lama inpainting (default: True)')
    parser.add_argument('--no-lama', action='store_true', help='Disable Lama inpainting, use fallback method')
    args = parser.parse_args()

    # Determine whether to use Lama inpainting
    use_lama_inpaint = args.use_lama and not args.no_lama

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

    process_depth_map(args.depth_path, args.color_path, args.outdir, section_ranges, use_lama_inpaint)