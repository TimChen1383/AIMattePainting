
# Dynamically add library paths for embedded Python
import os
import sys

# Add library paths to sys.path for embedded Python environment
script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(os.path.dirname(script_dir))
lib_paths = [
    os.path.join(plugin_dir, 'Libraries', 'DepthMap'),
    script_dir,  # Add the script directory itself to find depth_anything_v2 module
    # Add other library folders here if needed
]

for lib_path in lib_paths:
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)
        print(f"Added to sys.path: {lib_path}")
    else:
        print(f"Library path not found or already in sys.path: {lib_path}")

# Debug: Print current working directory and script location
print(f"Script directory: {script_dir}")
print(f"Plugin directory: {plugin_dir}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[:3]}...")  # Print first 3 paths to avoid clutter

import argparse
try:
    import cv2
    print("Successfully imported cv2")
except ImportError as e:
    print(f"Failed to import cv2: {e}")
    sys.exit(1)

import glob
try:
    import matplotlib
    print("Successfully imported matplotlib")
except ImportError as e:
    print(f"Failed to import matplotlib: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("Successfully imported numpy")
except ImportError as e:
    print(f"Failed to import numpy: {e}")
    sys.exit(1)

try:
    import torch
    print("Successfully imported torch")
except ImportError as e:
    print(f"Failed to import torch: {e}")
    sys.exit(1)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print("Successfully imported DepthAnythingV2")
except ImportError as e:
    print(f"Failed to import DepthAnythingV2: {e}")
    sys.exit(1)


def run_depth_estimation(img_path, input_size=518, outdir='./vis_depth', encoder='vitl', pred_only=True, grayscale=True):
    """
    Complete pipeline that can be called from Unreal Engine
    """
    try:
        # Simulate command line arguments
        class Args:
            def __init__(self):
                self.img_path = img_path
                self.input_size = input_size
                self.outdir = outdir
                self.encoder = encoder
                self.pred_only = pred_only
                self.grayscale = grayscale
        
        args = Args()
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {DEVICE}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        depth_anything = DepthAnythingV2(**model_configs[args.encoder])
        
        # Use new absolute path for checkpoint
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Models', 'Depth_Anything_V2', 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, f'depth_anything_v2_{args.encoder}.pth')
        print(f"Loading checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            return False
        depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()
        print("Model loaded successfully")
        
        # Validate input path
        if not os.path.isfile(args.img_path):
            print(f"ERROR: Input image file does not exist or is not a file: {args.img_path}")
            return False
        # Only process one image
        os.makedirs(args.outdir, exist_ok=True)
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        try:
            raw_image = cv2.imread(args.img_path)
            if raw_image is None:
                print(f"ERROR: Could not read image: {args.img_path}")
                return False
            depth = depth_anything.infer_image(raw_image, args.input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            output_filename = os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(args.img_path))[0] + '_depth.png'
            )
            if args.pred_only:
                cv2.imwrite(output_filename, depth)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth])
                cv2.imwrite(output_filename, combined_result)
            print(f"Processed: {args.img_path} -> {output_filename}")
            return True
        except Exception as e:
            print(f"Error processing {args.img_path}: {e}")
            return False
        
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Depth Anything V2')
        
        parser.add_argument('--img-path', type=str)
        parser.add_argument('--input-size', type=int, default=518)
        parser.add_argument('--outdir', type=str, default='./vis_depth')
        
        parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
        
        parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
        parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
        
        args = parser.parse_args()
        print(f"Arguments: {args}")
        
        if not args.img_path:
            print("ERROR: --img-path is required")
            sys.exit(1)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {DEVICE}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        depth_anything = DepthAnythingV2(**model_configs[args.encoder])
        
        # Use new absolute path for checkpoint
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Models', 'Depth_Anything_V2', 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, f'depth_anything_v2_{args.encoder}.pth')
        print(f"Loading checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            print(f"Available files in checkpoints directory:")
            if os.path.exists(checkpoint_dir):
                for file in os.listdir(checkpoint_dir):
                    print(f"  - {file}")
            else:
                print(f"  Checkpoints directory does not exist: {checkpoint_dir}")
            sys.exit(1)
        try:
            depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print("Successfully loaded checkpoint")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            sys.exit(1)
        depth_anything = depth_anything.to(DEVICE).eval()
        
        # Validate input path
        print(f"Input path: {args.img_path}")
        if not os.path.isfile(args.img_path):
            print(f"ERROR: Input image file does not exist or is not a file: {args.img_path}")
            sys.exit(1)
        os.makedirs(args.outdir, exist_ok=True)
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        try:
            raw_image = cv2.imread(args.img_path)
            if raw_image is None:
                print(f"ERROR: Could not read image: {args.img_path}")
                sys.exit(1)
            print(f"Processing image with shape: {raw_image.shape}")
            depth = depth_anything.infer_image(raw_image, args.input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            output_filename = os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(args.img_path))[0] + '_depth.png'
            )
            if args.pred_only:
                cv2.imwrite(output_filename, depth)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth])
                cv2.imwrite(output_filename, combined_result)
            print(f"Successfully saved: {output_filename}")
            print("Processing completed!")
        except Exception as e:
            print(f"ERROR processing {args.img_path}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)