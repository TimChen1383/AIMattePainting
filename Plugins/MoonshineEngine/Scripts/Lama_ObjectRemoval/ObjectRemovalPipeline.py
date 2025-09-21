import os
import sys

# Add library paths to sys.path for embedded Python environment
script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(os.path.dirname(script_dir))
lib_paths = [
    os.path.join(plugin_dir, 'Libraries', 'ObjectRemoval'),
    os.path.join(plugin_dir, 'Libraries', 'DepthMap')
]

for lib_path in lib_paths:
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

from PIL import Image
import numpy as np

def pad_to_mod8(img):
    h, w = img.shape[:2]
    pad_h = (8 - h % 8) if h % 8 != 0 else 0
    pad_w = (8 - w % 8) if w % 8 != 0 else 0
    if pad_h or pad_w:
        if img.ndim == 2:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    return img

def load_mask(path):
    mask_img = Image.open(path).convert('RGB')
    mask_np = np.array(mask_img)
    # Red area: R > 127, G < 127, B < 127
    red_mask = ((mask_np[..., 0] > 127) & (mask_np[..., 1] < 127) & (mask_np[..., 2] < 127)).astype(np.uint8) * 255
    return red_mask

def run_lama_inpaint(image_path, mask_path, out_path):
    try:
        # Import torch and model loading only when needed
        import torch
        from omegaconf import OmegaConf
        import yaml
        lama_src_path = os.path.join(os.path.dirname(__file__), 'LaMa_src')
        if lama_src_path not in sys.path:
            sys.path.insert(0, lama_src_path)
        from PIL import Image
        from saicinpainting.training.trainers import load_checkpoint
        from saicinpainting.evaluation.utils import move_to_device

        # Update model_dir and perceptual_root to new location (fix missing parenthesis and indentation)
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
        
        # Debug: Print the paths being checked
        print(f"Checking paths:")
        print(f"  Config: {config_path} - Exists: {os.path.exists(config_path)}")
        print(f"  Checkpoint: {checkpoint_path} - Exists: {os.path.exists(checkpoint_path)}")
        print(f"  Input image: {image_path} - Exists: {os.path.exists(image_path)}")
        print(f"  Mask image: {mask_path} - Exists: {os.path.exists(mask_path)}")
        
        # List temp directory contents to help debug
        temp_dir = os.path.dirname(mask_path)
        if os.path.exists(temp_dir):
            print(f"Contents of {temp_dir}:")
            for item in os.listdir(temp_dir):
                print(f"  - {item}")
        else:
            print(f"Temp directory does not exist: {temp_dir}")
        
        # Check if required files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask image not found: {mask_path}")

        with open(config_path, 'r') as f:
            config = OmegaConf.create(yaml.safe_load(f))
        # Override hardcoded perceptual loss weights path with a plugin-relative path
        try:
            if 'losses' in config and 'resnet_pl' in config.losses:
                if os.path.isdir(perceptual_root):
                    config.losses.resnet_pl.weights_path = perceptual_root
                    print(f"Set perceptual loss weights_path to: {perceptual_root}")
                else:
                    print(f"Perceptual weights directory not found at {perceptual_root}. Disabling resnet_pl loss.")
                    # Disable perceptual loss to avoid FileNotFoundError
                    if 'weight' in config.losses.resnet_pl:
                        config.losses.resnet_pl.weight = 0
        except Exception as _e:
            print(f"Warning adjusting perceptual loss path: {_e}")
        model = load_checkpoint(config, checkpoint_path, map_location='cpu', strict=False)
        model.eval()

        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        mask_np = load_mask(mask_path)
        image_np = pad_to_mod8(image_np)
        mask_np = pad_to_mod8(mask_np)

        batch = {
            'image': torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.,
            'mask': torch.from_numpy(mask_np[None]).float().unsqueeze(0) / 255.
        }
        batch = move_to_device(batch, 'cpu')
        with torch.no_grad():
            result = model(batch)
            result = result['inpainted'].cpu().numpy()[0]
            result = (result.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        Image.fromarray(result).save(out_path)
        print(f"Successfully saved inpainted image to: {out_path}")
        return True
        
    except Exception as e:
        print(f"Error during inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_full_pipeline(texture_path, png_export_path, mask_path, output_path):
    """
    Complete pipeline that can be called from Unreal Engine
    """
    try:
        # For now, just call the inpainting function
        # You can extend this to include other steps
        return run_lama_inpaint(png_export_path, mask_path, output_path)
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LaMa Inpainting')
    parser.add_argument('--image', type=str, required=True, help='Absolute path to the input image')
    parser.add_argument('--mask', type=str, required=True, help='Absolute path to the mask image')
    parser.add_argument('--output', type=str, required=True, help='Absolute path to save the output image')
    args = parser.parse_args()

    run_lama_inpaint(args.image, args.mask, args.output)

if __name__ == '__main__':
    main()
