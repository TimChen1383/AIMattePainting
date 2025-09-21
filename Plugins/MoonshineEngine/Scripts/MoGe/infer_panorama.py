# Dynamically add library paths for embedded Python
import os
import sys

# Enable OpenEXR support
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Add library paths to sys.path for embedded Python environment
script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(os.path.dirname(script_dir))
lib_paths = [
    os.path.join(plugin_dir, 'Libraries', 'DepthMap_Panorama'),
    script_dir,  # Add the script directory itself to find moge module
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

from pathlib import Path
# Updated sys.path manipulation for new location
if (_package_root := str(Path(__file__).absolute())) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import json
import warnings

import click

         
def run_panorama_inference(input_path, output_path='./panorama_output', pretrained_model_path=None, device_name='cuda', 
                          resize_to=None, resolution_level=9, threshold=0.03, batch_size=4, 
                          save_splitted=False, save_maps=True, save_glb=False, save_ply=False, show=False):
    """
    Complete panorama inference pipeline that can be called from Unreal Engine
    """
    try:
        # Set default model path if not provided
        if pretrained_model_path is None:
            plugin_dir = os.path.dirname(os.path.dirname(script_dir))
            pretrained_model_path = os.path.join(plugin_dir, 'Models', 'MoGe', 'model.pt')
        
        # Simulate command line arguments
        class Args:
            def __init__(self):
                self.input_path = input_path
                self.output_path = output_path
                self.pretrained_model_name_or_path = pretrained_model_path
                self.device_name = device_name
                self.resize_to = resize_to
                self.resolution_level = resolution_level
                self.threshold = threshold
                self.batch_size = batch_size
                self.save_splitted = save_splitted
                self.save_maps_ = save_maps
                self.save_glb_ = save_glb
                self.save_ply_ = save_ply
                self.show = show
        
        args = Args()
        
        # Run the main inference logic
        return main_inference(args)
        
    except Exception as e:
        print(f"Panorama inference error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main_inference(args):
    """
    Main inference logic extracted from the click command
    """
    try:
        # Lazy import
        import cv2
        import numpy as np
        from numpy import ndarray
        import torch
        from PIL import Image
        from tqdm import tqdm, trange
        import trimesh
        import trimesh.visual
        from scipy.sparse import csr_array, hstack, vstack
        from scipy.ndimage import convolve
        from scipy.sparse.linalg import lsmr

        import utils3d
        from moge.model.v1 import MoGeModel
        from moge.utils.io import save_glb, save_ply
        from moge.utils.vis import colorize_depth
        from moge.utils.panorama import spherical_uv_to_directions, get_panorama_cameras, split_panorama_image, merge_panorama_depth

        device = torch.device(args.device_name)

        include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
        if Path(args.input_path).is_dir():
            image_paths = sorted(itertools.chain(*(Path(args.input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
        else:
            image_paths = [Path(args.input_path)]
        
        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {args.input_path}')

        # Write outputs
        if not any([args.save_maps_, args.save_glb_, args.save_ply_]):
            warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", or "--ply" to specify the output.')
            args.save_maps_ = args.save_glb_ = args.save_ply_ = True

        print(f"Loading model from: {args.pretrained_model_name_or_path}")
        model = MoGeModel.from_pretrained(args.pretrained_model_name_or_path).to(device).eval()

        for image_path in (pbar := tqdm(image_paths, desc='Total images', disable=len(image_paths) <= 1)):
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            if args.resize_to is not None:
                height, width = min(args.resize_to, int(args.resize_to * height / width)), min(args.resize_to, int(args.resize_to * width / height))
                image = cv2.resize(image, (width, height), cv2.INTER_AREA)
            
            splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
            splitted_resolution = 512
            splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution)

            # Infer each view 
            print('Inferring...') if pbar.disable else pbar.set_postfix_str(f'Inferring')

            splitted_distance_maps, splitted_masks = [], []
            for i in trange(0, len(splitted_images), args.batch_size, desc='Inferring splitted views', disable=len(splitted_images) <= args.batch_size, leave=False):
                image_tensor = torch.tensor(np.stack(splitted_images[i:i + args.batch_size]) / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                fov_x, fov_y = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(splitted_intriniscs[i:i + args.batch_size])))
                fov_x = torch.tensor(fov_x, dtype=torch.float32, device=device)
                output = model.infer(image_tensor, fov_x=fov_x, apply_mask=False)
                distance_map, mask = output['points'].norm(dim=-1).cpu().numpy(), output['mask'].cpu().numpy()
                splitted_distance_maps.extend(list(distance_map))
                splitted_masks.extend(list(mask))

            # Save splitted
            if args.save_splitted:
                splitted_save_path = Path(args.output_path, image_path.stem, 'splitted')
                splitted_save_path.mkdir(exist_ok=True, parents=True)
                for i in range(len(splitted_images)):
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}.jpg'), cv2.cvtColor(splitted_images[i], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(splitted_save_path / f'{i:02d}_distance_vis.png'), cv2.cvtColor(colorize_depth(splitted_distance_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))

            # Merge
            print('Merging...') if pbar.disable else pbar.set_postfix_str(f'Merging')

            merging_width, merging_height = min(1920, width), min(960, height)
            panorama_depth, panorama_mask = merge_panorama_depth(merging_width, merging_height, splitted_distance_maps, splitted_masks, splitted_extrinsics, splitted_intriniscs)
            panorama_depth = panorama_depth.astype(np.float32)
            panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
            panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0
            points = panorama_depth[:, :, None] * spherical_uv_to_directions(utils3d.numpy.image_uv(width=width, height=height))
            
            # Write outputs
            print('Writing outputs...') if pbar.disable else pbar.set_postfix_str(f'Writing outputs')
            # Save directly to output directory without creating subfolders
            save_path = Path(args.output_path)
            save_path.mkdir(exist_ok=True, parents=True)
            if args.save_maps_:
                # Only save depth as 16-bit PNG (normalized)
                masked_depth = panorama_depth[panorama_mask] if np.any(panorama_mask) else panorama_depth.flatten()
                depth_min = np.percentile(masked_depth, 1)
                depth_max = np.percentile(masked_depth, 99)
                if depth_max > depth_min:
                    # Invert depth: closer objects are white, farther are black
                    norm_depth = 1.0 - (panorama_depth - depth_min) / (depth_max - depth_min)
                    norm_depth = np.clip(norm_depth, 0, 1)
                else:
                    norm_depth = np.zeros_like(panorama_depth)
                depth_png = (norm_depth * 65535).astype(np.uint16)
                # Save with the same name as input but with _depth suffix
                depth_filename = f"{image_path.stem}_depth.png"
                cv2.imwrite(str(save_path / depth_filename), depth_png)

            # Export mesh & visulization
            if args.save_glb_ or args.save_ply_ or args.show:
                normals, normals_mask = utils3d.numpy.points_to_normals(points, panorama_mask)
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=panorama_mask & ~(utils3d.numpy.depth_edge(panorama_depth, rtol=args.threshold) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                    tri=True
                )

            if args.save_glb_:
                glb_filename = f"{image_path.stem}_mesh.glb"
                save_glb(save_path / glb_filename, vertices, faces, vertex_uvs, image)

            if args.save_ply_:
                ply_filename = f"{image_path.stem}_mesh.ply"
                save_ply(save_path / ply_filename, vertices, faces, vertex_colors)

            if args.show:
                trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    faces=faces, 
                    process=False
                ).show()
        
        print(f"Panorama inference completed! Output saved to: {args.output_path}")
        return True
        
    except Exception as e:
        print(f"Main inference error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@click.command(help='Inference script for panorama images')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--output', '-o', 'output_path', type=click.Path(), default='./output', help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='../../Models/MoGe', help='Pretrained model name or path. Defaults to local Model folder')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level of inference. The higher, the better but slower. Defaults to 9. Note that it is irrelevant to the output resolution.')
@click.option('--threshold', type=float, default=0.03, help='Threshold for removing edges. Defaults to 0.03. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--batch_size', type=int, default=4, help='Batch size for inference. Defaults to 4.')
@click.option('--splitted', 'save_splitted', is_flag=True, help='Whether to save the splitted images. Defaults to False.')
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps and fov(image, depth, mask, points, fov).')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a.glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a.ply file. The color will be saved as vertex colors.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')
def main(
    input_path: str,
    output_path: str,
    pretrained_model_name_or_path: str,
    device_name: str,
    resize_to: int,
    resolution_level: int,
    threshold: float,
    batch_size: int,
    save_splitted: bool,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):  
    # Create args object and call main_inference
    class Args:
        def __init__(self):
            self.input_path = input_path
            self.output_path = output_path
            self.pretrained_model_name_or_path = pretrained_model_name_or_path
            self.device_name = device_name
            self.resize_to = resize_to
            self.resolution_level = resolution_level
            self.threshold = threshold
            self.batch_size = batch_size
            self.save_splitted = save_splitted
            self.save_maps_ = save_maps_
            self.save_glb_ = save_glb_
            self.save_ply_ = save_ply_
            self.show = show
    
    args = Args()
    return main_inference(args)  


if __name__ == '__main__':
    main()