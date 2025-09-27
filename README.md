# AI Matte Painting

![Dune](https://github.com/user-attachments/assets/4f8ac793-139b-42cf-b26f-76f56a0f4c81)
![360](https://github.com/user-attachments/assets/0f6af19c-1b9d-48ab-95b2-5dfdea5d9f8b)



## Overview

This tool provides a comprehensive AI toolset to help creators build 3D scenes in no time, from image generation to 3D model. The toolset consists of five main modules: Image Generation, Image Cleanup, Depth Calculation, Image Slicing, and Image to 3D Conversion. Each module can operate independently, but can also be used in conjunction with each other.

## Installation
Currently the plugin support Unreal Engine 5.3. Before using it, make sure the required models, packages, and CUDA are successfully installed. This ensures the plugin work properly.

#### 1. Models and Libraries
- Libraries: Place the Libraries folder under the plugin (Plugins\MoonshineEngine\Libraries)
- AI Models: Place the Models folder under the plugin (Plugins\MoonshineEngine\Models)

#### 2. CUDA
The MoGe model (360° depth calculation) within the tool requires CUDA to run. To match the Torch version installed within the tool, CUDA version 12.9 is required. For a comparison of Torch and CUDA versions, refer to "https://pytorch.org/get-started/locally/." You can verify your computer's CUDA version by entering the following command in CMD:
```python
nvidia-smi
```

## Tool Introduction
#### 1. Image Generation
<img width="1024" height="416" alt="generator_BeforeAfter" src="https://github.com/user-attachments/assets/25c3cd59-54e3-4db1-b9bb-2ac4a4504852" />

<img width="1024" height="416" alt="generator_UI" src="https://github.com/user-attachments/assets/ba552ecb-e08c-4b90-81ca-5423ecb0dd5d" />

Users can generate images by entering prompt. The image generator supports general images or panoramic images generation, depending on user needs. Image generation is performed using the JuntoAI service, eliminating the need to use local resources.
- Prompt: Enter prompt words (examples: snow, park)
- Image preview: Preview the resulting image
- Image Type: Select between general images and panoramic images
- AI Generate: Renders the image based on the prompt words entered


#### 2. Image Cleanup
![ObjectRemoval](https://github.com/user-attachments/assets/769854b5-fa0a-4eb0-8f2f-6c4dcfb6a48a)

If an image contains small objects, the quality of the subsequent depth map generation will be affected. The tool provides an Inpaint model to assist users with image cleanup. Through an interactive UI, users can use a brush to paint the area they wish to remove. The AI ​​model will then remove the objects based on the area they have drawn.
- Eraser Input Image: Enter the image you wish to erase.
- Image Preview: Preview the image and draw a mask to define the area to be erased.
- Slider: Drag to control the size of the mask brush.
- Clear Mask: Clears the mask you have drawn.
- AI Clean Up: Starts image cleanup.


#### 3. Depth Calculation
![DepthGenerate](https://github.com/user-attachments/assets/22c7bf49-7aa0-46ce-9177-43172e3d7e61)

In order to create 3D effect, the tool can generate a depth map based on the input image, facilitating subsequent 3D model creation. The tool offers two depth models: 2D (normal) and 360 (surround view), user can choose the model based on the needs.
- Depth Model Type: Select whether to calculate a 2D or 360 depth map.
- Clear Preview: The tool will automatically preview the calculated depth map, which can be cleared.
- AI Depth Map: Calculate the image using the selected depth model.


#### 4. Image Slicing
![DepthBasedSlicing](https://github.com/user-attachments/assets/5073b0a4-1958-4df6-81d8-ae49721a6436)
![foreground](https://github.com/user-attachments/assets/fa696037-12dd-47d5-824d-0acd715cd83d)

A parallax effect can be created by appropriately slicing an image into foreground, middle, and background. The tool slices based on depth values. Users can preview the depth values ​​and positions before slicing. Generally, slicing at the intersection of objects works best, but users can adjust the slicing positions to suit their needs. After slicing, the background may be partially obscured by the foreground, and the tool will automatically fill in the missing areas with AI model. To create a parallax effect, it is recommended to select an image with distinct foreground, middle, and background that are not contiguous (the foreground, middle, and background should not be connected).

- Slice Input Image: Input the image to slice.
- Slice Input Depth: Input the image's depth map.
- Slice Preview: Instantly preview the depth values ​​relative to the image's position.
- Slice Value - Add Value: Record the values ​​to slice.
- Slice Value - Remove Value: Remove the most recently added slicing value.
- Slice: Slices the image and fills in the missing areas due to foreground occlusion.


#### 5. Image to 3D
<img width="1024" height="416" alt="displacement_BeforeAfter" src="https://github.com/user-attachments/assets/d7e7d29f-da80-49ff-aa47-d6fccb03bc43" />

<img width="1024" height="416" alt="3D_UI" src="https://github.com/user-attachments/assets/acf6dab9-43ef-490a-a8d6-3e9236d22553" />

The tool generates a 3D model using the ambient and depth maps (BP_DepthMeshGenerator). The subdivision count and displacement size of the object's surface can be adjusted in real time, as well as the model's position. Displacement can be based on a circle, semicircle, or plane. To allow for dynamic adjustments, the tool uses a Dynamic Mesh Component. However, for use cases where the Dynamic Mesh Component isn't feasible (such as nDisplay), the tool provides a function to convert a Dynamic Mesh to a Static Mesh (Bake Static Mesh).
- Bake Static Mesh: After confirming all desired parameters, click the button to convert the Dynamic Mesh to a Static Mesh.
- Frozen: Stops calculating the geometry script to avoid constantly calculating displacement as the model moves, which can consume computer resources.
- Output Mesh Name: The name of the converted Static Mesh file.
- Scale: Adjusts the object's size. To ensure the object's size is preserved when converted to a Static Mesh, adjust the object's size using this parameter.
- Rotation: Adjust the object's rotation. To ensure the object's rotation is preserved when converted to a Static Mesh, adjust the object's rotation using this parameter.
- Source Mesh: This tool provides models such as circles, semicircles, and planes as the basis for displacement.
- Displacement Map: Input the depth map you created earlier.
- Material Instance: Input a Material Instance. You can use the M_Diffuse plugin as a base to create a Material Instance and input the previously cut image in the Material Instance. When baking a Static Mesh, this Material Instance will be copied and assigned to the baked Static Mesh (this Material Instance will not be modified).
- Subdivision: Adjust the level of surface subdivision.
- Magnitude: Determines the strength of the depth map's displacement of the model.
- Smooth Strength: Adjust the smoothness of the model's surface.

![Template](https://github.com/user-attachments/assets/0db739b2-b8bc-4250-acbf-10cfa3d614a4)
There are some templates (sub-levels) prepared in the plugin. You can drag the template into the scene to use it. You can adjust it according to your needs.

- SL_1Plane: A one-layer planar 3D depth model
- SL_2Plane: A two-layer planar 3D depth model
- SL_3Plane: A three-layer planar 3D depth model
- SL_4Plane: A four-layer planar 3D depth model
- SL_Hemisphere: A hemisphere-based 3D depth model
- SL_Sphere: A sphere-based 3D depth model

## Examples
There are also some examples (Levels) attached to the plugin.
![Example](https://github.com/user-attachments/assets/57ff08b8-f0a3-4519-89e8-98c3173fbafe)

