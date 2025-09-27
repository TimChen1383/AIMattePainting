# AI Matte Painting

![Dune](https://github.com/user-attachments/assets/4f8ac793-139b-42cf-b26f-76f56a0f4c81)
![360](https://github.com/user-attachments/assets/0f6af19c-1b9d-48ab-95b2-5dfdea5d9f8b)



## Overview

This tool provides a comprehensive AI toolset to help creators build 3D scenes in no time, from image generation to 3D model. The toolset consists of five main modules: Image Generation, Image Cleanup, Depth Calculation, Image Slicing, and Image to 3D Conversion. Each module can operate independently, but can also be used in conjunction with each other.

## Installation
Currently the plugin support Unreal Engine 5.3. Before using it, make sure the required models, packages, and CUDA are successfully installed. This ensures the plugin work properly.

### 1. Models and Libraries
- Libraries: Place the Libraries folder under the plugin (Plugins\MoonshineEngine\Libraries)
- AI Models: Place the Models folder under the plugin (Plugins\MoonshineEngine\Models)

### 2. CUDA
The MoGe model (360° depth calculation) within the tool requires CUDA to run. To match the Torch version installed within the tool, CUDA version 12.9 is required. For a comparison of Torch and CUDA versions, refer to "https://pytorch.org/get-started/locally/." You can verify your computer's CUDA version by entering the following command in CMD:
```python
nvidia-smi
```

## Tool Introduction
### 1. Image Generation
<img width="1024" height="416" alt="generator_BeforeAfter" src="https://github.com/user-attachments/assets/25c3cd59-54e3-4db1-b9bb-2ac4a4504852" />

<img width="1024" height="416" alt="generator_UI" src="https://github.com/user-attachments/assets/ba552ecb-e08c-4b90-81ca-5423ecb0dd5d" />



### Image Cleanup
![ObjectRemoval](https://github.com/user-attachments/assets/769854b5-fa0a-4eb0-8f2f-6c4dcfb6a48a)

### Depth Calculation
![DepthGenerate](https://github.com/user-attachments/assets/22c7bf49-7aa0-46ce-9177-43172e3d7e61)

### Image Slicing
![DepthBasedSlicing](https://github.com/user-attachments/assets/5073b0a4-1958-4df6-81d8-ae49721a6436)

### Image to 3D


(not done yet)
*make sure to add lines to slice the sections
*make sure to add images for better visualization
- abstract
- installation
- Features, how to use the functions
- result Gallery
- debug
