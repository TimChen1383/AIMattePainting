import os
import sys

# Add library paths to sys.path for embedded Python environment
script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(os.path.dirname(script_dir))
lib_paths = [
    os.path.join(plugin_dir, 'Libraries', 'JuntoAI')
]

for lib_path in lib_paths:
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

import requests
import json
from datetime import datetime
import random
import time
import argparse

def run_junto_ai_generation(prompt=None, output_dir=None):
    # Default values
    default_prompt = "create a 360 hdr, <lora:alimama-creativeFLUX.1-Turbo-Alpha:1.0> <lora:HDR360:1.0>"
    default_output_dir = r"D:\UnrealProject\AIMattePainting_53\Temp"
    lora_suffix = ", create a 360 hdr, <lora:alimama-creativeFLUX.1-Turbo-Alpha:1.0> <lora:HDR360:1.0>"

    # Use provided values or defaults
    if prompt:
        user_prompt = prompt.strip()
        final_prompt = user_prompt + lora_suffix
    else:
        final_prompt = default_prompt
    final_output_dir = output_dir if output_dir else default_output_dir
    
    payload = {  
        "prompt": final_prompt,
        
        "params": {
            "pipeline": "panorama",
            "cfg_scale": 1.0,
            "sampler_name": "k_euler",
            "denoise": 1.0,
            "denoising_strength": 0.05,
            "ddim_steps": 8.0,
            "scheduler": "simple",
            "width": 1536,
            "height": 768,
            "guidance": 1.6,
            "max_shift": 1.15,
            "base_shift": 0.5,
            "vae_model": "flux_ae",
            "unet_model": "flux1-schnell",
            "dualclip": {
                "clip_name1": "clip_l",
                "clip_name2": "t5xxl_fp16",
                "dualclip_type": "flux",
            },
        },
        "nsfw": True,
        "censor_nsfw": True,
        "image_is_control": True,
        "models": ["flux1-schnell"],
        "workers": [
            #"457e73cc-c75e-4b43-923e-3dcc815bff57"
            # "f506c899-6b1c-4d94-b25e-5348d750874a"
             "081aa226-356b-46c2-ae7f-607f80b679f4"
        ],    
        "application": "junto-storage",
        "shared": False,
        "r2": True,
        "jobId": "",
        "index": 0,
        "gathered": False,
        "failed": False
    }

    # Render machine
    api_key = 'utIsdwT-hAHIijSooPGayg'
    horde_url = "http://192.168.8.66:8081/"

    headers = {
        'Content-Type': 'application/json',
        'apikey': api_key,
    }
    
    print(f"=== JuntoAI Generation Started ===")
    print(f"Prompt: {final_prompt}")
    print(f"Output directory: {final_output_dir}")
    print(f"Time: {datetime.now()}")
    
    try:
        # Post request
        r = requests.post(f'{horde_url}/api/v2/generate/async', headers=headers, data=json.dumps(payload))
        r_dict = json.loads(r.text)
        print('==============================================================')
        print(r_dict)
        
        if "id" not in r_dict:
            print("ERROR: No ID returned from server")
            return False
            
        id = r_dict["id"]
        timer = 1000  # waiting time

        for i in range(timer):
            check = requests.get(f'{horde_url}/api/v2/generate/check/{id}', headers=headers)

            try:
                check_dict = json.loads(check.text)
            except json.JSONDecodeError as e:
                print("無法解析 JSON:", e)
                print("伺服器返回內容:", check.text)
                continue

            if "finished" in check_dict:
                finished = check_dict["finished"]
                if finished == 1:
                    result = requests.get(f'{horde_url}/api/v2/generate/status/{id}', headers=headers)
                    result_dict = json.loads(result.text)
                    
                    if "generations" in result_dict and len(result_dict["generations"]) > 0:
                        url = result_dict["generations"][0]["img"]
                        print(f"SUCCESS: Image generated: {url}")
                        
                        # Download the image to the output directory
                        if not os.path.exists(final_output_dir):
                            os.makedirs(final_output_dir)

                        # Sanitize user prompt for filename (remove special characters, spaces, and limit length)
                        def sanitize_filename(s):
                            import re
                            s = re.sub(r'[^a-zA-Z0-9_-]', '_', s)
                            return s[:32]  # limit length for safety

                        if prompt:
                            sanitized_prompt = sanitize_filename(prompt.strip())
                        else:
                            sanitized_prompt = sanitize_filename(default_prompt)
                        filename = f"{sanitized_prompt}_JuntoAI.png"
                        output_path = os.path.join(final_output_dir, filename)
                        
                        # Clear proxy environment variables to avoid HTTPS proxy issues in Unreal Engine
                        os.environ.pop('HTTP_PROXY', None)
                        os.environ.pop('http_proxy', None)
                        os.environ.pop('HTTPS_PROXY', None)
                        os.environ.pop('https_proxy', None)

                        # Download and save the image automatically
                        download_successful = False
                        try:
                            # Proxy settings inside Moonshine
                            # proxies = {
                            #     "http": "http://192.168.8.3:3128",
                            #     "https": "http://192.168.8.3:3128"
                            # }
                            # img_response = requests.get(url, stream=True, proxies=proxies)

                            # Proxy settings outside Moonshine
                            img_response = requests.get(url, stream=True)

                            if img_response.status_code == 200:
                                print(f"Current working directory: {os.getcwd()}")
                                print(f"Saving image to: {output_path}")
                                with open(output_path, "wb") as f:
                                    for chunk in img_response.iter_content(1024):
                                        f.write(chunk)
                                print(f"SUCCESS: Image saved to: {output_path}")
                                download_successful = True
                            else:
                                print(f"Failed to download image. Status code: {img_response.status_code}")
                                download_successful = False
                        except Exception as e:
                            print(f"Error downloading image: {e}")
                            download_successful = False
                        
                        return download_successful
                    else:
                        print("ERROR: No generations found in result")
                        return False
                    break
            else:
                print("伺服器返回的內容缺少 'finished' 鍵:", check_dict)

            time.sleep(1)
            if i % 10 == 0:  # Print progress every 10 seconds
                print(f"Waiting... {i+1}/{timer} seconds")

        print("Timeout reached")
        return False
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JuntoAI 360 HDR Generation')
    parser.add_argument('--prompt', type=str, help='Prompt for image generation')
    parser.add_argument('--output', type=str, help='Output directory for generated image')
    
    args = parser.parse_args()
    
    success = run_junto_ai_generation(args.prompt, args.output)
    sys.exit(0 if success else 1)
