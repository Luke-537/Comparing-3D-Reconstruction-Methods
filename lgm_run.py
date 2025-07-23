
import os
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
from pathlib import Path
import sys

lgm_base = Path("external") / "LGM"
LGM_ROOT = Path(__file__).resolve().parent / lgm_base
sys.path.insert(0, str(LGM_ROOT))
sys.path.insert(0, str(LGM_ROOT / "src"))

from external.LGM.core.options import config_defaults, Options
from external.LGM.core.models import LGM
from external.LGM.mvdream.pipeline_mvdream import MVDreamPipeline
from external.LGM.convert import Converter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

output_base = Path("outputs") / "lgm"
input_base = Path("images")

opt = config_defaults["big"]
opt.resume = str(lgm_base / "pretrained" / "model_fp16.safetensors")
opt.workspace = str(output_base)
opt.test_path = str(input_base)

# model
model = LGM(opt)

# resume pretrained checkpoint
ckpt = load_file(opt.resume, device='cpu')
model.load_state_dict(ckpt, strict=False)
print(f'[INFO] Loaded checkpoint from {opt.resume}')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load image dream
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe = pipe.to(device)

# load rembg
bg_remover = rembg.new_session()

# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = kiui.read_image(path, mode='uint8')

    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # generate mv
    image = image.astype(np.float32) / 255.0

    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

    mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        mesh_path = os.path.join(opt.workspace, name + '.ply')
        model.gs.save_ply(gaussians, mesh_path)
        print(f"Mesh saved to {mesh_path}")
       
# Extract all input files from input directory
input_files = [
        os.path.join(input_base, file) 
        for file in os.listdir(input_base) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')
    ]
print(f'Total number of input images: {len(input_files)}')

# Create point cloud for each input file
for idx, file in enumerate(input_files):
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Input file {file} does not exist.")
    else:
        print(f'[{idx+1}/{len(input_files)}] Creating point cloud for {file} ...')
        process(opt, file)

# Extract all output point clouds from output directory
output_point_clouds = [
        os.path.join(output_base, file) 
        for file in os.listdir(output_base) 
        if file.endswith('.ply')
    ]
print(f'Total number of point clouds saved: {len(output_point_clouds)}')

# Create mesh for each point cloud
for idx, file in enumerate(output_point_clouds):
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Input file {file} does not exist.")
    else:
        print(f'[{idx+1}/{len(output_point_clouds)}] Creating mesh for {file} ...')
        mesh_opt = config_defaults["big"]
        mesh_opt.resume = str(lgm_base / "pretrained" / "model_fp16.safetensors")
        mesh_opt.workspace = str(output_base)
        mesh_opt.test_path = str(file)
        converter = Converter(opt=mesh_opt).cuda()
        converter.fit_nerf()
        converter.fit_mesh()
        converter.fit_mesh_uv()
        converter.export_mesh(mesh_opt.test_path.replace('.ply', '.glb'))