import torch
from met3r import MEt3R

IMG_SIZE = 128

# Initialize MEt3R
metric = MEt3R(
    img_size=IMG_SIZE, # Default to 256, set to `None` to use the input resolution on the fly!
    use_norm=True, # Default to True 
    backbone="mast3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
    feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
    feature_backbone_weights="mhamilton723/FeatUp", # Default
    upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
    distance="cosine", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
    freeze=True, # Default to True
).cuda()

# Prepare inputs of shape (batch, views, channels, height, width): views must be 2
# RGB range must be in [-1, 1]
# Reduce the batch size in case of CUDA OOM
inputs = torch.randn((10, 2, 3, IMG_SIZE, IMG_SIZE)).cuda()
inputs = inputs.clip(-1, 1)

# Evaluate MEt3R
score, *_ = metric(
    images=inputs, 
    return_overlap_mask=False, # Default 
    return_score_map=False, # Default 
    return_projections=False # Default 
)

# Should be between 0.25 - 0.35
print(score.mean().item())

# Clear up GPU memory
torch.cuda.empty_cache()