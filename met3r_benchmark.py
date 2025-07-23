import sys
from pathlib import Path
import csv
met3r_base = Path("external") / "met3r" / "met3r"
MET3R_ROOT = Path(__file__).resolve().parent / met3r_base
sys.path.insert(0, str(MET3R_ROOT))
import torch
from met3r import MEt3R
from PIL import Image
import torchvision.transforms as transforms

IMG_SIZE = 128

def met3r_score(img1_path, img2_path):
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # outputs [0,1]
        transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1, 1]
    ])
    # Load and transform all images
    images = [transform(Image.open(path).convert('RGB')) for path in [img1_path, img2_path]]

    # Stack images to shape (10, 3, IMG_SIZE, IMG_SIZE)
    images_tensor = torch.stack(images)

    inputs = images_tensor.view(1, 2, 3, IMG_SIZE, IMG_SIZE).cuda()

    torch.cuda.empty_cache()

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
    # inputs = torch.randn((5, 2, 3, IMG_SIZE, IMG_SIZE)).cuda()
    # inputs = inputs.clip(-1, 1)

    # Evaluate MEt3R
    score, *_ = metric(
        images=inputs, 
        return_overlap_mask=False, # Default 
        return_score_map=False, # Default 
        return_projections=False # Default 
    )

    # Should be between 0.25 - 0.35
    # Clear up GPU memory
    torch.cuda.empty_cache()
    return (score.mean().item())


# Dummy scoring function – replace with real MET3R call
# def met3r_score(img1_path, img2_path):
#     # For real usage, import your MET3R scoring pipeline here.
#     # This dummy version just returns a fake float.
#     import random
#     return random.uniform(0.8, 1.0)

def run_met3r_evaluation(input_dir="views_for_met3r", output_csv="met3r_scores.csv"):
    results = []

    # Go through each model subdir
    for model_dir in sorted(Path(input_dir).rglob("*")):
        if not model_dir.is_dir():
            continue

        # Find PNG files (assumes views are named view_00.png, etc.)
        images = sorted(model_dir.glob("*.png"))
        if len(images) < 2:
            continue  # Skip incomplete sets

        scores = []
        for i in range(len(images)):
            img1 = images[i]
            img2 = images[(i + 1) % len(images)]  # Circular connection
            score = met3r_score(str(img1), str(img2))
            scores.append(score)

        model_name = model_dir.relative_to(input_dir).as_posix()
        mean_score = sum(scores) / len(scores)
        print(f"✅ {model_name}: MET3R mean score = {mean_score:.4f}")

        results.append({
            "model": model_name,
            "mean_score": mean_score,
            "individual_scores": scores
        })

    # Save to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["model", "mean_score"] + [f"score_{i}" for i in range(max(len(r['individual_scores']) for r in results))]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "model": r["model"],
                "mean_score": f"{r['mean_score']:.4f}",
            }
            row.update({f"score_{i}": f"{score:.4f}" for i, score in enumerate(r["individual_scores"])})
            writer.writerow(row)

    print(f"\n📄 CSV saved: {output_csv}")

# Example usage
run_met3r_evaluation("views_for_met3r", "met3r_consistency_scores.csv")





