import csv
from PIL import Image
from pathlib import Path
import sys
import torch
import torchvision.transforms as transforms
met3r_base = Path("external") / "met3r" / "met3r"
MET3R_ROOT = Path(__file__).resolve().parent / met3r_base
sys.path.insert(0, str(MET3R_ROOT))
from met3r import MEt3R

IMG_SIZE = 512

def met3r_score(img1_path, img2_path, metric):
    """Calculate the met3r score (between 0 and 1) for two images to measure their consistency."""
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # outputs [0,1]
        transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1, 1]
    ])
    # Load and transform all images
    images = [transform(Image.open(path).convert('RGB')) for path in [img1_path, img2_path]]

    images_tensor = torch.stack(images)
    inputs = images_tensor.view(2, 2, 3, IMG_SIZE, IMG_SIZE).cuda()

    torch.cuda.empty_cache()

    # Evaluate MEt3R
    score, *_ = metric(
        images=inputs, 
        return_overlap_mask=False,
        return_score_map=False,
        return_projections=False
    )
    
    # Clear up GPU memory
    torch.cuda.empty_cache()
    return (score.mean().item())

def run_met3r_evaluation(input_dir="views_for_met3r", output_csv="met3r_scores.csv"):
    """Evaluate met3r scores and save the output scores to a .csv file."""
    # Initialize MEt3R
    metric = MEt3R(
        img_size=IMG_SIZE,
        use_norm=True, 
        backbone="mast3r",
        feature_backbone="dino16",
        feature_backbone_weights="mhamilton723/FeatUp", # Default
        upsampler="featup",
        distance="cosine",
        freeze=True,
    ).cuda()
    
    results = []

    for model_dir in sorted(Path(input_dir).rglob("*")):
        if not model_dir.is_dir():
            continue

        images = sorted(model_dir.glob("*.png"))
        if len(images) < 2:
            continue  # Skip incomplete sets

        scores = []
        for i in range(len(images)):
            img1 = images[i]
            img2 = images[(i + 1) % len(images)]  # Circular connection
            score = met3r_score(str(img1), str(img2), metric)
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


if __name__ == "__main__":
    run_met3r_evaluation("views_for_met3r", "met3r_consistency_scores.csv")