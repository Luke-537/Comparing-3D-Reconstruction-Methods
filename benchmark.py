import trimesh
import torch
from pytorch3d.loss import chamfer_distance

def load_mesh(filepath):
    """Helper function to safely load meshes from different file formats."""
    loaded = trimesh.load(filepath)
    
    if isinstance(loaded, trimesh.Scene):
        # Merge all geometry into a single mesh
        mesh = trimesh.util.concatenate(
            [g for g in loaded.geometry.values()]
        )
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unknown loaded type: {type(loaded)}")

    return mesh

# Load meshes
ground_truth_mesh = load_mesh("outputs/hunyuan/minecraft_pig.glb")
huyang3d_mesh = load_mesh("outputs/hunyuan/minecraft_pig.glb")

### Chamfer Distance
# Sample point clouds from surface (e.g., 10k points)
points1 = ground_truth_mesh.sample(10000)
points2 = huyang3d_mesh.sample(10000)

# Convert to PyTorch tensors
pc1 = torch.tensor(points1, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
pc2 = torch.tensor(points2, dtype=torch.float32).unsqueeze(0)  # [1, M, 3]

# Compute Chamfer Distance
cd, _ = chamfer_distance(pc1, pc2)
print(f"Chamfer Distance: {cd.item()}")


### IoU
# Voxelize both meshes (adjust pitch/resolution)
vox_pred = huyang3d_mesh.voxelized(pitch=0.01).matrix
vox_gt = ground_truth_mesh.voxelized(pitch=0.01).matrix

# Compute IoU
intersection = (vox_pred & vox_gt).sum()
union = (vox_pred | vox_gt).sum()
iou = intersection / union
print(f"IoU: {iou:.4f}")

