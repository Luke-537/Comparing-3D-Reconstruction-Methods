import trimesh
import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance

def align_pc(source_points, target_points):
    """Align source point cloud to target point cloud using ICP."""
    # Convert to Open3D point clouds
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_points)
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source_points)

    #R = pcd_source.get_rotation_matrix_from_xyz((0, np.pi * 1.5, np.pi))
    #pcd_source.rotate(R, center=(0,0,0))
    #save_point_cloud(np.asarray(pcd_source.points), "tests/test.ply")

    # Run ICP
    reg_result = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, 
        max_correspondence_distance=0.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # Apply transformation to the source points
    transformation = reg_result.transformation
    aligned_points = (np.asarray(source_points) @ transformation[:3, :3].T) + transformation[:3, 3]
    
    return aligned_points

def save_point_cloud(points, filename):
    """Save point cloud to .ply file"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def load_glb_as_mesh(filepath):
    """Load a GLB file and extract the first mesh found"""
    scene = trimesh.load(filepath)
    
    # If the GLB contains a scene with multiple meshes, combine them
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()
    elif isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        raise ValueError("GLB file contains unsupported data")
    
    return mesh

def pointcloud_to_voxel(points, pitch=0.01):
    """Convert point cloud to voxel grid"""
    # Shift points to positive coordinates (voxelization requires positive values)
    points = points - np.min(points, axis=0)
    
    # Voxelize
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
        voxel_size=pitch
    )
    
    # Convert to binary 3D array
    voxels = voxel_grid.get_voxels()
    grid_shape = np.max([vox.grid_index for vox in voxels], axis=0) + 1
    voxel_matrix = np.zeros(grid_shape, dtype=bool)
    
    for vox in voxels:
        voxel_matrix[tuple(vox.grid_index)] = True
    
    return voxel_matrix

def normalize(points):
    return (points - np.mean(points, axis=0)) / np.max(np.ptp(points, axis=0))

# Load and preprocess meshes
ground_truth_mesh = load_glb_as_mesh("references/elephant_gt.glb").process(validate=True)
output_mesh = trimesh.load("outputs/triposr/2/mesh.obj").process(validate=True)

# Sample point clouds from aligned meshes
points_gt = ground_truth_mesh.sample(10000)
points_output = output_mesh.sample(10000)

#points_gt = normalize(points_gt)
#points_output = normalize(points_output)

# Align meshes (align output_mesh to ground_truth_mesh)
points_aligned = align_pc(points_output.copy(), points_gt)

# Export point clouds for comparison
save_point_cloud(points_gt, "tests/gt.ply")
save_point_cloud(points_output, "tests/output.ply")
save_point_cloud(points_aligned, "tests/aligned.ply")

# Chamfer
# Convert to PyTorch tensors
pc_gt = torch.tensor(points_gt, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
pc_aligned = torch.tensor(points_aligned, dtype=torch.float32).unsqueeze(0)  # [1, M, 3]

# Compute Chamfer Distance
cd, _ = chamfer_distance(pc_gt, pc_aligned)
print(f"Chamfer Distance after alignment: {cd.item():.6f}")


# IoU
# Voxelize point clouds (aligned and ground truth)
voxel_size = 0.01  # Adjust based on your point cloud density
vox_gt = pointcloud_to_voxel(points_gt, pitch=voxel_size)
vox_aligned = pointcloud_to_voxel(points_aligned, pitch=voxel_size)

# Pad voxel grids to same dimensions (if they differ)
max_shape = np.maximum(vox_gt.shape, vox_aligned.shape)
vox_gt_padded = np.zeros(max_shape, dtype=bool)
vox_aligned_padded = np.zeros(max_shape, dtype=bool)

vox_gt_padded[:vox_gt.shape[0], :vox_gt.shape[1], :vox_gt.shape[2]] = vox_gt
vox_aligned_padded[:vox_aligned.shape[0], :vox_aligned.shape[1], :vox_aligned.shape[2]] = vox_aligned

# Compute IoU
intersection = np.sum(vox_gt_padded & vox_aligned_padded)
union = np.sum(vox_gt_padded | vox_aligned_padded)
iou = intersection / union if union > 0 else 0
print(f"IoU (Point Cloud): {iou:.4f}")

print("GT scale:", np.ptp(points_gt, axis=0))  # Peak-to-peak range per axis
print("Output scale:", np.ptp(points_aligned, axis=0))