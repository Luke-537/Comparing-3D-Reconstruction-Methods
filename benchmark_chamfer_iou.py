import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import pandas as pd
from pytorch3d.loss import chamfer_distance
import torch
from tqdm import tqdm
import trimesh

def save_point_cloud(points, filename):
    """Save a point cloud to .ply file."""
    o3d.io.write_point_cloud(filename, points)
    
def save_point_cloud_image_matplotlib(pc1, pc2, save_path="comparison.png"):
    """Save two point clouds to an image in order to see their alignment."""
    pc1_np = np.asarray(pc1.points)
    pc2_np = np.asarray(pc2.points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1_np[:, 0], pc1_np[:, 1], pc1_np[:, 2], c='red', s=1, label='GT')
    ax.scatter(pc2_np[:, 0], pc2_np[:, 1], pc2_np[:, 2], c='blue', s=1, label='Pred')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("GT vs Prediction (Aligned Point Clouds)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_and_prepare_mesh(filepath):
    """Load mesh or scene, merge if necessary, normalize scale and center."""
    loaded = trimesh.load(filepath)

    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in loaded.geometry.values()])
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.points.PointCloud):
        mesh = trimesh.Trimesh(vertices=loaded.vertices, process=False)
    else:
        raise ValueError(f"Unknown loaded type: {type(loaded)}")

    # Normalize: center + scale to unit bounding box
    mesh.vertices -= mesh.centroid
    scale = mesh.bounding_box.extents.max()
    mesh.vertices /= scale
    return mesh

def sample_point_cloud(mesh, num_points=10000):
    """Sample points from the mesh surface."""
    return mesh.sample(num_points)

def normalize_scale(pcd):
    """Center and scale point cloud to unit cube."""
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    scale = np.linalg.norm(points - center, axis=1).max()
    points = (points - center) / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def preprocess_point_cloud_from_numpy(points: np.ndarray):
    """Convert numpy array to Open3D PointCloud and preprocess."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.remove_non_finite_points()
    pcd.estimate_normals()
    return normalize_scale(pcd)

def compute_chamfer(pc1, pc2):
    """Compute the chamfer distance of two point clouds"""
    pts1 = np.asarray(pc1.points)
    pts2 = np.asarray(pc2.points)
    t1 = torch.tensor(pts1, dtype=torch.float32).unsqueeze(0)
    t2 = torch.tensor(pts2, dtype=torch.float32).unsqueeze(0)
    dist, _ = chamfer_distance(t1, t2)
    return dist.item()

def compute_voxel_iou_pointclouds(pcd1, pcd2, pitch=0.05):
    """Voxelize two point clouds into a shared grid and compute the intersection over union (IoU)."""
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    all_points = np.vstack([points1, points2])
    min_bound = all_points.min(axis=0)
    max_bound = all_points.max(axis=0)
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int)

    def voxelize(points):
        indices = ((points - min_bound) / pitch).astype(int)
        voxel_grid = np.zeros(grid_shape, dtype=bool)
        valid = (indices >= 0).all(axis=1) & (indices < grid_shape).all(axis=1)
        indices = indices[valid]
        voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return voxel_grid

    vox1 = voxelize(points1)
    vox2 = voxelize(points2)

    intersection = np.logical_and(vox1, vox2).sum()
    union = np.logical_or(vox1, vox2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def get_fpfh(pcd, voxel_size):
    """Computes downsampled point cloud and its Fast Point Feature Histograms (FPFH) features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down, o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

def global_registration(pred, gt, pred_fpfh, gt_fpfh, voxel_size):
    """Performs global registration between two point clouds using RANSAC with feature matching."""
    distance_threshold = voxel_size * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pred, gt, pred_fpfh, gt_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

def refine_icp(pred, gt, init_trans, voxel_size):
    """Refines an initial registration transformation using the Iterative Closest Point (ICP) algorithm."""
    return o3d.pipelines.registration.registration_icp(
        pred, gt, voxel_size * 0.4, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

def evaluate_models_on_ground_truth(
    gt_dir,
    model_dirs,
    output_csv,
    voxel_size=0.05,
    pitch=0.05
):
    """Evaluate the each 3D model based on the ground truth by computing the chamfer distance and the IoU"""
    results = []
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.glb'))]
    for gt_file in tqdm(gt_files, desc="Evaluating samples"):
        gt_path = os.path.join(gt_dir, gt_file)
        try:
            mesh_gt = load_and_prepare_mesh(gt_path)
        except Exception as e:
            print(f"Failed to load GT {gt_file}: {e}")
            continue
        
        # Get pointcloud for ground truth
        pts_gt = sample_point_cloud(mesh_gt, num_points=10000)
        pcd_gt = preprocess_point_cloud_from_numpy(pts_gt)
        gt_down, gt_fpfh = get_fpfh(pcd_gt, voxel_size)
        
        # Extract base name (e.g., "chair" from "chair_gt.glb")
        base_name = os.path.splitext(gt_file)[0].replace("_gt", "")
        
        # Extract prediction for all the models of the currently selected ground truth
        for model_name, model_dir in model_dirs.items():
            found_path = None
            for ext in [".ply", ".obj", ".glb"]:
                candidate = os.path.join(model_dir, base_name + ext)
                if os.path.exists(candidate):
                    found_path = candidate
                    break
            if not found_path:
                print(f"{model_name} missing file for {base_name} (searched .ply, .obj, .glb)")
                continue
            try:
                mesh_pred = load_and_prepare_mesh(found_path)
            except Exception as e:
                print(f"Failed to load prediction for {model_name}/{found_path}: {e}")
                continue
            
            # Get pointcloud for prediction
            pts_pred = sample_point_cloud(mesh_pred, num_points=10000)
            pcd_pred = preprocess_point_cloud_from_numpy(pts_pred)
            pred_down, pred_fpfh = get_fpfh(pcd_pred, voxel_size)
            
            # Coarse alignment
            result_ransac = global_registration(pred_down, gt_down, pred_fpfh, gt_fpfh, voxel_size)
            
            # Refined Alignment
            result_icp = refine_icp(pcd_pred, pcd_gt, result_ransac.transformation, voxel_size)
            pcd_pred.transform(result_icp.transformation)
            
            # Compute Metrics
            chamfer = compute_chamfer(pcd_gt, pcd_pred)
            iou = compute_voxel_iou_pointclouds(pcd_gt, pcd_pred, pitch)
            
            results.append({
                "sample": base_name,
                "model": model_name,
                "ChamferDistance": f"{chamfer:.5f}",
                "IoU": f"{iou:.5f}"
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    evaluate_models_on_ground_truth(
        gt_dir="references",
        model_dirs={
            "hunyuan": "outputs/hunyuan",
            "instantmesh": "outputs/instantmesh",
            "lgm": "outputs/lgm",
            "triposr": "outputs/triposr",
            "zeroshape": "outputs/zeroshape"
        },
        output_csv="mesh_metrics.csv",
        voxel_size=0.05,
        pitch=0.05
    )