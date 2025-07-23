import numpy as np
import open3d as o3d
import os
from pathlib import Path

def render_views(mesh_path, output_dir="rendered_views", num_views=2, width=128, height=128):
    """Render different views from input mesh to use them for met3r later."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Check for vertex colors or texture
    has_vertex_colors = mesh.has_vertex_colors()
    has_textures = mesh.has_textures()

    # Create renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background

    # Create material
    mat = o3d.visualization.rendering.MaterialRecord()
    if has_vertex_colors:
        print("Using vertex colors.")
        mat.shader = "defaultUnlit"  # "defaultLit" doesn’t show vertex colors
    elif has_textures:
        print("Using texture map.")
        mat.shader = "defaultLit"
    else:
        print("No vertex colors or texture. Using default grey.")
        mat.shader = "defaultLit"
        mat.base_color = [0.6, 0.6, 0.6, 1.0]  # fallback gray

    renderer.scene.add_geometry("mesh", mesh, mat)

    # Set up camera
    center = mesh.get_center()
    radius = max(mesh.get_max_bound() - mesh.get_min_bound())
    cam_distance = radius * 2.0
    
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views  # azimuth (horizontal angle)
        num_zigzags = 1.5
        elevation_angle = 10 * np.sin(num_zigzags * theta)  # degrees
        phi = np.radians(90 - elevation_angle)  # inclination angle for spherical coords

        # Convert spherical to Cartesian coordinates
        x = cam_distance * np.sin(phi) * np.cos(theta)
        y = cam_distance * np.cos(phi)
        z = cam_distance * np.sin(phi) * np.sin(theta)

        cam_pos = center + np.array([x, y, z])

        renderer.scene.camera.look_at(center, cam_pos, [0, 1, 0])

        img = renderer.render_to_image()
        output_path = os.path.join(output_dir, f"view_{i:02d}.png")
        o3d.io.write_image(output_path, img)
        print(f"Saved view: {output_path}")
    renderer.scene.clear_geometry()
    

def is_mesh_file(file):
    return file.suffix.lower() in ['.obj', '.ply', '.glb', '.gltf']

def crawl_and_render(base_input_dir, base_output_dir, num_views=8):
    for dirpath, dirnames, filenames in os.walk(base_input_dir):
        for filename in filenames:
            mesh_file = Path(filename)
            if not is_mesh_file(mesh_file):
                continue

            full_input_path = Path(dirpath) / filename
            model_name = Path(dirpath).relative_to(base_input_dir)
            mesh_name = mesh_file.stem  # without extension

            # Construct output path: views_for_met3r/model_name/mesh_name/
            output_path = Path(base_output_dir) / model_name / mesh_name
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"Rendering {full_input_path} to {output_path}")
            try:
                render_views(str(full_input_path), output_dir=str(output_path), num_views=num_views)
            except Exception as e:
                print(f"Failed to render {full_input_path}: {e}")


if __name__ == "__main__":
    crawl_and_render("outputs", "views_for_met3r", num_views=8)