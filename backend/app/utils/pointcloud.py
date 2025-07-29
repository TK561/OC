import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from typing import Tuple, Optional, Union
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

def generate_pointcloud(
    image: Image.Image,
    depth_map: Union[np.ndarray, Image.Image],
    density: float = 1.0,
    max_points: int = 100000
) -> o3d.geometry.PointCloud:
    """
    Generate 3D point cloud from RGB image and depth map
    
    Args:
        image: RGB image
        depth_map: Depth map (numpy array or PIL Image)
        density: Point density factor (0.1 - 2.0)
        max_points: Maximum number of points
        
    Returns:
        Open3D PointCloud object
    """
    try:
        # Convert inputs to numpy arrays
        if isinstance(image, Image.Image):
            rgb_array = np.array(image)
        else:
            rgb_array = image
            
        if isinstance(depth_map, Image.Image):
            depth_array = np.array(depth_map.convert('L'))
        else:
            depth_array = depth_map
            
        # Ensure depth is 2D
        if len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0]
            
        # Resize if dimensions don't match
        if rgb_array.shape[:2] != depth_array.shape:
            height, width = depth_array.shape
            rgb_pil = Image.fromarray(rgb_array)
            rgb_pil = rgb_pil.resize((width, height), Image.Resampling.LANCZOS)
            rgb_array = np.array(rgb_pil)
        
        height, width = depth_array.shape
        
        # Create coordinate grids
        fx = fy = max(width, height)  # Simple focal length estimation
        cx, cy = width / 2, height / 2  # Principal point at center
        
        # Apply density factor for downsampling
        step = max(1, int(1.0 / density))
        
        # Create meshgrid with step
        u_indices, v_indices = np.meshgrid(
            np.arange(0, width, step),
            np.arange(0, height, step),
            indexing='xy'
        )
        
        # Get corresponding depth and color values
        depth_values = depth_array[v_indices, u_indices]
        rgb_values = rgb_array[v_indices, u_indices]
        
        # Normalize depth values (assuming depth is in 0-255 range)
        depth_normalized = depth_values.astype(np.float32) / 255.0
        
        # Convert to actual depth (scale factor for visualization)
        depth_scale = 3.0  # より自然な3D表示のためのスケール調整
        depth_actual = depth_normalized * depth_scale
        
        # Convert to 3D coordinates - 正しい画像座標系変換
        z = depth_actual + 0.5  # オフセットを追加して全体を前に移動
        x = (u_indices - cx) * z / fx * 0.8  # X方向をやや圧縮
        y = -(v_indices - cy) * z / fy * 0.8  # Y軸反転（画像上部が3D上部になるように）
        
        # Filter out invalid points
        valid_mask = (z > 0) & (z < depth_scale * 0.95)
        
        # Flatten and filter
        points_3d = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
        colors = rgb_values[valid_mask].reshape(-1, 3) / 255.0
        
        # Limit number of points if necessary
        if len(points_3d) > max_points:
            indices = np.random.choice(len(points_3d), max_points, replace=False)
            points_3d = points_3d[indices]
            colors = colors[indices]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        logger.info(f"Generated point cloud with {len(pcd.points)} points")
        return pcd
        
    except Exception as e:
        logger.error(f"Point cloud generation failed: {e}")
        raise

def save_pointcloud(
    pointcloud: o3d.geometry.PointCloud,
    output_path: str,
    format: str = "ply"
) -> bool:
    """
    Save point cloud to file
    
    Args:
        pointcloud: Open3D PointCloud object
        output_path: Output file path
        format: Output format ('ply', 'obj', 'pcd')
        
    Returns:
        Success status
    """
    try:
        format = format.lower()
        
        if format == "ply":
            success = o3d.io.write_point_cloud(output_path, pointcloud)
        elif format == "obj":
            # Convert to mesh first for OBJ export
            mesh = create_mesh_from_pointcloud(pointcloud)
            success = o3d.io.write_triangle_mesh(output_path, mesh)
        elif format == "pcd":
            success = o3d.io.write_point_cloud(output_path, pointcloud)
        else:
            # Default to PLY
            success = o3d.io.write_point_cloud(output_path, pointcloud)
        
        if success:
            logger.info(f"Point cloud saved to {output_path}")
        else:
            logger.error(f"Failed to save point cloud to {output_path}")
            
        return success
        
    except Exception as e:
        logger.error(f"Point cloud saving failed: {e}")
        return False

def create_mesh_from_pointcloud(
    pointcloud: o3d.geometry.PointCloud,
    method: str = "poisson"
) -> o3d.geometry.TriangleMesh:
    """
    Create triangle mesh from point cloud
    
    Args:
        pointcloud: Input point cloud
        method: Reconstruction method ('poisson', 'ball_pivoting')
        
    Returns:
        Triangle mesh
    """
    try:
        # Estimate normals
        pointcloud.estimate_normals()
        pointcloud.orient_normals_consistent_tangent_plane(100)
        
        if method == "poisson":
            # Poisson surface reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pointcloud, depth=9
            )
        elif method == "ball_pivoting":
            # Ball pivoting reconstruction
            distances = pointcloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2 * avg_dist
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pointcloud,
                o3d.utility.DoubleVector([radius, radius * 2])
            )
        else:
            # Default to Poisson
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pointcloud, depth=9
            )
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        logger.info(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        return mesh
        
    except Exception as e:
        logger.error(f"Mesh creation failed: {e}")
        # Return empty mesh on failure
        return o3d.geometry.TriangleMesh()

def generate_heightmap(
    depth_array: np.ndarray,
    scale: float = 1.0
) -> o3d.geometry.TriangleMesh:
    """
    Generate heightmap mesh from depth array
    
    Args:
        depth_array: 2D depth array
        scale: Height scale factor
        
    Returns:
        Triangle mesh representing heightmap
    """
    try:
        height, width = depth_array.shape
        
        # Normalize depth
        depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        depth_scaled = depth_normalized * scale
        
        # Create coordinate grids
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)
        
        # Create vertices
        vertices = np.column_stack([
            X.flatten(),
            Y.flatten(), 
            depth_scaled.flatten()
        ])
        
        # Create triangles
        triangles = []
        for i in range(height - 1):
            for j in range(width - 1):
                # Current quad vertices
                v1 = i * width + j
                v2 = i * width + j + 1
                v3 = (i + 1) * width + j
                v4 = (i + 1) * width + j + 1
                
                # Two triangles per quad
                triangles.append([v1, v2, v3])
                triangles.append([v2, v4, v3])
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        logger.info(f"Generated heightmap with {len(vertices)} vertices")
        return mesh
        
    except Exception as e:
        logger.error(f"Heightmap generation failed: {e}")
        return o3d.geometry.TriangleMesh()

def apply_point_cloud_filters(
    pointcloud: o3d.geometry.PointCloud,
    remove_outliers: bool = True,
    voxel_size: Optional[float] = None
) -> o3d.geometry.PointCloud:
    """
    Apply filters to clean up point cloud
    
    Args:
        pointcloud: Input point cloud
        remove_outliers: Whether to remove statistical outliers
        voxel_size: Voxel size for downsampling (None to skip)
        
    Returns:
        Filtered point cloud
    """
    try:
        filtered_pcd = pointcloud
        
        # Voxel downsampling
        if voxel_size:
            filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size)
            logger.info(f"Downsampled to {len(filtered_pcd.points)} points")
        
        # Remove outliers
        if remove_outliers:
            filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            logger.info(f"Removed outliers, {len(filtered_pcd.points)} points remaining")
        
        return filtered_pcd
        
    except Exception as e:
        logger.error(f"Point cloud filtering failed: {e}")
        return pointcloud

def compute_point_cloud_metrics(pointcloud: o3d.geometry.PointCloud) -> dict:
    """
    Compute metrics for point cloud quality assessment
    
    Args:
        pointcloud: Input point cloud
        
    Returns:
        Dictionary with metrics
    """
    try:
        points = np.asarray(pointcloud.points)
        
        metrics = {
            "num_points": len(points),
            "bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
                "size": (points.max(axis=0) - points.min(axis=0)).tolist()
            }
        }
        
        if len(points) > 0:
            # Compute distances to nearest neighbors
            distances = pointcloud.compute_nearest_neighbor_distance()
            metrics["density"] = {
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
                "min_distance": float(np.min(distances)),
                "max_distance": float(np.max(distances))
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return {"num_points": 0}