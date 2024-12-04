import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
import trimesh

from scipy.ndimage import median_filter

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from src.Evaluater import visualize_obj
from src.utils.common_utils import seed_everything
from src.Decoder import SemanticDecoder
from visualizer.viser_utils import (
    load_camera_mesh,
    load_scene_data_online,
    render_sam,
    get_rendervars_sem,
    blend_images
)

def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()

def filter_depth_outliers(depth_map, kernel_size=3, threshold=1.0):
    median_filtered = median_filter(depth_map, size=kernel_size)
    abs_diff = np.abs(depth_map - median_filtered)
    outlier_mask = abs_diff > threshold
    depth_map_filtered = np.where(outlier_mask, median_filtered, depth_map)
    return depth_map_filtered

def clean_mesh(mesh):
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(
        mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 100
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {old_idx: vertex_count +
                         new_idx for new_idx, old_idx in enumerate(component)}
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[np.any(
            np.isin(mesh_tri.faces, component), axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    # cleaned_mesh_tri.remove_degenerate_faces()
    # cleaned_mesh_tri.remove_duplicate_faces()
    cleaned_mesh_tri.update_faces(cleaned_mesh_tri.unique_faces())
    print(
        f'Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}')

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces)
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64))

    return cleaned_mesh

def export_tsdf_mesh(cfg):
    scene_path = os.path.join(cfg["logdir"], "params.npz")
    classifier_path = os.path.join(cfg["logdir"], "classifier.pth")
    
    # Load semantic decoder
    semantic_decoder = SemanticDecoder(16, 256)
    semantic_decoder.load_state_dict(torch.load(classifier_path))
            
    # Load Scene Data
    first_frame_w2c, k, org_width, org_height = load_camera_mesh(cfg, scene_path)

    # print("first w2c: ", first_frame_w2c)

    cfg['viz_w'] = org_width
    cfg['viz_h'] = org_height

    fx, fy, cx, cy = k[0,0],k[1,1], k[0,2], k[1,2]

    params, all_w2cs = load_scene_data_online(scene_path)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        org_width, org_height, fx, fy, cx, cy)
    
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # print("all_w2cs[0]:", all_w2cs[0])

    for i in tqdm(range(0, len(all_w2cs), 5)):
        cur_w2c =  all_w2cs[i]

        scene_data = get_rendervars_sem(params, cur_w2c, curr_timestep=i)
        im, depth, sem_feature = render_sam(cur_w2c, k, scene_data, cfg)

        if cfg['render_mode'] == 'sem':
            logits = semantic_decoder(sem_feature)
            logits = torch.argmax(logits, dim=0)
            # print("logits: ", logits.shape) # torch.Size([340, 600])
            mask_unit8 = logits.cpu().numpy().astype(np.uint8)
            mask_unit8[mask_unit8 == 0] = 255
            pred_obj_mask = visualize_obj(mask_unit8)
            # pred_obj_mask = pred_obj_mask.reshape(-1, 3)

            # blend
            im_np = np.ascontiguousarray((torch2np(im.permute(1, 2, 0)) * 255).astype(np.uint8))
            obj_blend = blend_images(im_np, pred_obj_mask)

            im_np = np.ascontiguousarray(obj_blend)
        else:
            im_np = np.ascontiguousarray((torch2np(im.permute(1, 2, 0)) * 255).astype(np.uint8))

        depth_np = np.ascontiguousarray(depth.squeeze(0).cpu().numpy().astype(np.float32))
        
        depth_np = filter_depth_outliers(
            depth_np, kernel_size=20, threshold=0.1)

        color_image = o3d.geometry.Image(im_np)
        depth_image = o3d.geometry.Image(depth_np)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, depth_scale=1.0, depth_trunc=30.0, convert_rgb_to_intensity=False)

        volume.integrate(rgbd_image, intrinsic, cur_w2c)

    o3d_mesh = volume.extract_triangle_mesh()
    compensate_vector = (-0.0 * scale / 512.0, 2.5 * scale / 512.0, -2.5 * scale / 512.0)
    o3d_mesh = o3d_mesh.translate(compensate_vector)
    os.makedirs(os.path.join(cfg["logdir"], "mesh"), exist_ok=True)
    if cfg['render_mode'] == 'sem':
        file_name = os.path.join(cfg["logdir"], "mesh", "final_mesh_sem.ply")
    else:
        file_name = os.path.join(cfg["logdir"], "mesh", "final_mesh.ply")
    
    # clean mesh
    print("Clean mesh....")
    print(o3d_mesh)
    cleaned_mesh = clean_mesh(o3d_mesh)

    if cfg['render_mode'] == 'sem':
        cleaned_mesh_path = os.path.join(args.logdir, "mesh", "cleaned_mesh_sem.ply")
    else:
        cleaned_mesh_path = os.path.join(args.logdir, "mesh", "cleaned_mesh.ply")
    o3d.io.write_triangle_mesh(str(cleaned_mesh_path), cleaned_mesh)

    o3d.io.write_triangle_mesh(file_name, o3d_mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, help="Path to experiment file")
    parser.add_argument("--mode", type=str, default='color', help="['color, 'sem']")
    args = parser.parse_args()

    print(args.logdir)
    config_path = os.path.join(args.logdir, 'config.py')
    config = SourceFileLoader(
        os.path.basename(config_path), config_path
    ).load_module()

    seed_everything(seed=config.seed)

    viz_cfg = config.config['viz']
    viz_cfg['logdir'] = args.logdir
    viz_cfg["render_mode"] = args.mode
 
    # Visualize Final Reconstruction
    export_tsdf_mesh(viz_cfg)