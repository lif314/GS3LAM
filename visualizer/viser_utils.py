import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from copy import deepcopy
import imageio
import json

from src.utils.gaussian_utils import build_rotation
from src.Render import get_rasterizationSettings
from src.Evaluater import feature_to_rgb, visualize_obj
from gaussian_semantic_rasterization import GaussianRasterizer, GaussianRasterizationSettings


def load_camera_recon(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]
    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k

def load_camera_mesh(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    return w2c, k, org_width, org_height


def load_scene_data(scene_path, first_frame_w2c, intrinsics):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    transformed_pts = params['means3D']

    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'sh_objs': params["obj_dc"],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }

    return rendervar, all_w2cs

def load_scene_data_online(scene_path):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    params = all_params

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())
    
    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    return params, all_w2cs

def get_rendervars_sem(params, w2c, curr_timestep):
    params_timesteps = params['timestep']
    selected_params_idx = params_timesteps <= curr_timestep
    keys = [k for k in params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]
    selected_params = deepcopy(params)
    for k in keys:
        selected_params[k] = selected_params[k][selected_params_idx]
    transformed_pts = selected_params['means3D']
    w2c = torch.tensor(w2c).cuda().float()

    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': selected_params['rgb_colors'],
        'sh_objs': selected_params["obj_dc"],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
    }

    return rendervar


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render_sam(w2c, k, timestep_data, cfg):
    with torch.no_grad():
        cam = get_rasterizationSettings(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
        white_bg_cam = GaussianRasterizationSettings(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered,
            debug=cam.debug
        )

        rendered_image, rendered_objects, radii, rendered_depth, rendered_alpha = GaussianRasterizer(raster_settings=white_bg_cam)(**timestep_data)
        
        return rendered_image, rendered_depth, rendered_objects

def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    
    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def rgbd2pcd_sem(logits, depth, w2c, intrinsics, cfg):
    width, height = logits.shape[2], logits.shape[1]
    # print("logits: ", logits.shape) # torch.Size([256, 340, 600])
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
   
    logits = torch.argmax(logits, dim=0)
    # print("logits: ", logits.shape) # torch.Size([340, 600])
    pred_obj_mask = visualize_obj(logits.cpu().numpy().astype(np.uint8)) / 255.
    pred_obj_mask = pred_obj_mask.reshape(-1, 3)
    # print("pred_obj_mask: ", pred_obj_mask.shape) # (340, 600, 3)
    # cols = torch.permute(pred_obj, (1, 2, 0)).reshape(-1, 3)
    # cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy().astype(np.uint8))
    cols = o3d.utility.Vector3dVector(pred_obj_mask)
    return pts, cols

def rgbd2pcd_sem_feature(sem_feature, depth, w2c, intrinsics, cfg):
    width, height = sem_feature.shape[2], sem_feature.shape[1]
    # print("logits: ", logits.shape) # torch.Size([256, 340, 600])
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    feature_rgb = feature_to_rgb(sem_feature) / 255.
    # print("feature_rgb: ", feature_rgb.shape) # (340, 600, 3)
    cols = feature_rgb.reshape(-1, 3)
    cols = o3d.utility.Vector3dVector(cols)
    return pts, cols


def blend_images(im, mask):
    alpha = (mask == 0).astype(np.float32) * 0.5 + 0.5
    blend = (im.astype(np.float32) * alpha+ mask.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return blend

def rgbd2pcd_sem_color(color, logits, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    # print("logits: ", logits.shape) # torch.Size([256, 340, 600])
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
    im = cols.contiguous().double().cpu().numpy()

    logits = torch.argmax(logits, dim=0)
    # print("logits: ", logits.shape) # torch.Size([340, 600])
    logits[logits == 0] = 100
    pred_obj_mask = visualize_obj(logits.cpu().numpy().astype(np.uint8))
    pred_obj_mask = pred_obj_mask.reshape(-1, 3)

    # blend
    obj_blend = blend_images(im * 255., pred_obj_mask)
    # print("pred_obj_mask: ", pred_obj_mask.shape) # (340, 600, 3)
    # cols = torch.permute(pred_obj, (1, 2, 0)).reshape(-1, 3)
    # cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy().astype(np.uint8))
    cols = o3d.utility.Vector3dVector(obj_blend / 255.)
    return pts, cols

def save_screenshot(img_uint8, camera_json):
    screenshot_path = './screenshots'
    os.makedirs(screenshot_path, exist_ok=True)
    im_name = "im.png"
    json_name = 'camera.json'
    count = 1
    while os.path.exists(os.path.join(screenshot_path, im_name)):
        im_name = f"im_{count}.png"
        count += 1
    while os.path.exists(os.path.join(screenshot_path, json_name)):
        json_name = f"camera_{count}.json"
        count += 1
    
    imageio.imwrite(os.path.join(screenshot_path, im_name), img_uint8)
    with open(os.path.join(screenshot_path, json_name), 'w') as f:
                json.dump(camera_json, f)
    print(f"Save....{os.path.join(screenshot_path, im_name)}")


def save_image(img_uint8):
    filename = "screenshot.png"
    count = 1
    while os.path.exists(filename):
        filename = f"screenshot_{count}.png"
        count += 1
    
    imageio.imwrite(filename, img_uint8)
    print(f"Saved screenshot as {filename}")

def create_mesh_from_point_cloud(pcd):
    # 估计法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 使用Poisson重建生成网格
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # 裁剪掉低密度的区域（如果需要）
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh