import os
import sys
import random
import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from src.utils.sh_utils import RGB2SH
from src.utils.gaussian_utils import transform_to_frame, build_rotation
from src.Render import get_rasterizationSettings, transformed_params2rendervar
from gaussian_semantic_rasterization import GaussianRasterizer

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective", random_select=False):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    if random_select:
        num_pts = point_cld.shape[0]
        random_mask = random.sample(range(num_pts), int(0.005 * num_pts))
        # random_mask = random.sample(range(num_pts), 1024)
        point_cld = point_cld[random_mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[random_mask]

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, num_objects=16):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    
    fused_objects = RGB2SH(torch.rand((num_pts, num_objects), device="cuda"))
    fused_objects = fused_objects[:,:,None]
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
        "obj_dc": fused_objects.transpose(1, 2)
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    # Every Frame pose initialization
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, num_objects=16):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose, gt_objects = dataset[0]


    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = get_rasterizationSettings(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, num_objects)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio

    return params, variables, intrinsics, w2c, cam
    
def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution, num_objects=16):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    
    # random init obj_id
    fused_objects = RGB2SH(torch.rand((num_pts, num_objects), device="cuda"))
    fused_objects = fused_objects[:,:,None]
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales':  log_scales,
        "obj_dc": fused_objects.transpose(1, 2)
    }
   
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians_alpha(params, variables, curr_data, densify_thres, time_idx, mean_sq_dist_method, gaussian_distribution, num_objects=16):
    # Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    rendered_image, rendered_objects, radii, render_depth, rendered_alpha = GaussianRasterizer(raster_settings=curr_data["cam"])(**rendervar)

    # alpha mask
    non_presence_alpha_mask = (rendered_alpha < densify_thres)
    
    show_add_mask = False
    if show_add_mask:
        from PIL import Image
        os.makedirs("./logs/plots", exist_ok=True)
        masked_image_data = rendered_image * (~non_presence_alpha_mask)
        masked_image = Image.fromarray((masked_image_data.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))
        masked_image.save(f'./logs/plots/add_alpha_{time_idx}.png')

    # depth error mask
    gt_depth = curr_data['depth'][0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50 * depth_error.median())
    
    non_presence_mask = non_presence_alpha_mask | non_presence_depth_mask

    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
    curr_w2c = torch.eye(4).cuda().float()
    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
    curr_w2c[:3, 3] = curr_cam_tran

    # Get the new pointcloud in the world frame
    new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                mean_sq_dist_method=mean_sq_dist_method, random_select=False)
    new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution, num_objects)
    
    for k, v in new_params.items():
        params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))

    num_pts = params['means3D'].shape[0]
    variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
    variables['denom'] = torch.zeros(num_pts, device="cuda").float()
    variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
    new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
    variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables
