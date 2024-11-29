import torch
import torch.nn.functional as F
import numpy as np
import os
from plyfile import PlyElement, PlyData

from src.utils.metric_utils import evaluate_ate, calc_psnr
from src.utils.gaussian_utils import build_rotation, transform_to_frame
from src.datasets.geometryutils import relative_transformation
from src.Render import transformed_params2rendervar
from gaussian_semantic_rasterization import GaussianRasterizer


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res

def construct_list_of_attributes(f_dc, scale, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def convert_npz_to_ply(src, dest):
    params = np.load(src)

    xyz = params['means3D']
    normals = np.zeros_like(xyz)
    f_dc = params['rgb_colors']
    opacities = params['logit_opacities']
    scale = params['log_scales'].repeat(3, axis=-1)
    rotation = params['unnorm_rotations']

    dtype_full = [
        (attribute, 'f4') for attribute in construct_list_of_attributes(f_dc, scale, rotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(dest)


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)
    convert_npz_to_ply(save_path, os.path.join(output_dir, "gsplat.ply"))


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)
    convert_npz_to_ply(save_path, os.path.join(output_dir, "gsplat"+str(time_idx)+".ply"))


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)
    convert_npz_to_ply(save_path, os.path.join(output_dir, "gsplat.ply"))


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)
    convert_npz_to_ply(save_path, os.path.join(output_dir, "gsplat"+str(time_idx)+".ply"))

def report_progress(params, data, i, progress_bar, iter_time_idx, every_i=1, qual_every_i=1, 
                    tracking=False, mapping=False, wandb_run=None, wandb_step=None, wandb_save_qual=False, online_time_idx=None,
                    global_logging=True):
    if i % every_i == 0 or i == 1:
        if wandb_run is not None:
            if tracking:
                stage = "Tracking"
            elif mapping:
                stage = "Mapping"
            else:
                stage = "Current Frame Optimization"
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).cuda().float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)
            if wandb_run is not None:
                tracking_log = {f"{stage}/Latest Pose Error":iter_pt_error, 
                               f"{stage}/Latest Relative Pose Error":rel_pt_error,
                               f"{stage}/ATE RMSE":ate_rmse}

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_pts)
        im, rendered_objects, radius, rendered_depth, rendered_alpha = GaussianRasterizer(raster_settings=data["cam"])(**rendervar)
    
        # depth_sil_rendervar = transformed_params2depthplussilhouette(params, data['w2c'], 
                                                                    #  transformed_pts)
        # depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
        # rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (data['depth'] > 0)
        # silhouette = depth_sil[1, :, :]
        # presence_sil_mask = (silhouette > sil_thres)

        # if tracking:
        #     psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        # else:

        psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt(((rendered_depth - data['depth']) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rendered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rendered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rendered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)