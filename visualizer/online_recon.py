import argparse
import os
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import open3d as o3d
import torch
import imageio

from src.utils.common_utils import seed_everything
from src.Decoder import SemanticDecoder
from visualizer.viser_utils import (
    load_camera_recon,
    load_scene_data_online,
    render_sam,
    rgbd2pcd,
    rgbd2pcd_sem,
    rgbd2pcd_sem_color,
    rgbd2pcd_sem_feature,
    make_lineset,
    get_rendervars_sem
)

def online_recon(cfg):
    scene_path = os.path.join(cfg["logdir"], "params.npz")
    classifier_path = os.path.join(cfg["logdir"], "classifier.pth")
    video_path = os.path.join(cfg["logdir"], cfg["render_mode"] + str(".mp4"))
    
    # load semantic decoder
    semantic_decoder = SemanticDecoder(16, 256)
    semantic_decoder.load_state_dict(torch.load(classifier_path))
            
    # Load Scene Data
    first_frame_w2c, k = load_camera_recon(cfg, scene_path)

    params, all_w2cs = load_scene_data_online(scene_path)

    print(params['means3D'].shape)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']), 
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)


    scene_data = get_rendervars_sem(params, first_frame_w2c, curr_timestep=0)
    # rendered_image, rendered_objects, rendered_depth
    im, depth, sem_feature = render_sam(first_frame_w2c, k, scene_data, cfg)

    if cfg['render_mode'] == 'color' or  cfg['render_mode'] == 'depth':
        init_pts, init_cols = rgbd2pcd(im, depth, first_frame_w2c, k, cfg)

    if cfg['render_mode'] == 'sem':
        logits = semantic_decoder(sem_feature)
        init_pts, init_cols = rgbd2pcd_sem(logits, depth, first_frame_w2c, k, cfg)

    if cfg['render_mode'] == 'sem_color':
        logits = semantic_decoder(sem_feature)
        init_pts, init_cols = rgbd2pcd_sem_color(im, logits, depth, first_frame_w2c, k, cfg)

    if cfg['render_mode'] == 'sem_feature':
        init_pts, init_cols = rgbd2pcd_sem_feature(sem_feature, depth, first_frame_w2c, k, cfg)

    if cfg['render_mode'] == 'centers':
            init_pts = o3d.utility.Vector3dVector(params['means3D'].contiguous().double().cpu().numpy())
            init_cols = o3d.utility.Vector3dVector(params['rgb_colors'].contiguous().double().cpu().numpy())


    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    # Initialize Estimated Camera Frustums
    frustum_size = 0.045
    num_t = len(all_w2cs)
    cam_centers = []
    cam_colormap = plt.get_cmap('cool') # 
    norm_factor = 0.5
    total_num_lines = num_t - 1
    red_colormap = LinearSegmentedColormap.from_list("red_colormap", [(0, 'red'), (1, 'red')])
    line_colormap = plt.get_cmap('cool')
    
    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()

    cparams = o3d.camera.PinholeCameraParameters()
    first_view_w2c = first_frame_w2c
    first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])
    cparams.extrinsic = first_view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])

    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # Rendering of Online Reconstruction
    start_time = time.time()
    num_timesteps = num_t
    viz_start = True
    curr_timestep = 0
    if cfg['save_video']:
        frames = []
    while curr_timestep < (num_timesteps-1): #or not cfg['enter_interactive_post_online']:
        passed_time = time.time() - start_time
        passed_frames = passed_time * cfg['viz_fps']
        curr_timestep = int(passed_frames % num_timesteps)
        if not viz_start:
            if curr_timestep == prev_timestep:
                continue

        # Update Camera Frustum
        if curr_timestep == 0:
            cam_centers = []
            if not viz_start:
                vis.remove_geometry(prev_lines)
        if not viz_start:
            vis.remove_geometry(prev_frustum)
        new_frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[curr_timestep], frustum_size)
        new_frustum.paint_uniform_color(np.array(red_colormap(curr_timestep * norm_factor / num_t)[:3]))
        # new_frustum.paint_uniform_color(np.array(cam_colormap(curr_timestep * norm_factor / num_t)[:3]))
        vis.add_geometry(new_frustum)
        prev_frustum = new_frustum
        cam_centers.append(np.linalg.inv(all_w2cs[curr_timestep])[:3, 3])
        
        # Update Camera Trajectory
        if len(cam_centers) > 1 and curr_timestep > 0:
            num_lines = [1]
            cols = []
            for line_t in range(curr_timestep):
                cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
                # cols.append(np.array(red_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
            cols = np.array(cols)
            all_cols = [cols]
            out_pts = [np.array(cam_centers)]
            linesets = make_lineset(out_pts, all_cols, num_lines)
            lines = o3d.geometry.LineSet()
            lines.points = linesets[0].points
            lines.colors = linesets[0].colors
            lines.lines = linesets[0].lines
            vis.add_geometry(lines)
            prev_lines = lines
        elif not viz_start:
            vis.remove_geometry(prev_lines)

        # Get Current View Camera
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        view_w2c = cam_params.extrinsic
        view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])

        cam_params.extrinsic = view_w2c
        # cam_params.extrinsic = np.eye(4)

        view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)


        scene_data = get_rendervars_sem(params, view_w2c, curr_timestep=curr_timestep)
        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            # rendered_image, rendered_objects, rendered_depth
            im, depth, sem_feature = render_sam(view_w2c, k, scene_data, cfg)

            if cfg['render_mode'] == 'color' or  cfg['render_mode'] == 'depth':
                pts, cols = rgbd2pcd(im, depth, view_w2c, k, cfg)

            if cfg['render_mode'] == 'sem':
                logits = semantic_decoder(sem_feature)
                pts, cols = rgbd2pcd_sem(logits, depth, view_w2c, k, cfg)

            if cfg['render_mode'] == 'sem_color':
                logits = semantic_decoder(sem_feature)
                pts, cols = rgbd2pcd_sem_color(im, logits, depth, view_w2c, k, cfg)

            if cfg['render_mode'] == 'sem_feature':
                pts, cols = rgbd2pcd_sem_feature(sem_feature, depth, view_w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

        if cfg['save_video']:
            img = vis.capture_screen_float_buffer(False)
            img_np = np.asarray(img)  # 将Open3D图像对象转换为NumPy数组
            img_uint8 = (img_np * 255).astype('uint8')  # 将图像转换为 uint8 格式
            frames.append(img_uint8)  # 将图像添加到帧列表中

        prev_timestep = curr_timestep
        viz_start = False

    if cfg['save_video']:
        imageio.mimwrite(video_path, frames, fps=30)

    if cfg["save_imgs"]:
        sem_color_imgs_path = os.path.join(cfg['logdir'], "sem_color_imgs")
        os.makedirs(sem_color_imgs_path, exist_ok=True)
        for i, frame in enumerate(frames[::10]):
            imageio.imwrite(os.path.join(sem_color_imgs_path, f"sc_{i}.png"), frame)

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, help="Path to experiment file")
    parser.add_argument("--mode", type=str, default='color',
                        help="['color, 'depth', 'centers', 'sem', 'sem_color', 'sem_feature']")
    parser.add_argument("--save_video", type=bool, default=False, help="save video")
    parser.add_argument("--save_imgs", type=bool, default=False, help="save imgs")
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
    viz_cfg['save_video'] = args.save_video
    viz_cfg['save_imgs'] = args.save_imgs
 
    # Visualize Final Reconstruction
    online_recon(viz_cfg)