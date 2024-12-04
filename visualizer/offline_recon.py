import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import open3d as o3d
# import keyboard
import torch
import imageio
import json

from src.utils.common_utils import seed_everything
from src.Decoder import SemanticDecoder
from visualizer.viser_utils import (
    load_camera_recon,
    load_scene_data,
    render_sam,
    rgbd2pcd,
    rgbd2pcd_sem,
    rgbd2pcd_sem_color,
    rgbd2pcd_sem_feature,
    make_lineset,
    save_screenshot
)


def offine_recon(cfg, follow_cam=False):
    scene_path = os.path.join(cfg["logdir"], "params.npz")
    video_path = os.path.join(cfg["logdir"], cfg["render_mode"] + str("_it.mp4"))
    # sem_color_imgs_path = os.path.join(cfg['logdir'], "sem_color_imgs")
    # os.makedirs(sem_color_imgs_path, exist_ok=True)

    # load semantic decoder
    classifier_path = os.path.join(cfg["logdir"], "classifier.pth")
    semantic_decoder = SemanticDecoder(16, 256)
    semantic_decoder.load_state_dict(torch.load(classifier_path))
            
    # Load Scene Data
    w2c, k = load_camera_recon(cfg, scene_path)

    scene_data, all_w2cs = load_scene_data(scene_path, w2c, k)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']), 
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)


   
    # rendered_image, rendered_objects, rendered_depth
    im, depth, sem_feature = render_sam(w2c, k, scene_data, cfg)

    if cfg['render_mode'] == 'color' or  cfg['render_mode'] == 'depth':
        init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'sem':
        logits = semantic_decoder(sem_feature)
        init_pts, init_cols = rgbd2pcd_sem(logits, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'sem_color':
        logits = semantic_decoder(sem_feature)
        init_pts, init_cols = rgbd2pcd_sem_color(im, logits, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'sem_feature':
        init_pts, init_cols = rgbd2pcd_sem_feature(sem_feature, depth, w2c, k, cfg)

    if cfg['render_mode'] == 'centers':
            init_pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            init_cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())

    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols

    # export mesh
    # print("export mesh....")
    # mesh = create_mesh_from_point_cloud(pcd)

    # # args.output_path
    # o3d.io.write_triangle_mesh("mesh.ply", mesh)
    # print("export mesh done....")

    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']
    
    if cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        red_colormap = LinearSegmentedColormap.from_list("red_colormap", [(0, 'red'), (1, 'red')])
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            if i_t % 50 != 0:
                continue
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(red_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])
        
        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(red_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)


    # Initialize View Control
    view_control = vis.get_view_control()
    if os.path.exists(viz_cfg['cam_json']):
        with open(viz_cfg['cam_json'], 'r') as f:
            cam_params_dict = json.load(f)

        cparams = o3d.camera.PinholeCameraParameters()
        cparams.extrinsic = cam_params_dict['extrinsic']
        cparams.intrinsic.intrinsic_matrix = cam_params_dict['intrinsic']['intrinsic_matrix']
        cparams.intrinsic.height = cam_params_dict['intrinsic']['height']
        cparams.intrinsic.width = cam_params_dict['intrinsic']['width']

        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
    else:
        view_k = k * cfg['view_scale']
        view_k[2, 2] = 1
        # view_control = vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        if cfg['offset_first_viz_cam']:
            view_w2c = w2c
            view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
        else:
            view_w2c = w2c
        cparams.extrinsic = view_w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
        cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    if viz_cfg['save_video']:
        frames = []

    # Interactive Rendering
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sem_feature = render_sam(w2c, k, scene_data, cfg)

            if cfg['render_mode'] == 'color' or  cfg['render_mode'] == 'depth':
                pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)

            if cfg['render_mode'] == 'sem':
                logits = semantic_decoder(sem_feature)
                pts, cols = rgbd2pcd_sem(logits, depth, w2c, k, cfg)

            if cfg['render_mode'] == 'sem_color':
                logits = semantic_decoder(sem_feature)
                pts, cols = rgbd2pcd_sem_color(im, logits, depth, w2c, k, cfg)

            if cfg['render_mode'] == 'sem_feature':
                pts, cols = rgbd2pcd_sem_feature(sem_feature, depth, w2c, k, cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        # if keyboard.is_pressed('alt+c'):
        #     img = vis.capture_screen_float_buffer(False)
        #     img_np = np.asarray(img)
        #     img_uint8 = (img_np * 255).astype('uint8')

        #     cam_params_dict = {
        #             "intrinsic": {
        #                 "intrinsic_matrix": cam_params.intrinsic.intrinsic_matrix.tolist(),
        #                 "width": cam_params.intrinsic.width,
        #                 "height": cam_params.intrinsic.height
        #             },
        #             "extrinsic": cam_params.extrinsic.tolist()
        #         }
        #     save_screenshot(img_uint8, cam_params_dict)
            
        if not vis.poll_events():
            break

        vis.update_renderer()

        if cfg['save_video']:
            img = vis.capture_screen_float_buffer(False)
            img_np = np.asarray(img)
            img_uint8 = (img_np * 255).astype('uint8')
            frames.append(img_uint8)


    if cfg['save_video']:
        imageio.mimwrite(video_path, frames, fps=30)

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
    parser.add_argument("--plot_traj", type=bool, default=False, help="plot trajectory")
    parser.add_argument("--save_video", type=bool, default=False, help="save video")
    parser.add_argument("--cam_json", type=str, default='camera.json', help="save video")
    args = parser.parse_args()

    print(args.logdir)
    config_path = os.path.join(args.logdir, 'config.py')
    config = SourceFileLoader(
        os.path.basename(config_path), config_path
    ).load_module()

    seed_everything(seed=config.seed)


    scene_path = os.path.join(args.logdir, "params.npz")
    classifier_path = os.path.join(args.logdir, "classifier.pth")

    viz_cfg = config.config['viz']
    viz_cfg["render_mode"] = args.mode
    viz_cfg['save_video'] = args.save_video
    viz_cfg['cam_json'] = args.cam_json
    viz_cfg['logdir'] = args.logdir
    viz_cfg['visualize_cams'] = args.plot_traj
    
    # Visualize Final Reconstruction
    offine_recon(viz_cfg)