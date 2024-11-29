import os
import sys
import time
import random

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.dataconfig import load_dataset_config, get_dataset
from src.Mapper import initialize_first_timestep, add_new_gaussians_alpha
from src.Tracker import initialize_camera_pose
from src.Loss import initialize_optimizer, get_loss
from src.Evaluater import eval
from src.GaussianManager import prune_gaussians, densify
from src.Decoder import SemanticDecoder
from src.utils.logger import report_progress, save_params_ckpt, save_params
from src.utils.gaussian_utils import matrix_to_quaternion, build_rotation
from src.utils.common_utils import seed_everything

def run_gs3lam(config: dict):
    seed_everything(seed=config['seed'])

    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000

    # print(f"{config}")
        
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device(config["primary_device"])

    # Load Dataset
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        use_train_split=dataset_config["use_train_split"],
    )

    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Initialize Parameters & Canoncial Camera parameters
    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(
        dataset, num_frames, config['scene_radius_depth_ratio'],
        config['mean_sq_dist_method'], gaussian_distribution=config['gaussian_distribution'],
        num_objects=config['semantic']["num_objects"]
    )
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    frame_freps = {}

    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Iterate over Scan
    t_fps_list = []
    t_iter_start = torch.cuda.Event(enable_timing=True)
    t_iter_end = torch.cuda.Event(enable_timing=True)
    m_fps_list = []
    m_iter_start = torch.cuda.Event(enable_timing=True)
    m_iter_end = torch.cuda.Event(enable_timing=True)

    # Semantic Decoder
    semantic_decoder = SemanticDecoder(config['semantic']["num_objects"], config['semantic']["num_classes"])

    for time_idx in tqdm(range(0, num_frames)):
        #####################################################
        ###                 Data Reader                   ###
        #####################################################

        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose, gt_objects = dataset[time_idx]
        

        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # BGR->RGB
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames

        iter_time_idx = time_idx
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'obj': gt_objects, 
                     'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}


        #####################################################
        ###                 Tracking                      ###
        #####################################################
        tracking_curr_data = curr_data
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Start Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            tracking_optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            tracking_semantic_optimizer = torch.optim.Adam(semantic_decoder.parameters(), lr=0.0)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)

            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")

            t_iter_start.record()
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params,
                                                    tracking_curr_data, 
                                                    variables,
                                                    iter_time_idx,
                                                    loss_weights=config['tracking']['loss_weights'],
                                                    use_l1=config['tracking']['use_l1'], 
                                                    ignore_outlier_depth_loss=config['tracking']['ignore_outlier_depth_loss'], 
                                                    tracking=True,
                                                    use_semantic_for_tracking=config['tracking']['use_semantic_for_tracking'],
                                                    use_alpha_for_loss=config['tracking']['use_alpha_for_loss'],
                                                    alpha_thres=config['tracking']['alpha_thres'],
                                                    semantic_decoder=semantic_decoder,
                                                    num_classes=config['semantic']['num_classes'])

                # Backprop
                loss.backward()
                # Optimizer Update
                tracking_optimizer.step()
                tracking_optimizer.zero_grad(set_to_none=True)
                tracking_semantic_optimizer.step()
                tracking_semantic_optimizer.zero_grad()

                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                torch.cuda.empty_cache()

                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    if config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                    else:
                        break

            t_iter_end.record()
            torch.cuda.synchronize()
            t_iter_time = t_iter_start.elapsed_time(t_iter_end) / 1000.0
            t_fps_list.append(1.0 / t_iter_time)

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')



        #####################################################
        ###                 Mapping                       ###
        #####################################################
                
        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                densify_curr_data = curr_data
                
                params, variables = add_new_gaussians_alpha(params, variables, densify_curr_data, 
                                                    config['mapping']['densify_thres'], time_idx,
                                                    config['mean_sq_dist_method'],
                                                    gaussian_distribution=config['gaussian_distribution'],
                                                    num_objects=config['semantic']["num_objects"])

                # post_num_pts = params['means3D'].shape[0]
               
            # with torch.no_grad():
            #     # Get the current estimated rotation & translation
            #     curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
            #     curr_cam_tran = params['cam_trans'][..., time_idx].detach()
            #     curr_w2c = torch.eye(4).cuda().float()
            #     curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            #     curr_w2c[:3, 3] = curr_cam_tran
            #     # Select Keyframes for Mapping
            #     num_keyframes = config['mapping_window_size']-2
            #     # Check curr data for keyframe selection
            #     selected_keyframes = keyframe_selection_overlap_object(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
            #     selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
            #     if len(keyframe_list) > 0:
            #         # Add last keyframe to the selected keyframes
            #         selected_time_idx.append(keyframe_list[-1]['id'])
            #         selected_keyframes.append(len(keyframe_list)-1)
            #     # Add current frame to the selected keyframes
            #     selected_time_idx.append(time_idx)
            #     selected_keyframes.append(-1)
            #     # Print the selected keyframes
            #     print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            mapping_optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)
            mapping_semantic_optimizer = torch.optim.Adam(semantic_decoder.parameters(), lr=5e-4)

            # Mapping
            mapping_start_time = time.time()
            if time_idx == 0:
                frame_iter_mapping = config['mapping']['first_frame_mapping_iters']
            else:
                frame_iter_mapping = num_iters_mapping
            if frame_iter_mapping > 0:
                progress_bar = tqdm(range(frame_iter_mapping), desc=f"Mapping Time Step: {time_idx}, pts: {params['means3D'].shape[0]}")
            
            m_iter_start.record()
            for iter in range(frame_iter_mapping):
                torch.cuda.empty_cache()
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                if time_idx == 0 or iter % config['mapping']['opt_rskm_interval'] == 0:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                    if config['mapping']["use_semantic_for_mapping"]:
                        iter_object = gt_objects
                else:
                    keyframe_idx = random.choice(range(len(keyframe_time_indices)))
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[keyframe_idx]['id']
                    iter_color = keyframe_list[keyframe_idx]['color']
                    iter_depth = keyframe_list[keyframe_idx]['depth']
                    if config['mapping']["use_semantic_for_mapping"]:
                        iter_object = keyframe_list[keyframe_idx]['obj']

                # Record keyframe opt times
                if iter_time_idx in frame_freps:
                    frame_freps[iter_time_idx] += 1
                else:
                    frame_freps[iter_time_idx] = 1

                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                
                if config['mapping']["use_semantic_for_mapping"]:
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c,
                             "obj": iter_object}
                else:
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                
                # Loss for current frame
                loss, variables, losses = get_loss(params=params,
                                                   curr_data=iter_data,
                                                   variables=variables,
                                                   iter_time_idx=iter_time_idx,
                                                   loss_weights=config['mapping']['loss_weights'],
                                                   use_l1=config['mapping']['use_l1'],
                                                   ignore_outlier_depth_loss=config['mapping']['ignore_outlier_depth_loss'],
                                                   mapping=True,
                                                   use_reg_loss=config['mapping']['use_reg_loss'],
                                                   use_semantic_for_mapping=config['mapping']['use_semantic_for_mapping'],
                                                   semantic_decoder=semantic_decoder,
                                                   num_classes=config['semantic']["num_classes"])
                     
               
                loss.backward()
                
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, mapping_optimizer, iter, config['mapping']['pruning_dict'])
                        
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, mapping_optimizer, iter, config['mapping']['densify_dict'])
                       
                    # Optimizer Update
                    mapping_optimizer.step()
                    mapping_optimizer.zero_grad(set_to_none=True)
                    mapping_semantic_optimizer.step()
                    mapping_semantic_optimizer.zero_grad()

                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            
            m_iter_end.record()
            torch.cuda.synchronize()
            m_iter_time = m_iter_start.elapsed_time(m_iter_end) / 1000.0
            m_fps_list.append(1.0 / m_iter_time)

            if frame_iter_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        report_progress(params, curr_data, 1, progress_bar, time_idx,
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c,
                                 'color': color, 'depth': depth, "obj": gt_objects}
                
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))

        torch.cuda.empty_cache()
    
    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count


    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    print("Tracking FPS: {:.5f}".format(sum(t_fps_list) / len(t_fps_list)))
    print("Mapping FPS: {:.5f}".format(sum(m_fps_list) / len(m_fps_list)))

    # save result
    running_times = []
    running_times.append(f"Average Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    running_times.append(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    running_times.append(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    running_times.append(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    running_times = np.array(running_times, dtype='str')
    np.savetxt(os.path.join(eval_dir, "running_times.txt"), running_times, fmt='%s')

    keys = np.array(list(frame_freps.keys()))
    values = np.array(list(frame_freps.values()))
    np.savez(os.path.join(eval_dir, "keyframe_freq_keys.npz"), keys=keys, values=values)
    
    # Evaluate Final Parameters
    with torch.no_grad():
        eval(dataset, params, num_frames, eval_dir,
                mapping_iters=config['mapping']['num_iters'], 
                add_new_gaussians=config['mapping']['add_new_gaussians'],
                eval_every=config['eval_every'],
                use_semantic=config["use_semantic"],
                classifier=semantic_decoder if config["use_semantic"] else None)

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)
    torch.save(semantic_decoder.state_dict(), os.path.join(output_dir, "classifier.pth"))