from datetime import datetime

primary_device="cuda:0"
seed = 1
scene_name = "office0"

basedir = "./data/Replica"

first_frame_mapping_iters = 1000
tracking_iters = 40
mapping_iters = 60
opt_rskm_interval = 5
densify_thres=0.1 # For adding new Gaussians
end_frame = -1

use_semantic_for_mapping=True
map_every = 1
keyframe_every = 5
mapping_window_size = 24

group_name = "Replica"
run_name = str(datetime.now().strftime("%y%m%d-%H:%M:%S"))

config = dict(
    workdir=f"./logs/{group_name}/{scene_name}_seed{seed}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=100, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # (Meters) Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # Mean Squared Distance Calculation for Scale of Gaussians
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    densify_method="alpha", # ['depth_sil', 'alpha']
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    data=dict(
        basedir=basedir,
        gradslam_data_cfg="./configs/camera/replica.yaml",
        sequence=scene_name,
        use_train_split=True,
        desired_image_height=680,
        desired_image_width=1200,
        start=0,
        end=end_frame,
        stride=1,
        num_frames=-1,
    ),
    use_semantic=True,
    semantic=dict(
        use_pretrain=False,
        pretrain_path="./pretrain",
        num_objects=16, # in_channels
        num_classes=256, # out_channels
        inter_dims=[], #[64, 128],
        use_obj_3d_loss=False,
        use_3d_obj_for_tracking=False,
        reg3d_interval=5,
        reg3d_k = 5,
        reg3d_lambda_val = 2,
        reg3d_max_points = 2_0000,
        reg3d_sample_size = 1000,
        loss_obj_weight=1.0,
        loss_obj_3d_weight=1_000.0,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        use_semantic_for_tracking=True,
        use_obj2d_for_tracking=False,
        use_obj3d_for_tracking=False,
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_alpha_for_loss=True,
        # relative pose constraint
        use_rel_pose_loss=False,
        alpha_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            obj=0.001,
            big_gaussian_reg=0.05,
            small_gaussian_reg=0.005,
            rel_rgb=0.10,
            rel_depth=0.10,
            obj_3d=1000.0
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
            obj_dc=0.0,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        first_frame_mapping_iters=first_frame_mapping_iters,
        add_new_gaussians=True,
        densify_thres=densify_thres, # For Addition of new Gaussians
        use_semantic_for_mapping=True,
        opt_rskm_interval=opt_rskm_interval,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        use_sil_for_loss=False,
        # regularize Gaussians
        use_reg_loss=False,
        #use_reg_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            obj=0.01,
            big_gaussian_reg=0.01,
            small_gaussian_reg=0.001,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            obj_dc=0.0025,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)
