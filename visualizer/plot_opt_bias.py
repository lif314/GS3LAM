import torch
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import click

mean_psnr_latex = r'$\mu_{\mathrm{PSNR}}$'
variance_psnr_latex = r'$\sigma_{\mathrm{PSNR}}$'
db2_latex = r'$\mathrm{dB}^2$'
db_latex = r'$\mathrm{dB}$'

def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def load_w2cs(scene_path):
    params = dict(np.load(scene_path, allow_pickle=True))

    params = {k: torch.tensor(params[k]).float() for k in params.keys()}

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.numpy())

    return all_w2cs

def plot_camera_trajectory_2d_with_psnr_opt(all_w2cs, psnrs, opt_freqs, save_path):
    """
    Plot camera trajectory in 2D space with PSNR values and optimization frequencies.

    Parameters:
        all_w2cs (list): List of camera poses (4x4 numpy arrays).
        psnrs (list): List of PSNR values corresponding to each camera pose.
        opt_freqs (list): List of optimization frequencies corresponding to each camera pose.
    """
    # Extract camera positions
    positions = np.array([pose[:2, 3] for pose in all_w2cs])
    
    # Extract PSNR values
    psnr_values = np.array(psnrs)
    
    # Extract optimization frequencies
    opt_freqs = np.array(opt_freqs)

    # Calculate PSNR variance
    psnr_variance = np.var(psnr_values)

    # Create 2D plot
    plt.figure()

    # Define normalization for PSNR values
    norm = plt.Normalize(vmin=min(psnr_values), vmax=max(psnr_values))
    # norm = plt.Normalize(vmin=15, vmax=45)
    
    # Plot camera positions with PSNR as color and optimization frequency as size
    plt.scatter(positions[:, 0], positions[:, 1], c=psnr_values, cmap='viridis', norm=norm, s=opt_freqs / 10.0, label='Camera Position (Radius = Opt. Iters)')
    

    plt.plot(positions[:, 0], positions[:, 1], color='red', linestyle='-',  linewidth=0.5, label='Camera Trajectory')
    
    # Plot start position as red point
    plt.plot(positions[0, 0], positions[0, 1], marker='o', color='red', markersize=5, label='Start Point')

    plt.colorbar(label='PSNR')
    plt.title(f'{variance_psnr_latex}: {psnr_variance:.2f} {db2_latex}, {mean_psnr_latex}: {np.mean(psnr_values):.2f} {db_latex}', fontsize=15)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # Set legend
    # plt.legend(['Camera Positions (Optimization Frequency)'])
    # plt.legend()
    # plt.legend(loc='upper left')
    plt.legend(loc='lower left')

    # Save plot as PDF
    plt.savefig(save_path, format='pdf', dpi=300)

    # Show plot
    plt.grid(True)
    plt.axis('equal')
    plt.show()

@click.command()
@click.option('--logdir', type=str)
@click.option('--out_name', type=str, default='camera_trajectory_plot.pdf')
def main(logdir, out_name):
    scene_path = os.path.join(logdir, "params.npz")
    all_w2cs = load_w2cs(scene_path)

    keyframe_freq_path = os.path.join(logdir, "eval", "keyframe_freq_keys.npz")

    print("keyframe_freq_path: ", keyframe_freq_path)
    data = np.load(keyframe_freq_path)
    selected_data = {key: value for key, value in zip(data['keys'], data['values']) if key==0 or (key+1)%5==0}
    selected_keys = list(selected_data.keys())

    selected_values = list(selected_data.values())

    psnr_path = os.path.join(logdir, "eval", "psnr.txt")
    with open(psnr_path, 'r') as file:
        lines = file.readlines()
        psnrs = [float(line.strip()) for line in lines]

    print("Num of psnrs: ", len(psnrs))
    downsample_poses = [pose for i, pose  in enumerate(all_w2cs) if (i+1)%5==0 or i==0]
    print("Num of poses: ", len(downsample_poses))

    save_path = os.path.join(logdir, out_name)
    plot_camera_trajectory_2d_with_psnr_opt(downsample_poses, psnrs, selected_values, save_path)

if __name__ == "__main__":
    main()