import torch
import torch.nn.functional as F
from gaussian_semantic_rasterization import GaussianRasterizationSettings

def get_rasterizationSettings(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)

    raster_settings = GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False
    )
    return raster_settings

# All Rendering settings
def transformed_params2rendervar(params, transformed_gaussians):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    
    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': params['rgb_colors'],
        'sh_objs': params["obj_dc"],
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }

    return rendervar
