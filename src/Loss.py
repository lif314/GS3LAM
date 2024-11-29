import torch

from src.utils.gaussian_utils import transform_to_frame
from src.Render import transformed_params2rendervar
from src.utils.metric_utils import calc_ssim, l1_loss_v1
from gaussian_semantic_rasterization import GaussianRasterizer

def initialize_optimizer(params, lrs_dict, tracking):
    param_groups = [{'params': [v], 'name': k, 'lr': lrs_dict[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, 
             use_l1,ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, use_reg_loss=False,
             semantic_decoder=None,
             use_semantic_for_tracking=True,
             use_semantic_for_mapping=True,
             use_alpha_for_loss=False,
             alpha_thres=0.99,
             num_classes=256):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    
    # Rendering
    rendervar['means2D'].retain_grad()
    rendered_image, rendered_objects, radii, rendered_depth, rendered_alpha = GaussianRasterizer(raster_settings=curr_data["cam"])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(rendered_depth))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - rendered_depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10 * depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask

    if tracking and use_alpha_for_loss:
        presence_alpha_mask = (rendered_alpha > alpha_thres)
        mask = mask & presence_alpha_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].mean()
    
    # RGB Loss
    if tracking and ignore_outlier_depth_loss:
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - rendered_image)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - rendered_image).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(rendered_image, curr_data['im']) + 0.2 * (1.0 - calc_ssim(rendered_image, curr_data['im']))
    
    gt_obj = curr_data["obj"].long()
    logits = semantic_decoder(rendered_objects)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if tracking and use_semantic_for_tracking:
        if ignore_outlier_depth_loss:
            obj_mask = mask.detach().squeeze(0)
            loss_obj =  cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze()[obj_mask].sum()
        else:
            loss_obj =  cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().sum()

        losses['obj'] = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

    if mapping and use_semantic_for_mapping:
        loss_obj =  cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        losses['obj'] = loss_obj / torch.log(torch.tensor(num_classes))

    # regularize Gaussians, scale, meters
    if mapping and use_reg_loss:
        scaling = torch.exp(params['log_scales'])
        mean_scale = scaling.mean()
        std_scale = scaling.std()
        # 1 sigma: 68.3%; 2 sigma 95.4%; 3 sigma 99.7%
        upper_limit = mean_scale + 2 * std_scale
        lower_limit = mean_scale - 2 * std_scale
        # regularize very big Gaussian
        if upper_limit < scaling.max():
            losses["big_gaussian_reg"] = torch.mean(scaling[torch.where(scaling > upper_limit)])
        else:
            losses["big_gaussian_reg"] = 0.0
        # regularize very small Gaussian
        if lower_limit > scaling.min():
            losses["small_gaussian_reg"] = torch.mean(-torch.log(scaling[torch.where(scaling < lower_limit)]))
        else:
            losses["small_gaussian_reg"] = 0.0

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radii > 0
    variables['max_2D_radius'][seen] = torch.max(radii[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses