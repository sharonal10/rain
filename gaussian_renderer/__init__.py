import torch
import math
import sys
sys.path.append('..')
from submodules.diff_gaussian_rasterization.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, RGB2SH

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, low_pass = 0.3):    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        low_pass=low_pass
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}

def render_multi(viewpoint_camera, gaussians_list, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, low_pass = 0.3, gaussian_id=None, center_id=None):
    '''
    if gaussian_id=None, render everything normally.
    else, one item will be normal, all else is black.
    center_id=None means the original item is normal.
    else, an offset version is normal.
    '''
    xyz = []
    append_range = []
    current_index = 0
    for pc in gaussians_list:
        xyz.append(pc.get_xyz)
        if pc.id == gaussian_id and center_id==None:
            assert append_range == [], append_range
            append_range = [current_index, current_index + pc.get_xyz.shape[0]]
        current_index += pc.get_xyz.shape[0]
        for curr_id, center in enumerate(pc.centers):
            xyz.append(pc.get_xyz + center)
            if pc.id == gaussian_id and center_id==curr_id:
                assert append_range == [], append_range
                append_range = [current_index, current_index + pc.get_xyz.shape[0]]
            current_index += pc.get_xyz.shape[0]
    xyz = torch.cat(xyz, dim=0)
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians_list[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        low_pass=low_pass
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = []
    for pc in gaussians_list:
        opacity.append(pc.get_opacity)
        for center in pc.centers:
            opacity.append(pc.get_opacity)
    opacity = torch.cat(opacity, dim=0)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        assert False
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = []
        for pc in gaussians_list:
            scales.append(pc.get_scaling)
            for center in pc.centers:
                scales.append(pc.get_scaling)
        scales = torch.cat(scales, dim=0)
        
        rotations = []
        for pc in gaussians_list:
            rotations.append(pc.get_rotation)
            for center in pc.centers:
                rotations.append(pc.get_rotation)
        rotations = torch.cat(rotations, dim=0)

    shs = None
    colors_precomp = None
    if override_color is None:
        feats = []
        for pc in gaussians_list:
            if pc.id == None or (pc.id == gaussian_id and center_id==None):
                feats.append(pc.get_features)
            else:
                black = torch.zeros_like(pc.get_features)
                black[:, :3, 0 ] = RGB2SH(0.0)
                feats.append(black)
            for curr_id, center in enumerate(pc.centers):
                if pc.id == None or (pc.id == gaussian_id and center_id==curr_id):
                    feats.append(pc.get_features)
                else:
                    black = torch.zeros_like(pc.get_features)
                    black[:, :3, 0 ] = RGB2SH(0.0)
                    feats.append(black)
        if pipe.convert_SHs_python:
            shs_view = torch.cat(feats, dim=0).transpose(1, 2).view(-1, 3, (gaussians_list[0].max_sh_degree+1)**2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(torch.cat(feats, dim=0).shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians_list[0].active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = torch.cat(feats, dim=0)
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    if append_range:
        screenspace_points = screenspace_points[append_range[0]:append_range[1]]
        radii = radii[append_range[0]:append_range[1]]

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}
