import torch
import math
import sys
sys.path.append('..')
from submodules.diff_gaussian_rasterization.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def rotation_matrix_to_quaternion(rotation_matrix):
    """Convert a 3x3 rotation matrix to a quaternion."""
    R = rotation_matrix
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return torch.tensor([qw, qx, qy, qz], device=rotation_matrix.device)

def rotate_quaternions(quaternions, rotation_quaternion):
    """Rotate a batch of quaternions by another quaternion."""
    q1 = rotation_quaternion.unsqueeze(0).expand_as(quaternions)
    q2 = quaternions

    # Hamilton product of two quaternions q1 and q2
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    new_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    new_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    new_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    new_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((new_w, new_x, new_y, new_z), dim=-1)

def rotate_around_z(xyz, angle_degrees, centroid):
    # Step 1: Translate to origin
    xyz_centered = xyz - centroid

    # Step 2: Apply rotation
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees, device=xyz.device))
    cos_angle = torch.cos(angle_radians)
    sin_angle = torch.sin(angle_radians)
    rotation_matrix = torch.tensor([
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0]
    ], device=xyz.device)
    xyz_rotated = torch.matmul(xyz_centered, rotation_matrix.T)

    # Step 3: Translate back to original centroid
    xyz_rotated = xyz_rotated + centroid
    return xyz_rotated, rotation_matrix

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, low_pass = 0.3, center_id=None):    
    if center_id is not None:
        xyz = pc.get_xyz + pc.centers[center_id]
    else:
        xyz = pc.get_xyz
    centroid = xyz.mean(dim=0)

    xyz, rotation_matrix = rotate_around_z(xyz, 90, centroid)

    # center to origin then scale
    xyz = ((xyz - centroid) * pc.scale) + centroid

    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0

    rotation_quaternion = rotation_matrix_to_quaternion(rotation_matrix)

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

    means3D = xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        assert False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if rotations is not None:
            rotations = rotate_quaternions(rotations, rotation_quaternion)

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
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

def render_multi(viewpoint_camera, gaussians_list, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, low_pass = 0.3):
    xyz = []
    for pc in gaussians_list:
        curr = pc.get_xyz
        centroid = curr.mean(dim=0)
        # center to origin then scale
        curr = ((curr - centroid) * pc.scale) + centroid
        xyz.append(curr)
        for center in pc.centers:
            curr = pc.get_xyz + center
            centroid = curr.mean(dim=0)
            # center to origin then scale
            curr = ((curr - centroid) * pc.scale) + centroid
            xyz.append(curr)
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
            feats.append(pc.get_features)
            for center in pc.centers:
                feats.append(pc.get_features)
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

    return {"render": rendered_image}
