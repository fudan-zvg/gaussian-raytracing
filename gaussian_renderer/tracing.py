import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def render_image_trace(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):  
    rays_o, rays_d = viewpoint_camera.get_rays()
    # import pdb;pdb.set_trace()
    # import trimesh
    # m=trimesh.Trimesh((rays_o[:, None]+rays_d[:, None]*torch.linspace(0,1,10).cuda()[None, :, None]).reshape(-1, 3).detach().cpu().numpy()).export("test.ply")
    outputs = pc.trace(rays_o, rays_d)
    alpha = outputs['alpha'][:, None]
    depth = outputs['depth'][:, None]
    render = outputs['render']
    render = render + bg_color * (1 - alpha) 

    # import pdb;pdb.set_trace()
    return {
        "render":render.reshape(viewpoint_camera.image_height, viewpoint_camera.image_width, 3).permute(2, 0, 1),
        "depth": depth.reshape(viewpoint_camera.image_height, viewpoint_camera.image_width, 1).permute(2, 0, 1),
        "alpha": alpha.reshape(viewpoint_camera.image_height, viewpoint_camera.image_width, 1).permute(2, 0, 1),
    }


def render_trace(rays_o, rays_d, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):  
    outputs = pc.trace(rays_o, rays_d)
    alpha = outputs['alpha'][:, None]
    depth = outputs['depth'][:, None]
    render = outputs['render']
    render = render + bg_color * (1 - alpha) 
    return {
        "render": render,
        "depth": depth,
        "alpha": alpha,
    }
    