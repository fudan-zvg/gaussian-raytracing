import os
import numpy as np
import torch
import shutil

# CUDA extension
try:
    import _raytracing as _backend
except Exception as e:
    from torch.utils.cpp_extension import load

    _src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # # uncomment these lines to clean the cache
    # if os.path.exists(os.path.join(_src_path, 'build')):
    #     shutil.rmtree(os.path.join(_src_path, 'build'))
    # os.system("mkdir build && cd build && cmake .. && make")
    
    _backend = load(
        name="_raytracing",
        extra_cuda_cflags=[
            "-O3", "-std=c++17", 
            "--expt-extended-lambda", 
            "--expt-relaxed-constexpr", 
            "-U__CUDA_NO_HALF_OPERATORS__", 
            "-U__CUDA_NO_HALF_CONVERSIONS__", 
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        sources=[
            os.path.join(_src_path, "src", f) for f in [
                "bvh.cu",
                "bindings.cu",
            ]
        ],
        extra_include_paths=[
            os.path.join(_src_path, "include"), 
            os.path.join(_src_path, "build"),
            os.path.join(_src_path, "include", "optix"), 
        ],
        build_directory=os.path.join(_src_path, 'build'),
        verbose=True)


class _GaussianTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bvh, rays_o, rays_d, gs_idxs, means3D, opacity, SinvR, shs, alpha_min, transmittance_min, deg):    
        colors = torch.zeros_like(rays_o)
        depth = torch.zeros_like(rays_o[:, 0])
        alpha = torch.zeros_like(rays_o[:, 0])
        bvh.trace_forward(
            rays_o, rays_d, gs_idxs, means3D, opacity, SinvR, shs, 
            colors, depth, alpha, 
            alpha_min, transmittance_min, deg,
        )
        
        # Keep relevant tensors for backward
        ctx.alpha_min = alpha_min
        ctx.transmittance_min = transmittance_min
        ctx.deg = deg
        ctx.bvh = bvh
        ctx.save_for_backward(rays_o, rays_d, gs_idxs, means3D, opacity, SinvR, shs, colors, depth, alpha)
        return colors, depth, alpha

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_alpha):
        rays_o, rays_d, gs_idxs, means3D, opacity, SinvR, shs, colors, depth, alpha = ctx.saved_tensors
        
        grad_means3D = torch.zeros_like(means3D)
        grad_opacity = torch.zeros_like(opacity)
        grad_SinvR = torch.zeros_like(SinvR)
        grad_shs = torch.zeros_like(shs)
        
        ctx.bvh.trace_backward(
            rays_o, rays_d, gs_idxs, means3D, opacity, SinvR, shs, 
            colors, depth, alpha, 
            grad_means3D, grad_opacity, grad_SinvR, grad_shs,
            grad_out_color, grad_out_depth, grad_out_alpha,
            ctx.alpha_min, ctx.transmittance_min, ctx.deg,
        )
        grads = (
            None,
            None,
            None,
            None,
            grad_means3D,
            grad_opacity,
            grad_SinvR,
            grad_shs,
            None,
            None,
            None,
        )

        return grads


class GaussianTracer():
    def __init__(self):
        self.impl = _backend.create_gaussiantracer()
        
    def build_bvh(self, vertices_b, faces_b, gs_idxs):
        self.faces_b = faces_b
        self.gs_idxs = gs_idxs.int()
        self.impl.build_bvh(vertices_b[faces_b])

    def update_bvh(self, vertices_b, faces_b, gs_idxs):
        assert (self.faces_b == faces_b).all(), "Update bvh must keep the triangle id not change~"
        self.gs_idxs = gs_idxs.int()
        self.impl.update_bvh(vertices_b[faces_b])

    def trace(self, rays_o, rays_d, means3D, opacity, SinvR, shs, alpha_min=0.01, transmittance_min=0.001, deg=3):
    # def trace(self, rays_o, rays_d, use_optix=True):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]
        # inplace: write positions to rays_o, face_normals to rays_d

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        # inplace write intersections back to rays_o
        # SinvR = SinvR.transpose(-1,-2)
        colors, depth, alpha = _GaussianTrace.apply(self.impl, rays_o, rays_d, self.gs_idxs, means3D, opacity, SinvR, shs, alpha_min, transmittance_min, deg)
        # import pdb;pdb.set_trace()

        colors = colors.view(*prefix, 3)
        depth = depth.view(*prefix)
        alpha = alpha.view(*prefix)
        
        
        # means3D.retain_grad()
        # opacity.retain_grad()
        # shs.retain_grad()
        # SinvR.retain_grad()
        # (colors.mean()+depth.mean()+alpha.mean()).backward(retain_graph=True)
        # torch.cuda.synchronize()
        # # import pdb;pdb.set_trace()
        # if means3D.grad.isnan().any():
        #     import pdb;pdb.set_trace()
        # if opacity.grad.isnan().any():
        #     import pdb;pdb.set_trace()
        # if shs.grad.isnan().any():
        #     import pdb;pdb.set_trace()
        # if SinvR.grad.isnan().any():
        #     import pdb;pdb.set_trace()
           
        # import trimesh
        # idx = 458776
        # points = (rays_o[idx][None] + torch.linspace(0,10,1000).cuda()[:, None] * rays_d[idx][None])
        # trimesh.Trimesh(points.detach().cpu().numpy()).export("test3.ply")

        return colors, depth, alpha