#pragma once

#include <raytracing/common.h>
#include <raytracing/triangle.cuh>
#include <raytracing/gpu_memory.h>

#include <memory>

namespace raytracing {

class TriangleBvhBase {
public:
    TriangleBvhBase() {};
    virtual void gaussian_trace_forward(
        uint32_t n_elements, const vec3* rays_o, const vec3* rays_d, const int* gs_idxs, 
        const vec3* means3D, const float* opacity, const mat3x3* SinvR, const vec3* shs, 
        vec3* colors, float* depth, float* alpha, 
        const float alpha_min, const float transmittance_min, const int deg, const int max_coeffs, cudaStream_t stream) = 0;
    virtual void gaussian_trace_backward(
        uint32_t n_elements, const vec3* rays_o, const vec3* rays_d, const int* gs_idxs, 
        const vec3* means3D, const float* opacity, const mat3x3* SinvR, const vec3* shs, 
        const vec3* colors, const float* depth, const float* alpha, 
        vec3* grad_means3D, float* grad_opacity, mat3x3* grad_SinvR, vec3* grad_shs, 
        const vec3* grad_colors, const float* grad_depth, const float* grad_alpha, vec3* colors2, float* grad_w,
        const float alpha_min, const float transmittance_min, const int deg, const int max_coeffs, cudaStream_t stream) = 0;

    virtual void build_bvh(const Triangle* triangles, int n_triangles, cudaStream_t stream) = 0;
    virtual void update_bvh(const Triangle* triangles, int n_triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvhBase> make();
};

}