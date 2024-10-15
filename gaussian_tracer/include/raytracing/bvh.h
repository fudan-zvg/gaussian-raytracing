#pragma once
#include <iostream>
#include <string>
#include <vector>

#include <cstdint>
#include <cmath>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <glm/glm.hpp>
#include <raytracing/gpu_memory.h>

#include <memory>

namespace raytracing {

class TriangleBvhBase {
public:
    TriangleBvhBase() {};
    virtual void gaussian_trace_forward(
        uint32_t n_elements, const glm::vec3* rays_o, const glm::vec3* rays_d, const int* gs_idxs, 
        const glm::vec3* means3D, const float* opacity, const glm::mat3x3* SinvR, const glm::vec3* shs, 
        glm::vec3* colors, float* depth, float* alpha, 
        const float alpha_min, const float transmittance_min, const int deg, const int max_coeffs, cudaStream_t stream) = 0;
    virtual void gaussian_trace_backward(
        uint32_t n_elements, const glm::vec3* rays_o, const glm::vec3* rays_d, const int* gs_idxs, 
        const glm::vec3* means3D, const float* opacity, const glm::mat3x3* SinvR, const glm::vec3* shs, 
        const glm::vec3* colors, const float* depth, const float* alpha, 
        glm::vec3* grad_means3D, float* grad_opacity, glm::mat3x3* grad_SinvR, glm::vec3* grad_shs, 
        const glm::vec3* grad_colors, const float* grad_depth, const float* grad_alpha,
        const float alpha_min, const float transmittance_min, const int deg, const int max_coeffs, cudaStream_t stream) = 0;

    virtual void build_bvh(const float* triangles, int n_triangles, cudaStream_t stream) = 0;
    virtual void update_bvh(const float* triangles, int n_triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvhBase> make();
};

}