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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <raytracing/bvh.h>

namespace py = pybind11;
namespace raytracing {

class GaussianTracer {
public:
    GaussianTracer(){
        triangle_bvh = TriangleBvhBase::make();
    }

    void build_bvh(const torch::Tensor& triangles){
        const size_t n_triangles = triangles.size(0);
        cudaStream_t m_stream = at::cuda::getCurrentCUDAStream();;
        triangle_bvh->build_bvh(triangles.data_ptr<float>(), n_triangles, m_stream);
    }

    void update_bvh(const torch::Tensor& triangles){
        const size_t n_triangles = triangles.size(0);
        cudaStream_t m_stream = at::cuda::getCurrentCUDAStream();;
        triangle_bvh->update_bvh(triangles.data_ptr<float>(), n_triangles, m_stream);
    }

    void trace_forward(
        const torch::Tensor rays_o, const torch::Tensor rays_d, const torch::Tensor gs_idxs, 
        const torch::Tensor means3D, const torch::Tensor opacity, const torch::Tensor SinvR, const torch::Tensor shs, 
        torch::Tensor colors, torch::Tensor depth, torch::Tensor alpha, 
        const float alpha_min, const float transmittance_min, const int deg
        ){
        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        int max_coeffs = shs.size(1);

        triangle_bvh->gaussian_trace_forward(
            n_elements, (const glm::vec3*)rays_o.data_ptr<float>(), (const glm::vec3*)rays_d.data_ptr<float>(), gs_idxs.data_ptr<int>(), 
            (const glm::vec3*)means3D.data_ptr<float>(), opacity.data_ptr<float>(), (const glm::mat3x3*)SinvR.data_ptr<float>(), (const glm::vec3*)shs.data_ptr<float>(), 
            (glm::vec3*)colors.data_ptr<float>(), depth.data_ptr<float>(), alpha.data_ptr<float>(), 
            alpha_min, transmittance_min, deg, max_coeffs, stream);
    }
    
    void trace_backward(
        const torch::Tensor rays_o, const torch::Tensor rays_d, const torch::Tensor gs_idxs, 
        const torch::Tensor means3D, const torch::Tensor opacity, const torch::Tensor SinvR, const torch::Tensor shs, 
        const torch::Tensor colors, const torch::Tensor depth, const torch::Tensor alpha, 
        torch::Tensor grad_means3D, torch::Tensor grad_opacity, torch::Tensor grad_SinvR, torch::Tensor grad_shs, 
        const torch::Tensor grad_out_color, const torch::Tensor grad_out_depth, const torch::Tensor grad_out_alpha,
        const float alpha_min, const float transmittance_min, const int deg
        ){
        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        int max_coeffs = shs.size(1);

        triangle_bvh->gaussian_trace_backward(
            n_elements, (const glm::vec3*)rays_o.data_ptr<float>(), (const glm::vec3*)rays_d.data_ptr<float>(), gs_idxs.data_ptr<int>(), 
            (const glm::vec3*)means3D.data_ptr<float>(), opacity.data_ptr<float>(), (const glm::mat3x3*)SinvR.data_ptr<float>(), (const glm::vec3*)shs.data_ptr<float>(), 
            (const glm::vec3*)colors.data_ptr<float>(), depth.data_ptr<float>(), alpha.data_ptr<float>(), 
            (glm::vec3*)grad_means3D.data_ptr<float>(), grad_opacity.data_ptr<float>(), (glm::mat3x3*)grad_SinvR.data_ptr<float>(), (glm::vec3*)grad_shs.data_ptr<float>(), 
            (const glm::vec3*)grad_out_color.data_ptr<float>(), grad_out_depth.data_ptr<float>(), grad_out_alpha.data_ptr<float>(),
            alpha_min, transmittance_min, deg, max_coeffs, stream);
    }

    std::shared_ptr<TriangleBvhBase> triangle_bvh;
};

GaussianTracer* create_gaussiantracer() {
    return new GaussianTracer{};
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

py::class_<raytracing::GaussianTracer>(m, "GaussianTracer")
    .def("trace_forward", &raytracing::GaussianTracer::trace_forward)
    .def("trace_backward", &raytracing::GaussianTracer::trace_backward)
    .def("build_bvh", &raytracing::GaussianTracer::build_bvh)
    .def("update_bvh", &raytracing::GaussianTracer::update_bvh);

m.def("create_gaussiantracer", &raytracing::create_gaussiantracer);

}