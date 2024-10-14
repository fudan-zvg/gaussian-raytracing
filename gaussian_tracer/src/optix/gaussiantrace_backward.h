#pragma once

#include <raytracing/triangle.cuh>

#include <optix.h>

namespace raytracing {

struct Gaussiantrace_backward {
	struct Params {
		const vec3* ray_origins;
		const vec3* ray_directions;
		const int* gs_idxs;
		const vec3* means3D;
		const float* opacity;
		const mat3x3* SinvR;
		const vec3* shs;
		const vec3* colors;
		const float* depths;
		const float* alpha;
		vec3* grad_means3D;
		float* grad_opacity;
		mat3x3* grad_SinvR;
		vec3* grad_shs;
		const vec3* grad_colors;
		const float* grad_depths;
		const float* grad_alpha;
		vec3* colors2;
		float* grad_w;
		float alpha_min;
		float transmittance_min;
		int deg;
		int max_coeffs;
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};

}
