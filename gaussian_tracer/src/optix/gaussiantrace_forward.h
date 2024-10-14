#pragma once

#include <raytracing/triangle.cuh>

#include <optix.h>

namespace raytracing {

struct Gaussiantrace_forward {
	struct Params {
		const vec3* ray_origins;
		const vec3* ray_directions;
		const int* gs_idxs;
		const vec3* means3D;
		const float* opacity;
		const mat3x3* SinvR;
		const vec3* shs;
		vec3* colors;
		float* depths;
		float* alpha;
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
