#pragma once

#include "auxiliary.h"
#include <optix.h>

namespace raytracing {

struct Gaussiantrace_backward {
	struct Params {
		const glm::vec3* ray_origins;
		const glm::vec3* ray_directions;
		const int* gs_idxs;
		const glm::vec3* means3D;
		const float* opacity;
		const glm::mat3x3* SinvR;
		const glm::vec3* shs;
		const glm::vec3* colors;
		const float* depths;
		const float* alpha;
		glm::vec3* grad_means3D;
		float* grad_opacity;
		glm::mat3x3* grad_SinvR;
		glm::vec3* grad_shs;
		const glm::vec3* grad_colors;
		const float* grad_depths;
		const float* grad_alpha;
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
