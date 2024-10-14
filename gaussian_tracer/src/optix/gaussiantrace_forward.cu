#include <raytracing/common.h>

#include <optix.h>

#include "gaussiantrace_forward.h"
#include "auxiliary.h"

namespace raytracing {

extern "C" {
	__constant__ Gaussiantrace_forward::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	vec3 ray_o = params.ray_origins[idx.x];
	vec3 ray_d = params.ray_directions[idx.x];
	vec3 ray_origin;

	vec3 C = vec3(0.0f, 0.0f, 0.0f);
	float D = 0.0f, O = 0.0f, T = 1.0f, t_start = 0.0f, t_curr = 0.0f;

	HitInfo hitArray[MAX_BUFFER_SIZE];
	unsigned int hitArrayPtr0 = (unsigned int)((uintptr_t)(&hitArray) & 0xFFFFFFFF);
    unsigned int hitArrayPtr1 = (unsigned int)(((uintptr_t)(&hitArray) >> 32) & 0xFFFFFFFF);

	while ((t_start < T_SCENE_MAX) && (T > params.transmittance_min)){
		ray_origin = ray_o + t_start * ray_d;
		
		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			hitArray[i].t = 1e16f;
			hitArray[i].primIdx = -1;
		}
		optixTrace(
			params.handle,
			to_float3(ray_origin),
			to_float3(ray_d),
			0.0f,                // Min intersection distance
			T_SCENE_MAX,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
			0,                   // SBT offset
			1,                   // SBT stride
			0,                   // missSBTIndex
			hitArrayPtr0,
			hitArrayPtr1
		);

		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			int primIdx = hitArray[i].primIdx;

			if (primIdx == -1) {
				t_curr = T_SCENE_MAX;
				break;
			}
			else{
				t_curr = hitArray[i].t;
				int gs_idx = params.gs_idxs[primIdx];

				float o = params.opacity[gs_idx];
				vec3 mean3D = params.means3D[gs_idx];
				mat3x3 SinvR = params.SinvR[gs_idx];

				vec3 o_g = SinvR * (ray_o - mean3D); 
				vec3 d_g = SinvR * ray_d;
				float d = -dot(o_g, d_g) / dot(d_g, d_g);

				vec3 pos = ray_o + d * ray_d;
				vec3 p_g = SinvR * (mean3D - pos); 
				float alpha = o * __expf(-0.5f * dot(p_g, p_g));

				if (alpha<params.alpha_min) continue;

				vec3 c = computeColorFromSH_forward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs);

				float w = T * alpha;
				C += w * c;
				D += w * d;
				O += w;

				T *= (1 - alpha);

				if (T < params.transmittance_min){
					break;
				}
			}
			

		}
		if (t_curr==0.0f) break;
		t_start += t_curr;
	}
	
	params.colors[idx.x] = C;
	params.depths[idx.x] = D;
	params.alpha[idx.x] = O;
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ch() {
}

extern "C" __global__ void __anyhit__ah() {
	unsigned int hitArrayPtr0 = optixGetPayload_0();
    unsigned int hitArrayPtr1 = optixGetPayload_1();

    HitInfo* hitArray = (HitInfo*)((uintptr_t)hitArrayPtr0 | ((uintptr_t)hitArrayPtr1 << 32));

	float THit = optixGetRayTmax();
    int i_prim = optixGetPrimitiveIndex();
	HitInfo newHit = {THit, i_prim};

	for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
        if (hitArray[i].t > newHit.t) {
            HitInfo temp = hitArray[i];
            hitArray[i] = newHit;
            newHit = temp;
        }
    }

	// int i = MAX_BUFFER_SIZE - 1;
	// while (i > 0 && hitArray[i - 1].t > newHit.t) {
	// 	hitArray[i] = hitArray[i - 1];
	// 	--i;
	// }
	// hitArray[i] = newHit;

	if (THit < hitArray[MAX_BUFFER_SIZE - 1].t) {
        optixIgnoreIntersection(); 
    }

}

}
