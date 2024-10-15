#include <raytracing/common.h>

#include <optix.h>

#include "gaussiantrace_backward.h"
#include "auxiliary.h"

namespace raytracing {

extern "C" {
	__constant__ Gaussiantrace_backward::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	vec3 ray_o = params.ray_origins[idx.x];
	vec3 ray_d = params.ray_directions[idx.x];
	vec3 ray_origin;

	vec3 C = vec3(0.0f, 0.0f, 0.0f), C_final = params.colors[idx.x], grad_colors = params.grad_colors[idx.x];
	float D = 0.0f, D_final = params.depths[idx.x], grad_depths = params.grad_depths[idx.x];
	float O = 0.0f, O_final = params.alpha[idx.x], grad_alpha = params.grad_alpha[idx.x];

	float T = 1.0f, t_start = 0.0f, t_curr = 0.0f;

	HitInfo hitArray[MAX_BUFFER_SIZE];
	unsigned int hitArrayPtr0 = (unsigned int)((uintptr_t)(&hitArray) & 0xFFFFFFFF);
    unsigned int hitArrayPtr1 = (unsigned int)(((uintptr_t)(&hitArray) >> 32) & 0xFFFFFFFF);

	int k=0;
	while ((t_start < T_SCENE_MAX) && (T > params.transmittance_min)){
		k++;
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

				// Compute intersection point
				vec3 ray_o_mean3D = ray_o - mean3D;
				vec3 o_g = SinvR * ray_o_mean3D; 
				vec3 d_g = SinvR * ray_d;
				float dot_dg_dg = max(1e-6f, dot(d_g, d_g));
				float d = -dot(o_g, d_g) / dot_dg_dg;

				vec3 pos = ray_o + d * ray_d;
				vec3 mean_pos = mean3D - pos;
				vec3 p_g = SinvR * mean_pos; 

				float G = __expf(-0.5f * dot(p_g, p_g));
				float alpha = min(0.99f, o * G);
				if (alpha<params.alpha_min) continue;

				vec3 c = computeColorFromSH_forward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs);

				float w = T * alpha;
				C += w * c;
				D += w * d;
				O += w;

				T *= (1 - alpha);

				vec3 dL_dc = grad_colors * w;
				float dL_dd = grad_depths * w;
				float dL_dalpha = (
					dot(grad_colors, T * c - (C_final - C)) +
					grad_depths * (T * d - (D_final - D)) + 
					grad_alpha * (1 - O_final)
				) / max(1e-6f, 1 - alpha);
				computeColorFromSH_backward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs, dL_dc, params.grad_shs + gs_idx * params.max_coeffs);
				float dL_do = dL_dalpha * G;
				float dL_dG = dL_dalpha * o;
				vec3 dL_dpg = -dL_dG * G * p_g;
				mat3x3 dL_dSinvR = {
					dL_dpg.x * mean_pos.x, dL_dpg.y * mean_pos.x, dL_dpg.z * mean_pos.x, 
					dL_dpg.x * mean_pos.y, dL_dpg.y * mean_pos.y, dL_dpg.z * mean_pos.y, 
					dL_dpg.x * mean_pos.z, dL_dpg.y * mean_pos.z, dL_dpg.z * mean_pos.z
				};

				
				vec3 dL_dmean_pos = transpose(SinvR) * dL_dpg;
				// vec3 dL_dmean_pos = {
				// 	SinvR[0][0] * dL_dpg.x + SinvR[0][1] * dL_dpg.y + SinvR[0][2] * dL_dpg.z, 
				// 	SinvR[1][0] * dL_dpg.x + SinvR[1][1] * dL_dpg.y + SinvR[1][2] * dL_dpg.z, 
				// 	SinvR[2][0] * dL_dpg.x + SinvR[2][1] * dL_dpg.y + SinvR[2][2] * dL_dpg.z
				// };
				vec3 dL_dmean3D = dL_dmean_pos;

				dL_dd -= dot(dL_dmean_pos, ray_d);

				vec3 dL_dog = -dL_dd / dot_dg_dg * d_g;
				vec3 dL_ddg = -dL_dd / dot_dg_dg * o_g + 2 * dL_dd * dot(o_g, d_g) / max(1e-6f, dot_dg_dg * dot_dg_dg) * d_g;

				dL_dSinvR += mat3x3{
					dL_dog.x * ray_o_mean3D.x, dL_dog.y * ray_o_mean3D.x, dL_dog.z * ray_o_mean3D.x, 
					dL_dog.x * ray_o_mean3D.y, dL_dog.y * ray_o_mean3D.y, dL_dog.z * ray_o_mean3D.y, 
					dL_dog.x * ray_o_mean3D.z, dL_dog.y * ray_o_mean3D.z, dL_dog.z * ray_o_mean3D.z
				};

				dL_dmean3D -= transpose(SinvR) * dL_dog;
				//  dL_dmean3D -= vec3{
				// 	SinvR[0][0] * dL_dog.x + SinvR[0][1] * dL_dog.y + SinvR[0][2] * dL_dog.z, 
				// 	SinvR[1][0] * dL_dog.x + SinvR[1][1] * dL_dog.y + SinvR[1][2] * dL_dog.z, 
				// 	SinvR[2][0] * dL_dog.x + SinvR[2][1] * dL_dog.y + SinvR[2][2] * dL_dog.z
				// };

				dL_dSinvR += mat3x3{
					dL_ddg.x * ray_d.x, dL_ddg.y * ray_d.x, dL_ddg.z * ray_d.x, 
					dL_ddg.x * ray_d.y, dL_ddg.y * ray_d.y, dL_ddg.z * ray_d.y, 
					dL_ddg.x * ray_d.z, dL_ddg.y * ray_d.z, dL_ddg.z * ray_d.z
				};

        		atomic_add((float*)(params.grad_means3D+gs_idx), dL_dmean3D);
				atomicAdd(params.grad_opacity+gs_idx, dL_do);

				float* grad_SinvR = (float*)(params.grad_SinvR + gs_idx);
				for (int j=0; j<9;++j){
					atomicAdd(grad_SinvR+j, dL_dSinvR.d[j]);
				}

				if (T < params.transmittance_min){
					break;
				}
			}
		}
		if (t_curr==0.0f) break;
		t_start += t_curr;
		if (k>1000){printf("t_curr:%f\n",t_curr);}
	}

	params.colors2[idx.x] = C;
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
