#include "gaussiantrace_backward.h"
#include <optix.h>


namespace raytracing {

extern "C" {
	__constant__ Gaussiantrace_backward::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	float O_final = params.alpha[idx.x];
	if (O_final==0.0f) return;

	glm::vec3 ray_o = params.ray_origins[idx.x];
	glm::vec3 ray_d = params.ray_directions[idx.x];
	glm::vec3 ray_origin;
	glm::vec3 C = glm::vec3(0.0f, 0.0f, 0.0f), C_final = params.colors[idx.x], grad_colors = params.grad_colors[idx.x];
	float D = 0.0f, D_final = params.depths[idx.x], grad_depths = params.grad_depths[idx.x];
	float O = 0.0f, grad_alpha = params.grad_alpha[idx.x];

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
			make_float3(ray_origin.x, ray_origin.y, ray_origin.z),
			make_float3(ray_d.x, ray_d.y, ray_d.z),
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
				glm::vec3 mean3D = params.means3D[gs_idx];
				glm::mat3x3 SinvR = params.SinvR[gs_idx];

				// Compute intersection point
				glm::vec3 ray_o_mean3D = ray_o - mean3D;
				glm::vec3 o_g = SinvR * ray_o_mean3D; 
				glm::vec3 d_g = SinvR * ray_d;
				float dot_dg_dg = max(1e-6f, glm::dot(d_g, d_g));
				float d = -glm::dot(o_g, d_g) / dot_dg_dg;

				glm::vec3 pos = ray_o + d * ray_d;
				glm::vec3 mean_pos = mean3D - pos;
				glm::vec3 p_g = SinvR * mean_pos; 

				float G = __expf(-0.5f * glm::dot(p_g, p_g));
				float alpha = min(0.99f, o * G);
				if (alpha<params.alpha_min) continue;

				glm::vec3 c = computeColorFromSH_forward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs);

				float w = T * alpha;
				C += w * c;
				D += w * d;
				O += w;

				T *= (1 - alpha);

				glm::vec3 dL_dc = grad_colors * w;
				float dL_dd = grad_depths * w;
				float dL_dalpha = (
					glm::dot(grad_colors, T * c - (C_final - C)) +
					grad_depths * (T * d - (D_final - D)) + 
					grad_alpha * (1 - O_final)
				) / max(1e-6f, 1 - alpha);
				computeColorFromSH_backward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs, dL_dc, params.grad_shs + gs_idx * params.max_coeffs);
				float dL_do = dL_dalpha * G;
				float dL_dG = dL_dalpha * o;
				glm::vec3 dL_dpg = -dL_dG * G * p_g;
				glm::mat3x3 dL_dSinvR = glm::outerProduct(dL_dpg, mean_pos);
				// glm::mat3x3 dL_dSinvR = {
				// 	dL_dpg.x * mean_pos.x, dL_dpg.y * mean_pos.x, dL_dpg.z * mean_pos.x, 
				// 	dL_dpg.x * mean_pos.y, dL_dpg.y * mean_pos.y, dL_dpg.z * mean_pos.y, 
				// 	dL_dpg.x * mean_pos.z, dL_dpg.y * mean_pos.z, dL_dpg.z * mean_pos.z
				// };
				
				glm::vec3 dL_dmean_pos = glm::transpose(SinvR) * dL_dpg;
				// vec3 dL_dmean_pos = {
				// 	SinvR[0][0] * dL_dpg.x + SinvR[0][1] * dL_dpg.y + SinvR[0][2] * dL_dpg.z, 
				// 	SinvR[1][0] * dL_dpg.x + SinvR[1][1] * dL_dpg.y + SinvR[1][2] * dL_dpg.z, 
				// 	SinvR[2][0] * dL_dpg.x + SinvR[2][1] * dL_dpg.y + SinvR[2][2] * dL_dpg.z
				// };
				glm::vec3 dL_dmean3D = dL_dmean_pos;

				dL_dd -= glm::dot(dL_dmean_pos, ray_d);

				glm::vec3 dL_dog = -dL_dd / dot_dg_dg * d_g;
				glm::vec3 dL_ddg = -dL_dd / dot_dg_dg * o_g + 2 * dL_dd * glm::dot(o_g, d_g) / max(1e-6f, dot_dg_dg * dot_dg_dg) * d_g;

				dL_dSinvR += glm::outerProduct(dL_dog, ray_o_mean3D);
				// dL_dSinvR += glm::mat3x3{
				// 	dL_dog.x * ray_o_mean3D.x, dL_dog.y * ray_o_mean3D.x, dL_dog.z * ray_o_mean3D.x, 
				// 	dL_dog.x * ray_o_mean3D.y, dL_dog.y * ray_o_mean3D.y, dL_dog.z * ray_o_mean3D.y, 
				// 	dL_dog.x * ray_o_mean3D.z, dL_dog.y * ray_o_mean3D.z, dL_dog.z * ray_o_mean3D.z
				// };

				dL_dmean3D -= glm::transpose(SinvR) * dL_dog;
				//  dL_dmean3D -= vec3{
				// 	SinvR[0][0] * dL_dog.x + SinvR[0][1] * dL_dog.y + SinvR[0][2] * dL_dog.z, 
				// 	SinvR[1][0] * dL_dog.x + SinvR[1][1] * dL_dog.y + SinvR[1][2] * dL_dog.z, 
				// 	SinvR[2][0] * dL_dog.x + SinvR[2][1] * dL_dog.y + SinvR[2][2] * dL_dog.z
				// };

				dL_dSinvR += glm::outerProduct(dL_ddg, ray_d);
				// dL_dSinvR += glm::mat3x3{
				// 	dL_ddg.x * ray_d.x, dL_ddg.y * ray_d.x, dL_ddg.z * ray_d.x, 
				// 	dL_ddg.x * ray_d.y, dL_ddg.y * ray_d.y, dL_ddg.z * ray_d.y, 
				// 	dL_ddg.x * ray_d.z, dL_ddg.y * ray_d.z, dL_ddg.z * ray_d.z
				// };

        		atomic_add((float*)(params.grad_means3D+gs_idx), dL_dmean3D);
				atomicAdd(params.grad_opacity+gs_idx, dL_do);

				float* grad_SinvR = (float*)(params.grad_SinvR + gs_idx);
				for (int j=0; j<9;++j){
					atomicAdd(grad_SinvR+j, dL_dSinvR[j/3][j%3]);
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
            // HitInfo temp = hitArray[i];
            // hitArray[i] = newHit;
            // newHit = temp;
			host_device_swap<HitInfo>(hitArray[i], newHit);
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
