#include <raytracing/common.h>
#define MAX_BUFFER_SIZE 16
#define T_SCENE_MAX 100.0f

namespace raytracing{

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

struct HitInfo {
    float t;
    int primIdx;
};


__device__ vec3 computeColorFromSH_forward(int deg, vec3 dir, const vec3* sh)
{
	vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result = result + 0.5f;
	return max(result, 0.0f);
}

__device__ void computeColorFromSH_backward(const int deg, const vec3 dir, const vec3* sh, const vec3 dL_dcolor, vec3* dL_dsh)
{
    atomic_add((float*)dL_dsh, SH_C0 * dL_dcolor);
	if (deg > 0)
	{
        
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;

		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
        atomic_add((float*)(dL_dsh+1), dRGBdsh1 * dL_dcolor);
        atomic_add((float*)(dL_dsh+2), dRGBdsh2 * dL_dcolor);
        atomic_add((float*)(dL_dsh+3), dRGBdsh3 * dL_dcolor);

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
            atomic_add((float*)(dL_dsh+4), dRGBdsh4 * dL_dcolor);
            atomic_add((float*)(dL_dsh+5), dRGBdsh5 * dL_dcolor);
            atomic_add((float*)(dL_dsh+6), dRGBdsh6 * dL_dcolor);
            atomic_add((float*)(dL_dsh+7), dRGBdsh7 * dL_dcolor);
            atomic_add((float*)(dL_dsh+8), dRGBdsh8 * dL_dcolor);

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                atomic_add((float*)(dL_dsh+9), dRGBdsh9 * dL_dcolor);
                atomic_add((float*)(dL_dsh+10), dRGBdsh10 * dL_dcolor);
                atomic_add((float*)(dL_dsh+11), dRGBdsh11 * dL_dcolor);
                atomic_add((float*)(dL_dsh+12), dRGBdsh12 * dL_dcolor);
                atomic_add((float*)(dL_dsh+13), dRGBdsh13 * dL_dcolor);
                atomic_add((float*)(dL_dsh+14), dRGBdsh14 * dL_dcolor);
                atomic_add((float*)(dL_dsh+15), dRGBdsh15 * dL_dcolor);
			}
		}
	}
}

}
