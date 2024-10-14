#pragma once

#include <raytracing/common.h>

namespace raytracing {

// Triangle data structure
struct Triangle {

    __host__ __device__ vec3 normal() const {
		return normalize(cross(b - a, c - a));
    }

    __host__ __device__ float ray_intersect(const vec3 &ro, const vec3 &rd, vec3& n) const { // based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
        vec3 v1v0 = b - a;
        vec3 v2v0 = c - a;
        vec3 rov0 = ro - a;
		n = cross(v1v0, v2v0);
		vec3 q = cross(rov0, rd);
        float d = 1.0f / dot(rd, n);
		float u = d * -dot(q, v2v0);
		float v = d *  dot(q, v1v0);
		float t = d * -dot(n, rov0);
        if( u<0.0f || u>1.0f || v<0.0f || (u+v)>1.0f || t<0.0f) t = 1e6f;
        return t; // vec3( t, u, v );
    }

    __host__ __device__ float ray_intersect(const vec3 &ro, const vec3 &rd) const {
        vec3 n;
        return ray_intersect(ro, rd, n);
    }

    __host__ __device__ vec3 centroid() const {
        return (a + b + c) / 3.0f;
    }

    __host__ __device__ float centroid(int axis) const {
        return (a[axis] + b[axis] + c[axis]) / 3;
    }

    __host__ __device__ void get_vertices(vec3 v[3]) const {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }

    vec3 a, b, c;
};


inline std::ostream& operator<<(std::ostream& os, const Triangle& triangle) {
    os << "[";
    os << "a=[" << triangle.a.x << "," << triangle.a.y << "," << triangle.a.z << "], ";
    os << "b=[" << triangle.b.x << "," << triangle.b.y << "," << triangle.b.z << "], ";
    os << "c=[" << triangle.c.x << "," << triangle.c.y << "," << triangle.c.z << "]";
    os << "]";
    return os;
}


}