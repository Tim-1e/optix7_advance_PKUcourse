// 目前只实现面光源、球形光源、环境光、点光源
// 待实现物体光源、方向光
#pragma once
#include "optix7.h"
#include "PRD.h"

#define M_PIf 3.14159265359

namespace osc {
    using namespace gdt;

	enum LightType{
		SPHERE, QUAD, ENV, LightTypeNum
	};
	// DIRECTION, HIT_LIGHT_SOURCE, ENV_MISS

	struct LightSample {
		vec3f surfacePos;
		vec3f normal;
		vec3f emission;
		vec3f dir;
		vec2f uv;
		float pdf;
	};

	struct LightParams{
		vec3f position;
		vec3f normal;
		vec3f emission;
		vec3f u;
		vec3f v;
		vec3f direction;
		LightType lightType;
		float area;
		float radius;
		int id;

		inline __both__ LightParams(LightType type, int _id) :
			lightType(type), id(_id) { } 

		inline __both__ void initSphereLight(vec3f& pos, float r, vec3f& _emission) {
			position = pos;
			radius = r;
			emission = _emission;
			area = 4 * M_PIf * radius;
		}

		inline __both__ void initQuadLight(vec3f& pos, vec3f uin,vec3f vin, vec3f& _emission){
			position = pos;
			emission = _emission;
			u = uin;
			v = vin;
			normal = normalize(cross(u, v));
			area = length(u) * length(v);
		}

		inline __both__ void initEnvLight(vec3f& pos, float r ,vec3f& _emission) {
			position = pos;
			radius = r;
			emission = _emission;
		}

		static inline __both__ vec3f UniformSampleSphere(float u1, float u2)
		{
			// 按照柱坐标进行均匀采样
			float z = 1.f - 2.f * u1;
			float r = sqrtf(max(0.f, 1.f - z * z));
			float phi = 2.f * M_PIf * u2;
			float x = r * cosf(phi);
			float y = r * sinf(phi);
			return vec3f(x, y, z);
		}

		inline __both__ void sample(LightSample& sample, PRD& prd) {
			// Add this prefix to try to fit in cuda
			const float r1 = prd.random();
			const float r2 = prd.random();
			switch (lightType)
			{
			case SPHERE:
				sample.surfacePos = position + UniformSampleSphere(r1, r2) * radius;
				sample.normal = normalize(sample.surfacePos - position);
				sample.emission = emission;
				sample.pdf = 1.0f / area;
				break;
			case QUAD:
				sample.surfacePos = position + u * r1 + v * r2;
				sample.normal = normal;
				sample.emission = emission;
				sample.pdf = 1.0 / area;
				break;
			case ENV:
				// treat as a hemisphere.
				sample.dir = -UniformSampleSphere(r1, r2);
				sample.dir[2] = fabs(sample.dir[2]);
				sample.normal = sample.dir;
				sample.surfacePos = position - sample.dir * radius;
				sample.pdf = 1/ (2 * M_PIf);
				sample.emission = emission;;
				break;
			default:
				printf("Unrecognizable light type...\n");
				break;
			}
		}


		inline __both__ float Pdf_Light() {
			vec3f origin = optixGetWorldRayOrigin();
			vec3f dir = optixGetWorldRayDirection();
			switch (lightType)
			{
			case SPHERE:
				break;
			case QUAD: 
				break;
			case ENV:
				break;
			default:
				printf("Unrecognizable light type...\n");
				break;
			}
		}
	};


} // ::osc


