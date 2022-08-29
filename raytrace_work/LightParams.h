// 目前只实现面光源、球形光源、环境光、点光源
// 待实现物体光源、方向光
#pragma once
#include "optix7.h"
#include "PRD.h"

#define M_PIf 3.14159265359

namespace osc {
    using namespace gdt;

	enum LightType{
		QUAD,TRIANGLE,LightTypeNum
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
		int id;//store the num of the faces
		vec3f* vertex;//the arrary of the face
		vec3i* index;//the arrary of the face index
		inline __both__ LightParams(LightType type, int _id) :
			lightType(type), id(_id) { } 

		inline __both__ void initQuadLight(vec3f& pos, vec3f uin,vec3f vin, vec3f& _emission){
			position = pos;
			emission = _emission;
			u = uin;
			v = vin;
			normal = normalize(cross(u, v));
			area = length(u) * length(v);
		}

		inline __both__ void initTriangleLight(vec3f* mesh_vertex,vec3i* mesh_index)
		{
			vertex = mesh_vertex;
			index = mesh_index;
		}

		inline __both__ void sample(LightSample& sample, PRD& prd) {
			// Add this prefix to try to fit in cuda
			const float r1 = prd.random();
			const float r2 = prd.random();
			switch (lightType)
			{
			case QUAD:
				sample.surfacePos = position + u * r1 + v * r2;
				sample.normal = normal;
				sample.emission = emission;
				sample.pdf = 1.0 / area;
				break;
			default:
				printf("Unrecognizable light type...\n");
				break;
			}
		}


		inline __both__ float Pdf_Light(vec3f origin,vec3f dir) {
			vec3f cast_pos;
			switch (lightType)
			{
			case QUAD: 
				//Plane insert
				if (dot(normal,dir) == 0) {//判断有无交点
					if (dot(origin-position,normal) == 0) return 1;//在平面上,返回0
					else return 0;//非平面,无穷
				}
				float mydist;
				mydist=-dot(origin - position, normal) / dot(dir, normal);
				if (mydist < 0) return 0;
				if (dot(dir, normal) > 0) return 0;
				cast_pos = mydist * dir + origin;
				float u_cast, v_cast;
				u_cast = dot(cast_pos - position, normalize(u));
				v_cast = dot(cast_pos - position, normalize(v));
				if (u_cast >= 0 && u_cast <= length(u) && v_cast >= 0 && v_cast <= length(v)) {
					return 1.0 / area* mydist* mydist/dot(normal,-dir);
				}
				else
					return 0;
			default:
				printf("Unrecognizable light type...\n");
				break;
			}
		}
	};


} // ::osc


