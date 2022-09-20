#pragma once
#include "optix7.h"
#include "gdt/random/random.h"

#define M_PIf 3.14159265359

namespace osc {
	using namespace gdt;
	typedef LCG<16> Random;
	enum LightType {
		QUAD, TRIANGLE, LightTypeNum
	};
	// DIRECTION, HIT_LIGHT_SOURCE, ENV_MISS

	struct LightSample {
		vec3f position;
		vec3f normal;
		vec3f emission;
		vec3f origin;
		vec3f u;
		vec3f v;
		int meshID;
		int id;
		float pdf;
		float pdf_area;
		LightType lightType;

		inline __both__ float Pdf_Light(vec3f ray_origin, vec3f dir) {
			vec3f cast_pos, pos_to_cast;
			float pdu, pdv, udu, udv, vdv;
			float u_cast, v_cast;
			float mydist;
			//Plane insert
			if (dot(normal, dir) == 0) {//判断有无交点
				if (dot(ray_origin - origin, normal) == 0) return 1;//在平面上,返回0
				else return 0;//非平面,无穷
			}
			mydist = -dot(ray_origin - origin, normal) / dot(dir, normal);
			if (mydist < 0) return 0;
			if (dot(dir, normal) > 0) return 0;
			cast_pos = mydist * dir + ray_origin;
			pos_to_cast = cast_pos - origin;
			pdu = dot(pos_to_cast, u);
			pdv = dot(pos_to_cast, v);
			udu = dot(u, u); vdv = dot(v, v); udv = dot(u, v);
			u_cast = (pdu * vdv - pdv * udv) / (udu * vdv - udv * udv);
			v_cast = (pdv * udu - pdu * udv) / (udu * vdv - udv * udv);
			if (u_cast >= 0 && v_cast >= 0 && v_cast + u_cast <= 1) {
				return 1.0 / pdf_area * mydist * mydist / dot(normal, -dir);
			}
			else
				return 0;
		}
	};

	struct LightParams {
		vec3f emission;
		//above are read only
		vec3f direction;
		LightType lightType;
		int id;
		int num;//store the num of the faces
		vec3f* vertex;//the arrary of the face
		vec3i* index;//the arrary of the face index
		inline __both__ LightParams(LightType type, int _id) :
			lightType(type), id(_id) { }

		inline __both__ void initTriangleLight(vec3f* mesh_vertex, vec3i* mesh_index, vec3f& _emission, int num_)
		{
			vertex = mesh_vertex;
			index = mesh_index;
			emission = _emission;
			num = num_;
		}

		inline __both__ void sample(LightSample& sample, Random& rdm) {
			const float r1 = rdm();
			const float r2 = rdm();
			vec3f A, B, C, uu, vv;
			int chose;
			chose = int(rdm() * num);
			A = vertex[index[chose].x];
			B = vertex[index[chose].y];
			C = vertex[index[chose].z];
			uu = B - A;
			vv = C - A;
			sample.meshID = chose;
			sample.origin = A;
			sample.pdf_area = length(cross(uu, vv)) / 2 * num;
			sample.position = (1 - sqrt(r1)) * A + sqrt(r1) * (1 - r2) * B + sqrt(r1) * r2 * C;
			sample.normal = normalize(cross(uu, vv));
			sample.emission = emission;
			sample.pdf = 1.0 / sample.pdf_area;
			sample.origin = A;
			sample.u = uu;
			sample.v = vv;
			sample.lightType = lightType;
			sample.id = id;
		}

		static inline __both__ vec3f UniformSampleDir(vec3f position, vec3f normal, Random& rdm)
		{
			const float r1 = rdm();
			const float r2 = rdm();
			vec3f uu, vv;
			if (normal == vec3f(0.f, 0.f, 1.f) || normal == vec3f(0.f, 0.f, -1.f))
				uu = vec3f(0.f, 1.f, 0.f);		
			else
				uu = normalize(cross(normal, vec3f(0.f, 0.f, 1.f)));
			vv = normalize(cross(uu, normal));
			// 半球空间均匀采样
			float z = r1;
			float r = sqrtf(1.f - z * z);
			float phi = 2.f * M_PIf * r2;
			float x = r * cosf(phi);
			float y = r * sinf(phi);
			return normalize(x*uu+y*vv+z*normal);
		}
	};
} // ::osc