#pragma once
#include "optix7.h"
#include "gdt/random/random.h"

#define M_PIf 3.14159265359

namespace osc {
    using namespace gdt;
	typedef LCG<16> Random;
	enum LightType{
		QUAD,TRIANGLE,LightTypeNum
	};
	// DIRECTION, HIT_LIGHT_SOURCE, ENV_MISS

	struct LightSample {
		vec3f position;
		vec3f normal;
		vec3f emission;
		vec3f origin;
		vec3f u;
		vec3f v;
		float pdf;
		float pdf_area;
		LightType lightType;

		inline __both__ float Pdf_Light(vec3f ray_origin, vec3f dir) {
			vec3f cast_pos, pos_to_cast;
			float pdu, pdv, udu, udv, vdv;
			float u_cast, v_cast, u_length, v_length;
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

<<<<<<< HEAD
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

		inline __both__ void sample(LightSample& sample, Random& rdm) {
			// Add this prefix to try to fit in cuda
			const float r1 = rdm();
			const float r2 = rdm();
			vec3f A, B, C;
=======
		inline __both__ void sample(LightSample& sample, PRD& prd) {
			// Add this prefix to try to fit in cuda
			const float r1 = prd.random();
			const float r2 = prd.random();
			vec3f A, B, C, uu, vv;
>>>>>>> 5d50a58dd95006dc4f2ecf6298c9748f0bcd7528
			int chose;
			chose = int(prd.random() * num);
			A = vertex[index[chose].x];
			B = vertex[index[chose].y];
			C = vertex[index[chose].z];
			uu = B - A;
			vv = C - A;
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
		};

<<<<<<< HEAD
			case TRIANGLE:
				chose = int(rdm() * num);
				A= vertex[index[chose].x];
				B=  vertex[index[chose].y];
				C = vertex[index[chose].z];
				normal = normalize(cross(A - B, A - C));
				position = A;
				u = B - A;
				v = C - A;
				area = length(cross(u, v)) / 2;
				sample.surfacePos = (1 - sqrt(r1)) * A + sqrt(r1) * (1 - r2) * B + sqrt(r1) * r2 * C;
				sample.normal = normal;
				sample.emission = emission;
				sample.pdf = 1.0 / (area* num);
				break;
			default:
				printf("Unrecognizable light type...\n");
				break;
			}
		}


		inline __both__ float Pdf_Light(vec3f origin,vec3f dir) {
			vec3f cast_pos;
			float u_cast, v_cast, u_length, v_length;
			float mydist;
			vec3f z;
			float zdu, zdv, udu, vdv, udv, x, y;
			switch (lightType)
			{
			case QUAD: 
				//Plane insert
				if (dot(normal,dir) == 0) {//判断有无交点
					if (dot(origin-position,normal) == 0) return 1;//在平面上,返回0
					else return 0;//非平面,无穷
				}

				mydist=-dot(origin - position, normal) / dot(dir, normal);
				if (mydist < 0) return 0;
				if (dot(dir, normal) > 0) return 0;
				cast_pos = mydist * dir + origin;
				u_cast = dot(cast_pos - position, normalize(u));
				v_cast = dot(cast_pos - position, normalize(v));
				if (u_cast >= 0 && u_cast <= length(u) && v_cast >= 0 && v_cast <= length(v)) {
					return 1.0 / area* mydist* mydist/dot(normal,-dir);
				}
				else
					return 0;
			case TRIANGLE:
				//Plane insert
				if (dot(normal, dir) == 0) {//判断有无交点
					if (dot(origin - position, normal) == 0) return 1;//在平面上,返回0
					else return 0;//非平面,无穷
				}
				mydist = -dot(origin - position, normal) / dot(dir, normal);
				if (mydist < 0) return 0;
				if (dot(dir, normal) > 0) return 0;
				cast_pos = mydist * dir + origin;
				z = cast_pos - position;
				zdu = dot(z, u);
				zdv = dot(z, v);
				udu = dot(u, u);
				vdv = dot(v, v);
				udv = dot(u, v);
				y = (zdu * udv - zdv * udu) / (udv * udv - udu * udu);
				x = (zdv * udv - zdu * vdv) / (udv * udv - vdv * vdv);
				if (x >= 0 && y >= 0 && x + y <= 1)
					return 1.0 / (area * num) * mydist * mydist / dot(normal, -dir);
				else
					return 0;
				//u_cast = dot(cast_pos - position, normalize(u));
				//v_cast = dot(cast_pos - position, normalize(v));

				//if (u_cast >= 0 && v_cast >= 0 && v_cast*u_length+u_cast*v_length<=u_length*v_length) {
				//	return 1.0 / (area * num) * mydist * mydist / dot(normal, -dir);
				//}
				//else
				//	return 0;
			default:
				printf("Unrecognizable light type...\n");
				break;
			}
		}
=======
>>>>>>> 5d50a58dd95006dc4f2ecf6298c9748f0bcd7528
	};
} // ::osc