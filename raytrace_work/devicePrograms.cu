// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>
#include <vector>
#include "BDPT_function.h"
#include "LaunchParams.h"
#include "config.h"
using namespace osc;

namespace osc
{

    extern "C" __constant__ LaunchParams optixLaunchParams;

    extern "C" __global__ void __closesthit__shadow()
    {
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        vec2i& dir_hit = *getPRD<vec2i>();
        if (dir_hit.x == sbtData.ID && dir_hit.y == optixGetPrimitiveIndex()) {
            dir_hit.x = -1;
        }
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        PRD& prd = *getPRD<PRD>();
        if (prd.depth >= Maxdepth) {
            prd.end = 1;
            return;
        }
        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const vec3f &A = sbtData.vertex[index.x];
        const vec3f &B = sbtData.vertex[index.y];
        const vec3f &C = sbtData.vertex[index.z];
        vec3f Ng = cross(B - A, C - A);
        vec3f Ns = (sbtData.normal)
                       ? ((1.f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
                       : Ng;

        const vec3f rayDir = optixGetWorldRayDirection();
        
        if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
        Ng = normalize(Ng);

        vec3f diffuseColor = sbtData.color;
        if (sbtData.hasTexture && sbtData.texcoord)
        {
            const vec2f tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];

            vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
            diffuseColor *= (vec3f)fromTexture;
        }

        vec3f specColor = 0.0f;
        if (sbtData.hasSpecTexture && sbtData.texcoord)
        {
            const vec2f tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];
            vec4f fromTexture = tex2D<float4>(sbtData.spectexture, tc.x, tc.y);
            specColor = (vec3f)fromTexture;
        }

        const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];

        const float RR = 0.8f;//俄罗斯轮盘赌

        M_extansion mext;
        mext.diffuseColor = diffuseColor;
        mext.specColor = specColor;//材质属性
        
        if (sbtData.emissive_) {
            prd.end = 1;
            if (dot(Ns, -prd.direction) < 0) {
                return;
            }
            prd.path->vertexs[prd.depth].init(surfPos, Ng, sbtData.ID, mext, primID, optixLaunchParams.matHeader);
            int mapLight = optixLaunchParams.idToLightMap[sbtData.ID];
            if (mapLight == -1) printf("error,wrong light\n");
            LightParams* Lp = &optixLaunchParams.All_Lights[mapLight];
            LightSample Light_point;
            Lp->sample(Light_point, prd.random);
            prd.path->vertexs[prd.depth].pdf = Light_point.pdf;
            prd.path->length = prd.depth + 1;
            prd.lightColor = evalPath(*prd.path,0);
            prd.path->length = prd.depth -1;
            //prd.lightColor = singlePathContriCompute(*prd.path);
            prd.TouchtheLight = 1;
            return;
        }

        vec3f mont_dir;//光方向

        //Pass 将新点加入path
        prd.path->vertexs[prd.depth].init(surfPos, Ng, sbtData.ID, mext,primID,optixLaunchParams.matHeader);
        prd.path->length = prd.depth + 1;
        //取出路径顶点
        mont_dir = Sample_adjust(sbtData, Ng, rayDir,prd);
        prd.depth = prd.depth + 1;
        prd.normal = Ng;
        prd.sourcePos = surfPos;
        prd.direction = mont_dir;

        if (prd.random() > RR) {
            prd.end = 1;
            return;
        }

        return;
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }

    extern "C" __global__ void __anyhit__shadow()
    { /*! not going to be used */
    }

    extern "C" __global__ void __miss__radiance()
    {
        PRD& prd = *getPRD<PRD>();
        prd.end = 1;
        return;
    }

    extern "C" __global__ void __miss__shadow()
    {
    }

    extern "C" __global__ void __raygen__lightGen() {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const uint32_t fbIndex = ix + iy * LightRayGenerateNum;
            PRD prd;
            prd.random.init(ix + optixLaunchParams.frame.size.x * iy,
                optixLaunchParams.frame.frameID);
            uint32_t u0, u1;
            packPointer(&prd, u0, u1);

            vec3f rayDir;
            BDPTPath light_path;
            int* lightPathNum;
            light_path.vertexs = optixLaunchParams.lightPath + fbIndex * Maxdepth;
            lightPathNum = optixLaunchParams.lightPathNum + fbIndex;

            //Begin the light path build
            int num = optixLaunchParams.Lights_num;
            LightParams* Lp = &optixLaunchParams.All_Lights[int(num * prd.random())];
            LightSample Light_point;
            Lp->sample(Light_point, prd.random);

            light_path.length = 1;
            light_path.vertexs[0].pdf = Light_point.pdf;
            TriangleMeshSBTData mat;
            mat.ID = Light_point.id;
            mat.emission = Lp->emission;
            M_extansion ext;
            light_path.vertexs[0].init(Light_point.position, Light_point.normal, mat.ID, ext, Light_point.meshID,optixLaunchParams.matHeader);

            //prd.depth = 1;
            //prd.path = &light_path;
            //prd.end = 0;
            //rayDir = Lp->UniformSampleDir(Light_point.position, Light_point.normal, prd.random);
            //optixTrace(optixLaunchParams.traversable,
            //    Light_point.position,
            //    rayDir,
            //    1e-5f,    // tmin
            //    1e20f,  // tmax
            //    0.0f,   // rayTime
            //    OptixVisibilityMask(255),
            //    OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            //    RADIANCE_RAY_TYPE,            // SBT offset
            //    RAY_TYPE_COUNT,               // SBT stride
            //    RADIANCE_RAY_TYPE,            // missSBTIndex 
            //    u0, u1);
            //while (!prd.end)
            //{
            //    optixTrace(optixLaunchParams.traversable,
            //        prd.sourcePos ,
            //        prd.direction,
            //        1e-5f,    // tmin
            //        1e20f,  // tmax
            //        0.0f,   // rayTime
            //        OptixVisibilityMask(255),
            //        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            //        RADIANCE_RAY_TYPE,            // SBT offset`
            //        RAY_TYPE_COUNT,               // SBT stride
            //        RADIANCE_RAY_TYPE,            // missSBTIndex 
            //        u0, u1);
            //}
            *lightPathNum = light_path.length;
    }

    extern "C" __global__ void __raygen__renderFrame()
    {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const auto& camera = optixLaunchParams.camera;

        int numPixelSamples = optixLaunchParams.numPixelSamples;
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

        vec3f pixelColor = 0.f;

        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
        {
            PRD prd;
            prd.random.init(ix + optixLaunchParams.frame.size.x * iy+ sampleID,
                optixLaunchParams.frame.frameID);
            uint32_t u0, u1;
            packPointer(&prd, u0, u1);

            vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
                / vec2f(optixLaunchParams.frame.size));

            //calcu all num
            int LightVertexNum = 0;
            for (int i = 0; i < LightRayGenerateNum * LightRayGenerateNum; i++)
            {
                LightVertexNum += optixLaunchParams.lightPathNum[i];
            }

            // generate ray direction
            vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
            BDPTPath eye_path, light_path, connect_path;
            eye_path.vertexs = optixLaunchParams.eyePath + fbIndex * Maxdepth;
            connect_path.vertexs = optixLaunchParams.connectPath + fbIndex * Maxdepth * 2;
            int light_choose = int(prd.random() * LightRayGenerateNum * LightRayGenerateNum);
            light_path.vertexs = optixLaunchParams.lightPath+ light_choose * Maxdepth;
            light_path.length = *(optixLaunchParams.lightPathNum + light_choose);
            

            //Begin the eye path build
            eye_path.vertexs[0].init(camera.position);
            eye_path.vertexs[0].normal = camera.direction;
            eye_path.length = 1;
            prd.depth = 1;
            prd.path = &eye_path;
            prd.end = 0;
            prd.TouchtheLight = 0;
            prd.direction = rayDir;
            optixTrace(optixLaunchParams.traversable,
                camera.position,
                rayDir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                RADIANCE_RAY_TYPE,            // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                RADIANCE_RAY_TYPE,            // missSBTIndex 
                u0, u1);
            while (!prd.end)
            {
                optixTrace(optixLaunchParams.traversable,
                    prd.sourcePos ,
                    prd.direction,
                    1e-5f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    RADIANCE_RAY_TYPE,            // SBT offset`
                    RAY_TYPE_COUNT,               // SBT stride
                    RADIANCE_RAY_TYPE,            // missSBTIndex 
                    u0, u1);
            }
            if (prd.TouchtheLight) {
                pixelColor += prd.lightColor;
            }

            //float pdf = float(LightRayGenerateNum * LightRayGenerateNum) / (LightVertexNum);
            //float pdf = 1.f;
            for (int eye_length = 2; eye_length <= eye_path.length; eye_length++)
            {
                int light_choose = int(prd.random()* LightRayGenerateNum * LightRayGenerateNum);
                int length = optixLaunchParams.lightPathNum[light_choose];
                int length_choose = int(prd.random() * length)+1;
                vec3f turnColor;

                light_path.vertexs = optixLaunchParams.lightPath + light_choose * Maxdepth;
                light_path.length = length_choose;
                int light_length = length_choose;
                //可见性判断
                vec3f eyeLastPoint = eye_path.vertexs[eye_length - 1].position;
                vec3f lightLastPoint = light_path.vertexs[light_length - 1].position;
                vec3f lightDir = normalize(lightLastPoint - eyeLastPoint);
                vec2i dir_hit = vec2i(light_path.vertexs[light_length - 1].mat->ID, light_path.vertexs[light_length - 1].MeshID);
                packPointer(&dir_hit, u0, u1);
                optixTrace(optixLaunchParams.traversable,
                    eyeLastPoint ,
                    lightDir,
                    1e-5f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    SHADOW_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    SHADOW_RAY_TYPE,            // missSBTIndex 
                    u0, u1);
                if (dir_hit.x == -1)
                {
                    Connect_two_path(eye_path, light_path, connect_path, eye_length, light_length);
                    turnColor = evalPath(connect_path,1) * length;
                    pixelColor += turnColor;
                }
            }
        }
        vec4f rgba(pixelColor / numPixelSamples, 1.f);
         //rgba=clamp(rgba, vec4f(0.f), vec4f(100.f));
        
        if (optixLaunchParams.frame.frameID > 0)
        {
            rgba += float(optixLaunchParams.frame.frameID) * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
            rgba /= (optixLaunchParams.frame.frameID + 1.f);
        }
        optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
    }

} // ::osc
