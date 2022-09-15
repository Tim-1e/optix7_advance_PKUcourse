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

using namespace osc;

namespace osc
{

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    #define Maxdepth 30
    #define M_PIf 3.14159265359
    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__shadow()
    {
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        vec2i& dir_hit = *getPRD<vec2i>();
        if (dir_hit.x == sbtData.ID ||dir_hit.y== optixGetPrimitiveIndex()) {
            dir_hit.x = -1;
        }
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        PRD& prd = *getPRD<PRD>();
        if (prd.depth >= Maxdepth) {
            return;
        }
        if (sbtData.emissive_) {
            return;
        }
        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // ------------------------------------------------------------------
        // compute normal, using either shading normal (if avail), or
        // geometry normal (fallback)
        // ------------------------------------------------------------------
        const vec3f &A = sbtData.vertex[index.x];
        const vec3f &B = sbtData.vertex[index.y];
        const vec3f &C = sbtData.vertex[index.z];
        vec3f Ng = cross(B - A, C - A);
        vec3f Ns = (sbtData.normal)
                       ? ((1.f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
                       : Ng;

        // ------------------------------------------------------------------
        // face-forward and normalize normals
        // ------------------------------------------------------------------
        const vec3f rayDir = optixGetWorldRayDirection();
        
        if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
        Ng = normalize(Ng);

        if (dot(Ng, Ns) < 0.f)
            Ns -= 2.f * dot(Ng, Ns) * Ng;
        Ns = normalize(Ns);

        // ------------------------------------------------------------------
        // compute diffuse material color, including diffuse texture, if
        // available
        // ------------------------------------------------------------------
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

        //const float alpha = sbtData.alpha_;
        //const float d = sbtData.d;


        // ------------------------------------------------------------------
        // compute shadow
        // ------------------------------------------------------------------
        const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];

        float diffuse_max = max(max(diffuseColor[0], diffuseColor[1]), diffuseColor[2]);
        
        const float RR = 0.8f;//clamp(diffuse_max,0.3f,0.9f);//俄罗斯轮盘赌
        if (prd.random() > RR) {
            return;
        }


        PRD newprd;//新光线
        uint32_t u0, u1;
        vec3f mont_dir;//光方向
        M_extansion mext;
        mext.diffuseColor = diffuseColor;
        mext.specColor = specColor;//材质属性

        //Pass 将新点加入path
        //prd.path->vertexs[prd.depth].init(surfPos, Ns, (TriangleMeshSBTData*)optixGetSbtDataPointer(), mext,primID);
        //prd.path->length= prd.depth + 1;
        //取出路径顶点
        mont_dir = Sample_adjust(sbtData, Ns, rayDir,prd);
        packPointer(&newprd, u0, u1);
        newprd.random.init(prd.random() * 0x01000000, prd.random() * 0x01000000);
        newprd.depth = prd.depth + 1;
        //newprd.path = prd.path;

        optixTrace(optixLaunchParams.traversable,
            surfPos + 1e-3f * Ng,
            mont_dir,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RADIANCE_RAY_TYPE,            // SBT offset`
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,            // missSBTIndex 
            u0, u1);

        return;
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }

    extern "C" __global__ void __anyhit__shadow()
    { /*! not going to be used */
    }

    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance()
    {
    }

    extern "C" __global__ void __miss__shadow()
    {
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        //const float color_max_avilable = 1.f;
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const auto &camera = optixLaunchParams.camera;

        PRD prd;
        prd.random.init(ix + optixLaunchParams.frame.size.x * iy,
                        optixLaunchParams.frame.frameID);
        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        int numPixelSamples = optixLaunchParams.numPixelSamples;

        vec3f pixelColor = 0.f;
        //vec3f pixelNormal = 0.f;
        //vec3f pixelAlbedo = 0.f;

        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
        {
            vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
                / vec2f(optixLaunchParams.frame.size));


            // generate ray direction
            vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
            BDPTPath eye_path,light_path,connect_path;

            //Begin the eye path build
            prd.depth = 0;
            //prd.path=&eye_path;
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

            ////Begin the light path build
            //int num = optixLaunchParams.Lights_num;
            //LightParams* Lp = &optixLaunchParams.All_Lights[int(num * prd.random())];
            //LightSample Light_point;
            //Lp->sample(Light_point, prd.random);

            //light_path.vertexs[0].pdf = 1 / (2 * M_PIf);
            //light_path.vertexs[0].init(Light_point.position, Light_point.normal,Light_point.meshID);
            //prd.depth = 1;
            //prd.path = &light_path;
            //optixTrace(optixLaunchParams.traversable,
            //    camera.position,
            //    rayDir,
            //    0.f,    // tmin
            //    1e20f,  // tmax
            //    0.0f,   // rayTime
            //    OptixVisibilityMask(255),
            //    OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            //    RADIANCE_RAY_TYPE,            // SBT offset
            //    RAY_TYPE_COUNT,               // SBT stride
            //    RADIANCE_RAY_TYPE,            // missSBTIndex 
            //    u0, u1);

            ////printf("we get there with %d and %d\n", eye_path.length, light_path.length);
            //for (int eye_length = 1; eye_length < eye_path.length; eye_length++)
            //{
            //    for (int light_length = 1; light_length < light_path.length; light_length++)
            //    {
            //        //可见性判断
            //        vec3f eyeLastPoint = eye_path.vertexs[eye_length - 1].position;
            //        vec3f Ng = eye_path.vertexs[eye_length - 1].normal;
            //        vec3f lightLastPoint = light_path.vertexs[light_length - 1].position;
            //        vec3f lightDir = normalize(lightLastPoint - eyeLastPoint);
            //        vec2i dir_hit = vec2i(eye_path.vertexs[eye_length - 1].mat->ID, eye_path.vertexs[eye_length - 1].MeshID);
            //        packPointer(&dir_hit, u0, u1);
            //        optixTrace(optixLaunchParams.traversable,
            //            eyeLastPoint + 1e-3f * Ng,
            //            lightDir,
            //            0.f,    // tmin
            //            1e20f,  // tmax
            //            0.0f,   // rayTime
            //            OptixVisibilityMask(255),
            //            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            //            SHADOW_RAY_TYPE,            // SBT offset
            //            RAY_TYPE_COUNT,               // SBT stride
            //            SHADOW_RAY_TYPE,            // missSBTIndex 
            //            u0, u1);
            //        //printf("we try?\n");
            //        if (dir_hit.x != -1) continue;
            //        //printf("successful meet,with %d light %d eye\n", light_length, eye_length);
            //        Connect_two_path(eye_path,light_path,connect_path,eye_length,light_length);
            //        pixelColor+= evalPath(connect_path);
            //    }
            //}
        }

        vec4f rgba(pixelColor / numPixelSamples, 1.f);
        //vec4f albedo(pixelAlbedo / numPixelSamples, 1.f);
        //vec4f normal(pixelNormal / numPixelSamples, 1.f);
        
        // and write/accumulate to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        if (optixLaunchParams.frame.frameID > 0)
        {
            rgba += float(optixLaunchParams.frame.frameID) * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
            rgba /= (optixLaunchParams.frame.frameID + 1.f);
        }
        optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
        //optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
        //optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
    }

} // ::osc
