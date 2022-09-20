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
#include "config.h"
#include "LaunchParams.h"
#include "tool_function.h"

using namespace osc;

namespace osc
{

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;
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
        int& light_hit = *getPRD<int>();

        if (light_hit == sbtData.ID) {
            light_hit = -1;
        }
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        PRD& prd = *getPRD<PRD>();
        if (prd.depth >= MAX_DEPTH) {
            prd.pixelColor = vec3f(0.0f);
            prd.end = 1;
            return;
        }
        if (sbtData.emissive_) {
                int MeshId = sbtData.ID;
                int PrimId = optixGetPrimitiveIndex();
                int num = optixLaunchParams.Lights_num;
                vec3f light_pdf;
                switch (MY_MODE) {
                case MY_MIS:
                    for (int i = 0; i < num; i++)
                    {
                        if (optixLaunchParams.All_Lights[i].id == MeshId)
                        {
                            LightParams* hit_light = &optixLaunchParams.All_Lights[i];
                            LightSample hit_point;
                            hit_light->sample(hit_point, prd.random, PrimId);
                            light_pdf = hit_point.Pdf_Light(prd.sourcePos, prd.nextPosition);
                            break;
                        }
                    }
                    prd.pixelColor = sbtData.emission * prd.throughout/(prd.weight+ light_pdf * num);
                    break;
                case MY_BRDF:
                    prd.pixelColor = sbtData.emission * prd.throughout ;
                    break;
                case MY_NEE:
                    prd.pixelColor = vec3f(0);
                }
                prd.end = 1;
                return;
        }            
        prd.throughout /= prd.weight;//非光源，并入即可

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

        const float alpha = sbtData.alpha_;
        const float d = sbtData.d;

        const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];

        float diffuse_max = max(max(diffuseColor[0], diffuseColor[1]), diffuseColor[2]);
     
        // ------------------------------------------------------------------
        //Begin of the true brdf
        // ------------------------------------------------------------------

        uint32_t u0, u1;
        
        vec3f new_dir;//光方向
        vec3f weight = 1.0f;//权重

        M_extansion mext;
        mext.diffuseColor = diffuseColor;
        mext.specColor = specColor;//材质属性
        //直接光
        int lightNum = optixLaunchParams.Lights_num;
        LightParams *LP = &optixLaunchParams.All_Lights[int(lightNum * prd.random())];
        LightSample LS;

        LP->sample(LS, prd.random,int(LP->num*prd.random()));

        int light_hit = LP->id;

        packPointer(&light_hit, u0, u1);
        vec3f lightDir = normalize(LS.position - surfPos);
        optixTrace(optixLaunchParams.traversable,
            surfPos + 1e-3f * Ng,
            lightDir,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT, 
            SHADOW_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            SHADOW_RAY_TYPE,            // missSBTIndex 
            u0, u1);

        if (light_hit == -1) {
            float dis = length(LS.position - surfPos);
            weight *= lightNum;
            weight *= Eval(sbtData, Ns, rayDir, lightDir, mext);
            vec3f Dir_color_contri = prd.throughout * weight  * LS.emission ;
            float Pdf_nee = LS.pdf * dis * dis / dot(LS.normal, -lightDir);
            switch (MY_MODE)
            {
            case MY_BRDF:
                break;
            case MY_NEE:
                prd.pixelColor = Dir_color_contri / (Pdf_nee);
                break;
            case MY_MIS:
                prd.pixelColor = Dir_color_contri / (Pdf_nee + Pdf_brdf(sbtData, Ns, rayDir, lightDir));
                break;
            }
        }
            
        const float RR = clamp(diffuse_max, 0.3f, 0.9f);//俄罗斯轮盘赌

        new_dir = SampleNewRay(sbtData, Ns, rayDir, prd);
        weight = Eval(sbtData, Ns, rayDir, new_dir, mext);
        prd.depth = prd.depth + 1;
        prd.throughout = prd.throughout * weight / RR;
        prd.sourcePos = surfPos;
        prd.nextPosition = new_dir;


        prd.weight = Pdf_brdf(sbtData, Ns, rayDir, new_dir);
        prd.throughout = min(prd.throughout, vec3f(1e3f));
        prd.pixelNormal = Ns;
        prd.pixelAlbedo = diffuseColor;
        prd.pixelColor = max(prd.pixelColor,vec3f(0.f));

        if (prd.random() > RR) {
            prd.end = 1;
            return;
        }
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
        PRD &prd = *getPRD<PRD>();
        prd.pixelColor = 0.f;
        prd.end = 1;
    }

    extern "C" __global__ void __miss__shadow()
    {

    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        const float color_max_avilable = 1.f;
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
        vec3f pixelNormal = 0.f;
        vec3f pixelAlbedo = 0.f;
        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
        {
            // normalized screen plane position, in [0,1]^2

            // iw: note for denoising that's not actually correct - if we
            // assume that the camera should only(!) cover the denoised
            // screen then the actual screen plane we shuld be using during
            // rendreing is slightly larger than [0,1]^2
            vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
                / vec2f(optixLaunchParams.frame.size));
 
            // generate ray direction
            vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
            
            prd.pixelColor = vec3f(0.f);
            prd.pixelAlbedo = vec3f(0.f);
            prd.pixelNormal = vec3f(0.f);
            prd.depth = 0;
            prd.throughout = vec3f(1.f);
            prd.sourcePos = camera.position;
            prd.weight = vec3f(1.f);
            prd.end = 0;
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
            pixelColor += prd.pixelColor;
            pixelNormal += prd.pixelNormal;
            pixelAlbedo += prd.pixelAlbedo;

            while (!prd.end)
            {
                //间接光
                optixTrace(optixLaunchParams.traversable,
                    prd.sourcePos + 1e-3f * prd.pixelNormal,
                    prd.nextPosition,
                    0.f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    RADIANCE_RAY_TYPE,            // SBT offset`
                    RAY_TYPE_COUNT,               // SBT stride
                    RADIANCE_RAY_TYPE,            // missSBTIndex 
                    u0, u1);
                    pixelColor += max(prd.pixelColor, vec3f(0.f));
            }
            //printf("End!!");
        }

        vec4f rgba(pixelColor / numPixelSamples, 1.f);
        vec4f albedo(pixelAlbedo / numPixelSamples, 1.f);
        vec4f normal(pixelNormal / numPixelSamples, 1.f);

        // and write/accumulate to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        if (optixLaunchParams.frame.frameID > 0)
        {
            rgba += float(optixLaunchParams.frame.frameID) * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
            rgba /= (optixLaunchParams.frame.frameID + 1.f);
        }
        optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
        optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
        optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
        //printf("we got rgba as %f   %f    %f    %f\n", rgba.x, rgba.y, rgba.z, rgba.w);
    }

} // ::osc
