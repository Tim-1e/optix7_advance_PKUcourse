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

#include "LightParams.h"
#include "LaunchParams.h"
#include "tool_function.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 2

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
        /* not going to be used ... */
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
        PRD &prd = *getPRD<PRD>();
        const int Maxdepth = 4;
        const float refraction_color = 1.0f;
        const float reflection_color = 1.0f;
        if (prd.depth >= Maxdepth)
        {
            prd.pixelColor = 0.0f;
            return;
        }
        if (sbtData.emissive_)
        {
            prd.pixelColor *= sbtData.emission;
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

        if (dot(rayDir, Ng) > 0.f)
            Ng = -Ng;
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

        // start with some ambient term
        // vec3f pixelColor = (0.1f + 0.2f*fabsf(dot(Ns,rayDir)))*diffuseColor;
        vec3f pixelColor = 0.f;

        // ------------------------------------------------------------------
        // compute shadow
        // ------------------------------------------------------------------
        const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];

        const int numLightSamples = NUM_LIGHT_SAMPLES;
        for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++)
        {
            float reflection = 1.0f;
            vec3f rDir; //折射
            float cos_theta = dot(rayDir, Ns);
            if (d < 0.5 && refract(rDir, rayDir, Ns, prd.refraction_index))
            {
                //根据入射角的cos值来计算折射与反射的比率，在一定角度就全反射了，在垂直时是全折射
                //正入射进去,这是正常情况
                if (cos_theta < 0.0f)
                {
                    cos_theta = -cos_theta;
                }
                else
                {
                    //异常了使用折射光线再计算一次
                    cos_theta = dot(rDir, Ns);
                }

                reflection = fresnel_schlick(cos_theta);
                float rimportance = (1.0f - reflection) * refraction_color;
                PRD newprd;
                // the values we store the PRD pointer in:
                uint32_t u0, u1;
                packPointer(&newprd, u0, u1);
                newprd.random.init(prd.random() * 0x01000000, prd.random() * 0x01000000);
                newprd.pixelColor = prd.pixelColor * rimportance;
                newprd.depth = prd.depth + 1;
                if (prd.refraction_index > 1.f)
                    newprd.refraction_index = 0.684f;
                else
                    newprd.refraction_index = 1.46f;
                optixTrace(optixLaunchParams.traversable,
                           surfPos - 1e-3f * Ng,
                           rDir,
                           0.f,   // tmin
                           1e20f, // tmax
                           0.0f,  // rayTime
                           OptixVisibilityMask(255),
                           // For shadow rays: skip any/closest hit shaders and terminate on first
                           // intersection with anything. The miss shader is used to mark if the
                           // light was visible.
                           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                           RADIANCE_RAY_TYPE, // SBT offset
                           RAY_TYPE_COUNT,    // SBT stride
                           RADIANCE_RAY_TYPE, // missSBTIndex
                           u0, u1);
                pixelColor += newprd.pixelColor / numLightSamples;
            }
            if (cos_theta < 0.0f)
            {
                float limportance = reflection * reflection_color;
                // the values we store the PRD pointer in:
                PRD newprd;
                vec3f weight = 1.0f;
                vec3f mont_dir;
                mont_dir = Sample(diffuseColor, specColor, alpha, Ns, -rayDir, weight, prd);
                // mont_dir = Sample_adjust(sbtData, Ns, rayDir,prd);
                M_extansion mext;
                mext.diffuseColor = diffuseColor;
                mext.specColor = specColor;
                weight = Eval(sbtData, Ns, rayDir, mont_dir, mext);
                uint32_t u0, u1;
                packPointer(&newprd, u0, u1);
                newprd.random.init(prd.random() * 0x01000000, prd.random() * 0x01000000);
                newprd.depth = prd.depth + 1;
                newprd.refraction_index = 1.0;
                newprd.pixelColor = prd.pixelColor * weight * limportance;
                optixTrace(optixLaunchParams.traversable,
                           surfPos + 1e-3f * Ng,
                           mont_dir,
                           0.f,   // tmin
                           1e20f, // tmax
                           0.0f,  // rayTime
                           OptixVisibilityMask(255),
                           // For shadow rays: skip any/closest hit shaders and terminate on first
                           // intersection with anything. The miss shader is used to mark if the
                           // light was visible.
                           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                           RADIANCE_RAY_TYPE, // SBT offset
                           RAY_TYPE_COUNT,    // SBT stride
                           RADIANCE_RAY_TYPE, // missSBTIndex
                           u0, u1);
                pixelColor += newprd.pixelColor / numLightSamples;
            }
        }
        prd.pixelNormal = Ns;
        prd.pixelAlbedo = diffuseColor;
        prd.pixelColor = pixelColor;
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
        // set to constant white as background color
        // 这儿可能要改成环境光
        prd.pixelColor *= vec3f(10.f);
    }

    extern "C" __global__ void __miss__shadow()
    {
        // we didn't hit anything, so the light is visible
        vec3f &prd = *(vec3f *)getPRD<vec3f>();
        prd = vec3f(0.f);
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
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
        vec3f pixelNormal = 0.f;
        vec3f pixelAlbedo = 0.f;
        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
        {
            // normalized screen plane position, in [0,1]^2

            // iw: note for denoising that's not actually correct - if we
            // assume that the camera should only(!) cover the denoised
            // screen then the actual screen plane we shuld be using during
            // rendreing is slightly larger than [0,1]^2
            vec2f screen(vec2f(ix + prd.random(), iy + prd.random()) / vec2f(optixLaunchParams.frame.size));
            // screen
            //   = screen
            //   * vec2f(optixLaunchParams.frame.denoisedSize)
            //   * vec2f(optixLaunchParams.frame.size)
            //   - 0.5f*(vec2f(optixLaunchParams.frame.size)
            //           -
            //           vec2f(optixLaunchParams.frame.denoisedSize)
            //           );

            // generate ray direction
            vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);

            prd.pixelColor = vec3f(1.f);
            prd.depth = 0;
            prd.refraction_index = 1.46f;
            optixTrace(optixLaunchParams.traversable,
                       camera.position,
                       rayDir,
                       0.f,   // tmin
                       1e20f, // tmax
                       0.0f,  // rayTime
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                       RADIANCE_RAY_TYPE,             // SBT offset
                       RAY_TYPE_COUNT,                // SBT stride
                       RADIANCE_RAY_TYPE,             // missSBTIndex
                       u0, u1);
            pixelColor += prd.pixelColor;
            pixelNormal += prd.pixelNormal;
            pixelAlbedo += prd.pixelAlbedo;
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
    }

} // ::osc
