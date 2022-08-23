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

#include "LaunchParams.h"
#include "gdt/random/random.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 2

namespace osc {

    typedef gdt::LCG<16> Random;

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    /*! per-ray data now captures random number generator, so programs
        can access RNG state */
    struct PRD {
        Random random;
        int depth;
        vec3f  pixelColor;
        vec3f  pixelNormal;
        vec3f  pixelAlbedo;
        float refraction_index;//当前光线相对于空气的折射率
    };

    static __forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
        void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    /*
*  Calculates refraction direction
*  lDir   : refraction vector
*  i   : incident vector
*  n   : surface normal
*  ior : index of refraction ( n2 / n1 )
*  returns false in case of total internal reflection, in that case lDir is
*  initialized to (0,0,0).
*/
    static __forceinline__  __device__
        bool refract(vec3f& lDir, vec3f const& i, vec3f const& n, const float ior)
    {
        vec3f nn = n;
        float negNdotV = dot(i, nn);
        float eta;

        if (negNdotV > 0.0f)
        {
            eta = ior;
            nn = -n;
            negNdotV = -negNdotV;
        }
        else
        {
            eta = 1.f / ior;
        }

        const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

        if (k < 0.0f)
        {
            // Initialize this value, so that lDir always leaves this function initialized.
            lDir = vec3f(0.f);
            return false;
        }
        else
        {
            lDir = normalize(eta * i - (eta * negNdotV + sqrtf(k)) * nn);
            return true;
        }
    }

    //Schlick approximation of Fresnel reflectance
    static __forceinline__  __device__
        float fresnel_schlick(const float cos_theta, const float exponent = 3.0f,
            const float minimum = 0.1f, const float maximum = 1.0f)
    {
        /*
          Clamp the result of the arithmetic due to floating point precision:
          the result should lie strictly within [minimum, maximum]
          return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta, exponent),
                       minimum, maximum);
        */

        /* The max doesn'rDir seem like it should be necessary, but without it you get
            annoying broken pixels at the center of reflective spheres where cos_theta ~ 1.
        */
        return clamp(minimum + (maximum - minimum) * powf(fmaxf(0.0f, 1.0f - cos_theta), exponent),
            minimum, maximum);
    }


    static __forceinline__  __device__
        vec3f AxisAngle(const vec3f& w, const float cos2theta, const float phi)
    {
        const float cos_theta = std::sqrt(cos2theta);
        const float sin_theta = std::sqrt(1 - cos2theta);
        const vec3f u = normalize(cross(std::abs(w[0]) > float(.1) ? vec3f(0, 1, 0) : vec3f(1, 0, 0), w));
        const vec3f v = cross(w, u);
        return normalize(u * std::cos(phi) * sin_theta + v * std::sin(phi) * sin_theta + w * cos_theta);
    }
    
    static __forceinline__  __device__
        vec3f Sample(const vec3f diffuse, const vec3f spec, const float alpha, const vec3f& n, const vec3f& wi, vec3f& weight)
    {
        const float PI_ = 3.1415926535897932384626;
        const float k_d_ = (diffuse.x + diffuse.y + diffuse.z) / 3;
        const float k_s_ = (spec.x + spec.y + spec.z) / 3;
        const float R = k_s_ ? k_d_ / (k_d_ + k_s_) : 1.f;
        Random x;
        const float r0 = x();
        if (r0 < R) { // sample diffuse ray
            weight = k_d_ ? diffuse / R : vec3f(0, 0, 0);
            return AxisAngle(n, x(), x() * 2 * PI_);
        }

        else { // sample specular ray
            if (0) {
                const vec3f d = AxisAngle(n * 2 * dot(n, wi) - wi, std::pow(x(), float(2) / (alpha + 2)), x() * 2 * PI_);
                weight = dot(n, d) <= 0 || !k_s_ ? vec3f(0, 0, 0) : spec / (1 - R);
                return d;
            }
            else { // for ideal mirrors
                weight = k_s_ ? spec / (1 - R) : vec3f(0, 0, 0);
                return n * 2 * dot(n, wi) - wi;
            }
        }
    }

    // yqy's code below-------------------
        // tool functions---------------
    static __forceinline__  __device__
        float smithG_GGX(float NDotv, float alphaG)
    {
        float a = alphaG * alphaG;
        float b = NDotv * NDotv;
        return 1.0f / (NDotv + sqrtf(a + b - a * b));
    }
    static __forceinline__  __device__
        float SchlickFresnel(float u)
    {
        float m = clamp(1.0f - u, 0.0f, 1.0f);
        float m2 = m * m;
        return m2 * m2 * m; // pow(m,5)
    }
    static __forceinline__  __device__
        float GTR1(float NDotH, float a)
    {
        if (a >= 1.0f) return (1.0f / M_PI);
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
        return (a2 - 1.0f) / (M_PI * logf(a2) * t);
    }
    static __forceinline__  __device__
        float GTR2(float NDotH, float a)
    {
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
        return a2 / (M_PI * t * t);
    }
    template <typename T>
    static __forceinline__  __device__
        T lerp(T a, T b, float t)
    {
        return a+t*(b-a);
    }

    // PDF returns a percentage
    static __forceinline__  __device__
        float Pdf(const TriangleMeshSBTData& mat, const vec3f& normal, const vec3f& ray_in, const vec3f& ray_out)
    {
        vec3f n = normal;
        vec3f V = -ray_in;
        vec3f L = ray_out;

        float specularAlpha = max(0.001f, mat.roughness);
        //float clearcoatAlpha = std::lerp(0.1f, 0.001f, mat.clearcoatGloss);// 1.0 default
        float clearcoatAlpha = lerp(0.1f, 0.001f, 1.0f);
        
        float diffuseRatio = 0.5f * (1.f - mat.metallic);
        float specularRatio = 1.f - diffuseRatio;

        vec3f half = normalize(L + V);

        float cosTheta = abs(dot(half, n));
        float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
        float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

        // calculate diffuse and specular pdfs and mix ratio
        //float ratio = 1.0f / (1.0f + mat.clearcoat);//0.0 default
        float ratio = 1.0f / (1.0f + 0.0f);
        float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
        float pdfDiff = abs(dot(L, n)) * (1.0f / M_PI);

        // weight pdfs according to ratios
        return diffuseRatio * pdfDiff + specularRatio * pdfSpec;
    }

    static __forceinline__  __device__
        vec3f Sample_yqy(const TriangleMeshSBTData& mat, const vec3f& normal, const vec3f& ray_in)
    {        
        vec3f N = normal;
        vec3f V = -ray_in;

        vec3f dir;
        Random x;
        //const float probability = x();
        float diffuseRatio = 0.5f * (1.0f - mat.metallic);

        //const float r1 = y();
        //const float r2 = z();
        const float r2 = x();

        if (x() < diffuseRatio) // sample diffuse
        {
            dir = AxisAngle(N, x(), x() * 2 * M_PI);
        }
        else
        {
            float a = max(0.001f, mat.roughness);

            float phi = x() * 2.0f * M_PI;

            float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
            //float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
            //float sinPhi = sinf(phi);
            //float cosPhi = cosf(phi);

            vec3f half = AxisAngle(N, 2*cosTheta*cosTheta-1, phi);

            dir = 2.0f * dot(V, half) * half - V; //reflection vector

        }
        return dir;
    }

    static __forceinline__  __device__
        vec3f Eval(const TriangleMeshSBTData& mat, const vec3f& normal, const vec3f& ray_in, const vec3f& ray_out)
    {
        vec3f N = normal;
        vec3f V = -ray_in;
        vec3f L = ray_out;

        float NDotL = dot(N, L);
        float NDotV = dot(N, V);
        if (NDotL <= 0.0f || NDotV <= 0.0f) return vec3f(0.0f);

        vec3f H = normalize(L + V);
        float NDotH = dot(N, H);
        float LDotH = dot(L, H);

        vec3f Cdlin = mat.color;

        float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

        vec3f Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : vec3f(1.0f); // normalize lum. to isolate hue+sat
        //vec3f Cspec0 = lerp(mat.specular * 0.08f * lerp(vec3f(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
        //vec3f Csheen = lerp(vec3f(1.0f), Ctint, mat.sheenTint);
        vec3f Cspec0 = lerp(0.5f * 0.08f * lerp(vec3f(1.0f), Ctint, 0.0f), Cdlin, mat.metallic);
        vec3f Csheen = lerp(vec3f(1.0f), Ctint, 0.0f);

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
        // and mix in diffuse retro-reflection based on roughness
        float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
        float Fd90 = 0.5f + 2.0f * LDotH * LDotH * mat.roughness;
        float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

        // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
        // 1.25 scale is used to (roughly) preserve albedo
        // Fss90 used to "flatten" retroreflection based on roughness
        float Fss90 = LDotH * LDotH * mat.roughness;
        float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
        float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

        // specular
        //float aspect = sqrt(1-mat.anisotrokPic*.9);
        //float ax = Max(.001f, sqr(mat.roughness)/aspect);
        //float ay = Max(.001f, sqr(mat.roughness)*aspect);
        //float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);

        float a = max(0.001f, mat.roughness);
        float Ds = GTR2(NDotH, a);
        float FH = SchlickFresnel(LDotH);
        vec3f Fs = lerp(Cspec0, vec3f(1.0f), FH);
        float roughg = sqrt(mat.roughness * 0.5f + 0.5f);
        float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

        // sheen
        vec3f Fsheen = FH * mat.sheen * Csheen;

        // clearcoat (ior = 1.5 -> F0 = 0.04)
        //float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
        float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, 1.0f));
        float Fr = lerp(0.04f, 1.0f, FH);
        float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

        //vec3f out = ((1.0f / M_PI) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
        //    * (1.0f - mat.metallic)
         //   + Gs * Fs * Ds + 0.25f * mat.clearcoat * Gr * Fr * Dr;
        vec3f out = ((1.0f / M_PI) * lerp(Fd, ss, 0.0f) * Cdlin + Fsheen)
            * (1.0f - mat.metallic)
            + Gs * Fs * Ds + 0.25f * 0.0f * Gr * Fr * Dr;
        return out * clamp(dot(N, L), 0.0f, 1.0f);
    }
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
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        PRD& prd = *getPRD<PRD>();
        const int Maxdepth = 4;
        const float refraction_color = 1.0f;
        const float reflection_color = 1.0f;
        if (prd.depth >= Maxdepth) {
            prd.pixelColor = 0.0f;
            return;
        }
        if (sbtData.emissive_) {
            prd.pixelColor *= sbtData.emission;
            return;
        }
        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int   primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // ------------------------------------------------------------------
        // compute normal, using either shading normal (if avail), or
        // geometry normal (fallback)
        // ------------------------------------------------------------------
        const vec3f& A = sbtData.vertex[index.x];
        const vec3f& B = sbtData.vertex[index.y];
        const vec3f& C = sbtData.vertex[index.z];
        vec3f Ng = cross(B - A, C - A);
        vec3f Ns = (sbtData.normal)
            ? ((1.f - u - v) * sbtData.normal[index.x]
                + u * sbtData.normal[index.y]
                + v * sbtData.normal[index.z])
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
        if (sbtData.hasTexture && sbtData.texcoord) {
            const vec2f tc
                = (1.f - u - v) * sbtData.texcoord[index.x]
                + u * sbtData.texcoord[index.y]
                + v * sbtData.texcoord[index.z];

            vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
            diffuseColor *= (vec3f)fromTexture;
        }

        vec3f specColor = 0.0f;
        if (sbtData.hasSpecTexture && sbtData.texcoord) {
            const vec2f tc
                = (1.f - u - v) * sbtData.texcoord[index.x]
                + u * sbtData.texcoord[index.y]
                + v * sbtData.texcoord[index.z];
            vec4f fromTexture = tex2D<float4>(sbtData.spectexture, tc.x, tc.y);
            specColor = (vec3f)fromTexture;
        }

        const float alpha = sbtData.alpha_;
        const float d = sbtData.d;

        // start with some ambient term
        //vec3f pixelColor = (0.1f + 0.2f*fabsf(dot(Ns,rayDir)))*diffuseColor;
        vec3f pixelColor = 0.f;

        // ------------------------------------------------------------------
        // compute shadow
        // ------------------------------------------------------------------
        const vec3f surfPos
            = (1.f - u - v) * sbtData.vertex[index.x]
            + u * sbtData.vertex[index.y]
            + v * sbtData.vertex[index.z];

        const int numLightSamples = NUM_LIGHT_SAMPLES;
        for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
            float reflection = 1.0f;
            vec3f rDir;//折射
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
                float rimportance =  (1.0f - reflection) * refraction_color;
                PRD newprd;
                // the values we store the PRD pointer in:
                uint32_t u0, u1;
                packPointer(&newprd, u0, u1);
                newprd.pixelColor = prd.pixelColor * rimportance;
                newprd.depth = prd.depth + 1;
                if (prd.refraction_index > 1.f)
                    newprd.refraction_index = 0.684f;
                else
                    newprd.refraction_index = 1.46f;
                optixTrace(optixLaunchParams.traversable,
                    surfPos - 1e-3f * Ng,
                    rDir,
                    0.f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    // For shadow rays: skip any/closest hit shaders and terminate on first
                    // intersection with anything. The miss shader is used to mark if the
                    // light was visible.
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    RADIANCE_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    RADIANCE_RAY_TYPE,            // missSBTIndex 
                    u0, u1);
                pixelColor +=   newprd.pixelColor / numLightSamples;
            }
            if (cos_theta < 0.0f) {
                float limportance = reflection * reflection_color;
                // the values we store the PRD pointer in:
                PRD newprd;
                vec3f weight = 1.0f;
                vec3f mont_dir = Sample(diffuseColor, specColor, alpha, Ns, -rayDir, weight);
                //vec3f mont_dir = Sample_yqy(sbtData, Ns, rayDir);
                //weight = Eval(sbtData, Ns, rayDir, mont_dir);
                uint32_t u0, u1;
                packPointer(&newprd, u0, u1);
                newprd.depth = prd.depth + 1;
                newprd.refraction_index = 1.0;
                newprd.pixelColor = prd.pixelColor  * weight* limportance;
                optixTrace(optixLaunchParams.traversable,
                    surfPos + 1e-3f * Ng,
                    mont_dir,
                    0.f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    // For shadow rays: skip any/closest hit shaders and terminate on first
                    // intersection with anything. The miss shader is used to mark if the
                    // light was visible.
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    RADIANCE_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    RADIANCE_RAY_TYPE,            // missSBTIndex 
                    u0, u1);
                pixelColor +=  newprd.pixelColor / numLightSamples;
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
        PRD& prd = *getPRD<PRD>();
        // set to constant white as background color
        prd.pixelColor *= vec3f(10.f);
    }

    extern "C" __global__ void __miss__shadow()
    {
        // we didn't hit anything, so the light is visible
        vec3f& prd = *(vec3f*)getPRD<vec3f>();
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
        const auto& camera = optixLaunchParams.camera;

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
        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
            // normalized screen plane position, in [0,1]^2

            // iw: note for denoising that's not actually correct - if we
            // assume that the camera should only(!) cover the denoised
            // screen then the actual screen plane we shuld be using during
            // rendreing is slightly larger than [0,1]^2
            vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
                / vec2f(optixLaunchParams.frame.size));
            // screen
            //   = screen
            //   * vec2f(optixLaunchParams.frame.denoisedSize)
            //   * vec2f(optixLaunchParams.frame.size)
            //   - 0.5f*(vec2f(optixLaunchParams.frame.size)
            //           -
            //           vec2f(optixLaunchParams.frame.denoisedSize)
            //           );

            // generate ray direction
            vec3f rayDir = normalize(camera.direction
                + (screen.x - 0.5f) * camera.horizontal
                + (screen.y - 0.5f) * camera.vertical);

            prd.pixelColor = vec3f(1.f);
            prd.depth = 0;
            prd.refraction_index = 1.46f;
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
        }

        vec4f rgba(pixelColor / numPixelSamples, 1.f);
        vec4f albedo(pixelAlbedo / numPixelSamples, 1.f);
        vec4f normal(pixelNormal / numPixelSamples, 1.f);

        // and write/accumulate to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        if (optixLaunchParams.frame.frameID > 0) {
            rgba
                += float(optixLaunchParams.frame.frameID)
                * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
            rgba /= (optixLaunchParams.frame.frameID + 1.f);
        }
        optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
        optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
        optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
    }

} // ::osc
