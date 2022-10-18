#pragma once
#include "gdt/random/random.h"
#include "PRD.h"

  namespace osc {
    enum PdfType {
        DirectLight, InDirectLight
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
        return a + t * (b - a);
    }

    // PDF returns a percentage
    static __forceinline__  __device__
        float Pdf_brdf(const TriangleMeshSBTData& mat, const vec3f& normal, const vec3f& ray_in, const vec3f& ray_out)
    {
        vec3f n = normal;
        vec3f V = -ray_in;
        vec3f L = ray_out;

        float specularAlpha = max(0.001f, mat.roughness);
        float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);// 1.0 default

        float diffuseRatio = 0.5f * (1.f - mat.metallic);
        float specularRatio = 1.f - diffuseRatio;

        vec3f half = normalize(L + V);

        float cosTheta = abs(dot(half, n));
        float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
        float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

        // calculate diffuse and specular pdfs and mix ratio
        float ratio = 1.0f / (1.0f + mat.clearcoat);//0.0 default
        float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
        float pdfDiff = abs(dot(L, n)) * (1.0f / M_PI);

        // weight pdfs according to ratios
        return diffuseRatio * pdfDiff + specularRatio * pdfSpec;
    }

    static __forceinline__  __device__
        vec3f Sample_adjust(const TriangleMeshSBTData& mat, const vec3f& normal, const vec3f& ray_in, PRD& prd)
    {
        vec3f N = normal;
        vec3f V = -ray_in;

        vec3f dir;
        float probability = prd.random();
        float diffuseRatio = 0.5f * (1.0f - mat.metallic);

        float r1 = prd.random();
        float r2 = prd.random();


        if (probability < diffuseRatio) // sample diffuse
        {
            dir = AxisAngle(N, r1, r2 * 2 * M_PI);
        }
        else
        {
            float a = max(0.001f, mat.roughness);

            float phi = r1 * 2.0f * M_PI;

            float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));

            vec3f half = AxisAngle(N,  cosTheta * cosTheta, phi);

            dir = 2.0f * dot(V, half) * half - V; //reflection vector

        }
        return dir;
    }

    static __forceinline__  __device__
        vec3f Eval(const TriangleMeshSBTData& mat, const vec3f& normal, const vec3f& ray_in, const vec3f& ray_out,const M_extansion& mext )
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
        if (mat.hasTexture && mat.texcoord) Cdlin = mext.diffuseColor;
        float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

        vec3f Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : vec3f(1.0f); // normalize lum. to isolate hue+sat

        
        vec3f Cspec0 = lerp(mat.specular * 0.08f * lerp(vec3f(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
        vec3f Csheen = lerp(vec3f(1.0f), Ctint, mat.sheenTint);

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
        float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));

        float Fr = lerp(0.04f, 1.0f, FH);
        float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

        vec3f out = ((1.0f / M_PI) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
            * (1.0f - mat.metallic)
            + Gs * Fs * Ds + 0.25f * mat.clearcoat * Gr * Fr * Dr;
        return out;
    }
}

