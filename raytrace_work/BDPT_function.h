#pragma once
#include "BDPT.h"
#include "tool_function.h"

namespace osc {
    
    using namespace gdt;
    
    __forceinline__ __device__ vec3f contriCompute(const BDPTPath& path);
    __forceinline__ __device__ float pdfCompute(const BDPTPath& path, int lightPathLength);
    __forceinline__ __device__ vec3f evalPath(const BDPTPath& path)
    {
        float pdf = 0.0f;
        vec3f contri;
        contri = contriCompute(path);

        for (int i = 1; i < path.length-1; i++)
        {
            if (i > Maxdepth || path.length - i > Maxdepth) continue;
            pdf += pdfCompute(path, i);//i表示光路径中顶点个数
        }
        //std::printf("contri:%f,length:%d,pdf:%f\n", contri.r,path.length, pdf);
        vec3f ans = contri / float(pdf);
        if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z))
        {
            return vec3f(0.0f);
        }
        return ans;
    }

    __forceinline__ __device__ vec3f contriCompute(const BDPTPath& path)
    {
        vec3f throughput = vec3f(1.0f);
        const BDPTVertex& light = path.vertexs[path.length - 1];
        const BDPTVertex& lastMidPoint = path.vertexs[path.length - 2];
        vec3f lightLine = lastMidPoint.position - light.position;
        vec3f lightDirection = normalize(lightLine);
        float lAng = dot(light.normal, lightDirection);
        if (lAng < 0.0f)
        {
            return vec3f(0.0f);
        }
        vec3f Le = light.mat->emission * lAng;
        throughput *= Le;

        //std::printf("color: %f %f %f\n",light.mat->emission.x, light.mat->emission.y, light.mat->emission.z);

        //const BDPTVertex& eye = path.vertexs[0];
        //const BDPTVertex& firstHit = path.vertexs[1];
        //vec3f eyeLine = firstHit.position - eye.position;
        //vec3f eyeDirection = normalize(eyeLine);
        //throughput *= dot(eyeDirection, eye.normal);

        for (int i = 1; i < path.length - 1; i++)
        {
            const BDPTVertex& midPoint = path.vertexs[i];
            const BDPTVertex& lastPoint = path.vertexs[i - 1];
            const BDPTVertex& nextPoint = path.vertexs[i + 1];
            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection)) * Eval(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection, midPoint.ext);
        }
        return throughput;
    }

    __forceinline__ __device__ vec3f singlePathContriCompute(const BDPTPath& path)
    {
        const float RR_RATE = 0.8f;
        vec3f throughput = vec3f(1.0f);
        const BDPTVertex& light = path.vertexs[path.length - 1];
        const BDPTVertex& lastMidPoint = path.vertexs[path.length - 2];
        vec3f lightLine = lastMidPoint.position - light.position;
        vec3f lightDirection = normalize(lightLine);
        float lAng = dot(light.normal, lightDirection);
        if (lAng < 0.0f)
        {
            return vec3f(0.0f);
        }
        vec3f Le = light.mat->emission;
        throughput *= Le;

        for (int i = 1; i < path.length - 1; i++)
        {
            const BDPTVertex& midPoint = path.vertexs[i];
            const BDPTVertex& lastPoint = path.vertexs[i - 1];
            const BDPTVertex& nextPoint = path.vertexs[i + 1];
            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            vec3f EVAL, BRDF;
            EVAL = Eval(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection, midPoint.ext);
            BRDF = Pdf_brdf(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection);
            vec3f xx = -lastDirection;
            vec3f yy = nextDirection;
            throughput *= abs(dot(midPoint.normal, nextDirection))
                * EVAL
                /BRDF / RR_RATE;
        }
        return throughput;
    }

    __forceinline__ __device__ float pdfCompute(const BDPTPath& path, int lightPathLength)
    {
        int eyePathLength = path.length - lightPathLength;
        float pdf = 1.0f;
        const float RR_RATE = 0.8f;
        const int RR_BEGIN_DEPTH = 1;
        if (lightPathLength > RR_BEGIN_DEPTH)
        {
            pdf *= pow(RR_RATE, lightPathLength - RR_BEGIN_DEPTH);
        }
        if (eyePathLength > RR_BEGIN_DEPTH)
        {
            pdf *= pow(RR_RATE, eyePathLength - RR_BEGIN_DEPTH);
        }

        const BDPTVertex& light = path.vertexs[path.length - 1];
        pdf *= light.pdf;

        if (lightPathLength > 1)
        {
            pdf /= 2 * M_PI;

            for (int i = 1; i < lightPathLength; i++)
            {
                const BDPTVertex& midPoint = path.vertexs[path.length - i - 1];
                const BDPTVertex& lastPoint = path.vertexs[path.length - i];
                vec3f line = midPoint.position - lastPoint.position;
                vec3f lineDirection = normalize(line);
                pdf *= abs(dot(midPoint.normal, lineDirection));
            }

            for (int i = 1; i < lightPathLength - 1; i++)
            {
                const BDPTVertex& midPoint = path.vertexs[path.length - i - 1];
                const BDPTVertex& lastPoint = path.vertexs[path.length - i];
                const BDPTVertex& nextPoint = path.vertexs[path.length - i - 2];
                vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
                vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
                pdf *= Pdf_brdf(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection);
            }

        }

        for (int i = 1; i < eyePathLength; i++)
        {
            const BDPTVertex& midPoint = path.vertexs[i];
            const BDPTVertex& lastPoint = path.vertexs[i - 1];
            vec3f line = midPoint.position - lastPoint.position;
            vec3f lineDirection = normalize(line);
            pdf *=  abs(dot(midPoint.normal, lineDirection));

        }

        for (int i = 1; i < eyePathLength - 1; i++)
        {
            const BDPTVertex& midPoint = path.vertexs[i];
            const BDPTVertex& lastPoint = path.vertexs[i - 1];
            const BDPTVertex& nextPoint = path.vertexs[i + 1];
            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            pdf *= Pdf_brdf(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection);
        }
        vec3f connect_path = path.vertexs[eyePathLength].position - path.vertexs[eyePathLength - 1].position;
        pdf *= dot(connect_path, connect_path);
        return pdf;
    }
    __forceinline__ __device__ void Connect_two_path(const BDPTPath& eye_path, const BDPTPath& light_path, BDPTPath& merge_path,int eyelength,int lightlength)
    {
        for (int i = 0; i < eyelength; i++)
        {
            merge_path.vertexs[i] = eye_path.vertexs[i];
        }
        for (int i = 0; i < lightlength; i++)
        {
            merge_path.vertexs[i+eyelength] = light_path.vertexs[lightlength -1-i];
        }
        merge_path.length = eyelength + lightlength;
    }
}