#pragma once
#include "BDPT.h"
#include "gdt/math/vec.h"
#include "tool_function.h"

namespace ocs {
    using namespace gdt;
    __forceinline__ __device__ vec3f evalPath(std::vector<osc::BDPTVertex>& path)
    {
        float pdf = 0.0f;
        vec3f contri;
        contri = contriCompute(path);

        for (int i = 1; i < path.size() - 1; i++)
        {
            pdf += pdfCompute(path, i);
        }
        vec3f ans = contri / pdf;
        if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z))
        {
            return vec3f(0.0f);
        }
        return contri / pdf;
    }
    __forceinline__ __device__ vec3f contriCompute(std::vector<BDPTVertex>& path)
    {
        vec3f throughput = vec3f(1.0f);
        BDPTVertex& light = path[path.size() - 1];
        BDPTVertex& lastMidPoint = path[path.size() - 2];
        vec3f lightLine = lastMidPoint.position - light.position;
        vec3f lightDirection = normalize(lightLine);
        float lAng = dot(light.normal, lightDirection);
        if (lAng < 0.0f)
        {
            return vec3f(0.0f);
        }
        vec3f Le = light.mat.emission * lAng;
        throughput *= Le;
        for (int i = 1; i < path.size(); i++)
        {
            BDPTVertex& midPoint = path[i];
            BDPTVertex& lastPoint = path[i - 1];
            vec3f line = midPoint.position - lastPoint.position;
            throughput /= dot(line, line);
        }
        for (int i = 1; i < path.size() - 1; i++)
        {
            BDPTVertex& midPoint = path[i];
            BDPTVertex& lastPoint = path[i - 1];
            BDPTVertex& nextPoint = path[i + 1];
            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection))
                * Eval(midPoint.mat, midPoint.normal, -lastDirection, nextDirection, midPoint.ext);
        }
        return throughput;
    }
    __forceinline__ __device__ float pdfCompute(std::vector<BDPTVertex>& path, int lightPathLength)
    {
        int eyePathLength = path.size() - lightPathLength;
        float pdf = 1.0f;

        if (lightPathLength > RR_BEGIN_DEPTH)
        {
            pdf *= pow(RR_RATE, lightPathLength - RR_BEGIN_DEPTH);
        }
        if (eyePathLength > RR_BEGIN_DEPTH)
        {
            pdf *= pow(RR_RATE, eyePathLength - RR_BEGIN_DEPTH);
        }
        if (lightPathLength > 0)
        {
            BDPTVertex& light = path[path.size() - 1];
            pdf *= light.pdf;
        }
        if (lightPathLength > 1)
        {
            BDPTVertex& light = path[path.size() - 1];
            BDPTVertex& lastMidPoint = path[path.size() - 2];
            vec3f lightLine = lastMidPoint.position - light.position;
            vec3f lightDirection = normalize(lightLine);
            pdf *= abs(dot(lightDirection, light.normal)) / M_PI;

            /*��������ǵ��µ�pdf*/
            for (int i = 1; i < lightPathLength; i++)
            {
                BDPTVertex& midPoint = path[path.size() - i - 1];
                BDPTVertex& lastPoint = path[path.size() - i];
                vec3f line = midPoint.position - lastPoint.position;
                vec3f lineDirection = normalize(line);
                pdf *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
            }

            for (int i = 1; i < lightPathLength - 1; i++)
            {
                BDPTVertex& midPoint = path[path.size() - i - 1];
                BDPTVertex& lastPoint = path[path.size() - i];
                BDPTVertex& nextPoint = path[path.size() - i - 2];
                vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
                vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
                pdf *= Pdf_brdf(midPoint.mat, midPoint.normal, -lastDirection, nextDirection);
            }

        }
        /*����ͶӰ�ǵ��µ�pdf�仯*/
        for (int i = 1; i < eyePathLength; i++)
        {
            BDPTVertex& midPoint = path[i];
            BDPTVertex& lastPoint = path[i - 1];
            vec3f line = midPoint.position - lastPoint.position;
            vec3f lineDirection = normalize(line);
            pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        /*��������ĸ���*/
        for (int i = 1; i < eyePathLength - 1; i++)
        {
            BDPTVertex& midPoint = path[i];
            BDPTVertex& lastPoint = path[i - 1];
            BDPTVertex& nextPoint = path[i + 1];
            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            pdf *= Pdf_brdf(midPoint.mat, midPoint.normal, -lastDirection, nextDirection);
        }
        return pdf;
    }
}