#pragma once
#include "BDPT.h"
#include "tool_function.h"

namespace osc {
    
    using namespace gdt;
    
    __forceinline__ __device__ float pdfCompute(const BDPTPath& path, int lightPathLength);
    __forceinline__ __device__ float evalPathPdf(const BDPTPath& path)
    {
        float pdf = 0.0f;

        float PdfEye[Maxdepth*2] = {}, PdfLight[Maxdepth*2] = {};
        const BDPTVertex& light = path.vertexs[path.length - 1];
        PdfLight[0] = light.pdf;
        PdfEye[0] = 1.f;

        {
            PdfLight[1] = light.pdf / (2 * M_PI);
            const BDPTVertex& midPoint = path.vertexs[path.length - 2];
            const BDPTVertex& lastPoint = path.vertexs[path.length - 1];
            vec3f line = midPoint.position - lastPoint.position;
            vec3f lineDirection = normalize(line);
            PdfLight[1] *= abs(dot(midPoint.normal, lineDirection));
        }

        {
            const BDPTVertex& midPoint = path.vertexs[1];
            const BDPTVertex& lastPoint = path.vertexs[0];
            vec3f line = midPoint.position - lastPoint.position;
            vec3f lineDirection = normalize(line);
            PdfEye[1] = abs(dot(midPoint.normal, lineDirection));
        }

        const float RR_RATE = 0.8f;

        for (int lightPathLength = 2; lightPathLength <= path.length-2; lightPathLength++)
        {
            float pdfonce = 1.f;
            pdfonce *= RR_RATE;
            

            const BDPTVertex& midPoint = path.vertexs[path.length - lightPathLength];
            const BDPTVertex& lastPoint = path.vertexs[path.length - lightPathLength+1];
            const BDPTVertex& nextPoint = path.vertexs[path.length - lightPathLength-1];

            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            pdfonce *= Pdf_brdf(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection);
            vec3f line = midPoint.position - nextPoint.position;
            vec3f lineDirection = normalize(line);
            pdfonce *= abs(dot(nextPoint.normal, lineDirection));
            PdfLight[lightPathLength] = pdfonce * PdfLight[lightPathLength - 1];
        }

        for (int eyePathLength = 2; eyePathLength <= path.length-2; eyePathLength++)
        {
            float pdfonce = 1.f;
            pdfonce *= RR_RATE;
            
            
            const BDPTVertex& midPoint = path.vertexs[eyePathLength-1];
            const BDPTVertex& lastPoint = path.vertexs[eyePathLength-2];
            const BDPTVertex& nextPoint = path.vertexs[eyePathLength];

            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            pdfonce *= Pdf_brdf(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection);
            vec3f line = midPoint.position - nextPoint.position;
            vec3f lineDirection = normalize(line);
            pdfonce *= abs(dot(nextPoint.normal, lineDirection));
            PdfEye[eyePathLength] = pdfonce * PdfEye[eyePathLength - 1];
        }

        for (int i = 0; i <= path.length - 2; i++)
        {
            vec3f connect_path = path.vertexs[i].position - path.vertexs[i+1].position;
            pdf +=PdfEye[i]*PdfLight[path.length-i-2]* dot(connect_path, connect_path);
            //pdf+= pdfCompute(path, path.length - i - 1);
            /*printf("diff with %e and %e \n", x, y);*/
        }

        return pdf;
    }

    __forceinline__ __device__ vec3f ContriConnect(const BDPTPath& path,int EyePathLength,int lightLength)
    {
        vec3f throughput = vec3f(1.0f);

        for (int i = EyePathLength-1; i <= EyePathLength; i++)
        {
            const BDPTVertex& midPoint = path.vertexs[i];
            const BDPTVertex& lastPoint = path.vertexs[i - 1];
            const BDPTVertex& nextPoint = path.vertexs[i + 1];
            vec3f lastDirection = normalize(lastPoint.position - midPoint.position);
            vec3f nextDirection = normalize(nextPoint.position - midPoint.position);
            throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection)) * Eval(*midPoint.mat, midPoint.normal, -lastDirection, nextDirection, midPoint.ext);

            if (lightLength == 1) {
                throughput *= abs(dot(nextPoint.normal, -nextDirection));
                break;
            }
        }
        return throughput;
    }

    __forceinline__ __device__ float pdfCompute(const BDPTPath& path, int lightPathLength)
    {
        int eyePathLength = path.length - lightPathLength;
        float pdf = 1.0f;
        const float RR_RATE = 0.8f;
        const int RR_BEGIN_DEPTH = 2;
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
            pdf *= abs(dot(midPoint.normal, lineDirection));
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