#pragma once
#include "gdt/math/vec.h"
#include "gdt/random/random.h"
#include "BDPT.h"

namespace osc {
    using namespace gdt;
    typedef LCG<16> Random;

    struct PRD {
        Random random;
        int depth;
        vec3f throughout;
        vec3f  pixelColor;
        vec3f  pixelNormal;
        vec3f  pixelAlbedo;
        vec3f  sourcePos;
        vec3f nextPosition;
        vec3f weight;
        bool end;
        BDPTVertex* vertexs;

        //float refraction_index;//当前光线相对于空气的折射率
    };
}

