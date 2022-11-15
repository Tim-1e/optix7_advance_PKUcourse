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
        vec3f  sourcePos;
        vec3f direction;
        vec3f normal;
        vec3f lightColor;
        vec3f throughout;
        vec3f weight;
        vec3f Eval;
        vec3f xx, yy;
        int zz;
        BDPTPath* path;
        bool end;
        bool TouchtheLight;
        //vec3f  pixelColor;
        //vec3f  pixelNormal;
        //vec3f  pixelAlbedo;
        //float refraction_index;//当前光线相对于空气的折射率
    };
}

