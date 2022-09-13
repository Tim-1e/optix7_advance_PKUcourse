#pragma once
#include "gdt/math/vec.h"
#include "gdt/random/random.h"

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
        int MeshId;
        int PrimId;
        //std::vector<BDPTVertex> Vertexs;
        //float refraction_index;//当前光线相对于空气的折射率
    };
    struct M_extansion {
        vec3f diffuseColor;
        vec3f specColor;
    };
}

