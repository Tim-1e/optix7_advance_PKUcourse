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
        BDPTPath* path;
        bool end;
        bool TouchtheLight;
    };
}

