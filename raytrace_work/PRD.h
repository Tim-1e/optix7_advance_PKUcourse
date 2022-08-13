#pragma once
#include "gdt/random/random.h"
#include "LaunchParams.h"

namespace osc {
    typedef gdt::LCG<16> Random;

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
}

