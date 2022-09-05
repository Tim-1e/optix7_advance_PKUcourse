#pragma once
#include "gdt/math/vec.h"
#include "LaunchParams.h"
#include "PRD.h"

namespace osc {
    struct BDPTVertex
    {
        vec3f position;
        vec3f normal;
        TriangleMeshSBTData mat;
        M_extansion ext;
        float pdf;
        BDPTVertex(vec3f _position, vec3f _normal, TriangleMeshSBTData _mat, M_extansion _ext):
            position(_position),normal(_normal),mat(_mat),ext(_ext){};
    };
    
}