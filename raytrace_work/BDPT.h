#pragma once
#include "gdt/math/vec.h"
#include "config.h"
#include "SBTdata.h"

namespace osc {
    using namespace gdt;
    struct M_extansion {
        vec3f diffuseColor;
        vec3f specColor;
    };
    struct BDPTVertex
    {
        vec3f position;
        vec3f normal;
        TriangleMeshSBTData mat;
        M_extansion ext;
        int MeshID;//三角形面片id
        float pdf = 0;

        inline __both__ void init(vec3f _position, vec3f _normal, TriangleMeshSBTData _mat, M_extansion _ext,int _MeshID) {
            position=_position; 
            normal = _normal; 
            mat = _mat; 
            ext = _ext;
            MeshID = _MeshID;
        }
        inline __both__ void init(vec3f _position, vec3f _normal, int _MeshID) {
            position = _position;
            normal = _normal;
            MeshID = _MeshID;
        }
        inline __both__ void init(vec3f _position) {
            position = _position;
        }
    };
    struct BDPTPath {
        BDPTVertex *vertexs;
        int length;
    };
    
}