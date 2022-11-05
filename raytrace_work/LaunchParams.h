// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
#pragma once
#include <vector>
#include "gdt/math/vec.h"
#include "optix7.h"
#include "LightParams.h"
#include "config.h"
#include "SBTdata.h"
#include "BDPT.h"
namespace osc {
  using namespace gdt;

  // for this simple example, we have a single ray type
  enum { RADIANCE_RAY_TYPE=0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };
  enum { LIGHT_GENERATE = 0, EYE_GENERATE,GENERATE_COUNT };
  
  struct LaunchParams
  {
    int numPixelSamples = OneLightRayIterNum;
    int LightVertexNum = 0;
    struct {
      int       frameID = 0;
      float4   *colorBuffer;
      float4   *normalBuffer;
      float4   *albedoBuffer;
      
      /*! the size of the 1 buffer to render */
      vec2i     size;
    } frame;
    
    struct {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } camera;

    LightParams* All_Lights;
    int Lights_num;

    BDPTVertex* eyePath; 
    BDPTVertex* lightPath;
    BDPTVertex* connectPath;
    int* lightPathNum;
    TriangleMeshSBTData* matHeader;
    OptixTraversableHandle traversable;
  };
  
} // ::osc
