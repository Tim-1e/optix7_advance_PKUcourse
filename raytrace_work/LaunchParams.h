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
namespace osc {
  using namespace gdt;

  // for this simple example, we have a single ray type
  enum { RADIANCE_RAY_TYPE=0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };
  enum { LIGHT_GENERATE = 0, EYE_GENERATE,GENERATE_COUNT };
  struct TriangleMeshSBTData {
      vec3f  color;
      vec3f* vertex;
      vec3f* normal;
      vec2f* texcoord;
      vec3i* index;

      bool  hasTexture;
      int ID;//表明物体编号
      vec3f emission;
      bool emissive_;
      float d;//refractable
      float Kr;//refraction rate
      float alpha_; // shininess constant
      float subsurface;//次表面，控制漫反射形状
      float roughness;//粗糙度，影响漫反射和镜面反射 
      float metallic; //金属度，规定电介质为0，金属为1；
       //当值趋向1时：弱化漫反射比率，强化镜面反射强度，同时镜面反射逐渐附带上金属色
       //半导体材质情况特殊，尽量避免使用半导体调试效果
      float sheen;//光泽度，一种额外的掠射分量，一般用于补偿布料在掠射角下的光能  
      float sheenTint;//光泽色，控制sheen的颜色
      float specular;//高光强度(镜面反射强度)
       //控制镜面反射光占入射光的比率，用于取代折射率
      float specularTint;//高光染色，和baseColor一起，控制镜面反射的颜色
      //注意，这是非物理效果，且掠射镜面反射依然是非彩色
      float clearcoat;//清漆强度，控制第二个镜面反射波瓣形成及其影响范围
      float clearcoatGloss;//清漆光泽度，控制透明涂层的高光强度（光泽度）
      //规定缎面(satin)为0，光泽(gloss)为1；
      cudaTextureObject_t texture;
      bool hasSpecTexture;
      cudaTextureObject_t spectexture;
  };
  
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

    void* eyePath; 
    void*  lightPath;
    void*  connectPath;
    int* lightPathNum;
    OptixTraversableHandle traversable;
  };
  
} // ::osc
