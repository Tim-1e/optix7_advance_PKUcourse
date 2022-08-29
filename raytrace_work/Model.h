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

#include "gdt/math/AffineSpace.h"
#include "LaunchParams.h"
#include <vector>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
      std::vector<vec3f> vertex;
      std::vector<vec3f> normal;
      std::vector<vec2f> texcoord;
      std::vector<vec3i> index;

      // material data:
      vec3f              diffuse;
      int                diffuseTextureID{ -1 };
      vec3f               spec = 0.f;
      int                specTextureID{ -1 };
      float               d = 1.f;//refractable
      float               Kr = 0.f;//refraction rate
      float alpha_ = 0.f; // shininess constant
      bool emissive_ = 0;
      vec3f emission = 0.f;
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
      float clearcoat;//清漆强度，控制第二个镜面反射波瓣形成及其影响范围
      float clearcoatGloss;//清漆光泽度，控制透明涂层的高光强度（光泽度）
      //规定缎面(satin)为0，光泽(gloss)为1；
  };

  struct QuadLight {
    vec3f origin, du, dv, power;
  };
  
  struct Texture {
    ~Texture()
    { if (pixel) delete[] pixel; }
    
    uint32_t *pixel      { nullptr };
    vec2i     resolution { -1 };
  };
  
  struct Model {
    ~Model()
    {
      for (auto mesh : meshes) delete mesh;
      for (auto texture : textures) delete texture;
    }
    
    std::vector<TriangleMesh *> meshes;
    std::vector<Texture *>      textures;
    //! bounding box of all vertices in the model
    box3f bounds;
  };

  Model *loadOBJ(const std::string &objFile, std::vector<LightParams>& lights);
}
