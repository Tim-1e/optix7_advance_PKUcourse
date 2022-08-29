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
      float subsurface;//�α��棬������������״
      float roughness;//�ֲڶȣ�Ӱ��������;��淴�� 
      float metallic; //�����ȣ��涨�����Ϊ0������Ϊ1��
       //��ֵ����1ʱ��������������ʣ�ǿ�����淴��ǿ�ȣ�ͬʱ���淴���𽥸����Ͻ���ɫ
       //�뵼�����������⣬��������ʹ�ð뵼�����Ч��
      float sheen;//����ȣ�һ�ֶ�������������һ�����ڲ���������������µĹ���  
      float sheenTint;//����ɫ������sheen����ɫ
      float specular;//�߹�ǿ��(���淴��ǿ��)
      //���ƾ��淴���ռ�����ı��ʣ�����ȡ��������
      float specularTint;//�߹�Ⱦɫ����baseColorһ�𣬿��ƾ��淴�����ɫ
      float clearcoat;//����ǿ�ȣ����Ƶڶ������淴�䲨���γɼ���Ӱ�췶Χ
      float clearcoatGloss;//�������ȣ�����͸��Ϳ��ĸ߹�ǿ�ȣ�����ȣ�
      //�涨����(satin)Ϊ0������(gloss)Ϊ1��
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
