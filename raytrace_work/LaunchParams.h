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
      int ID;//����������
      vec3f emission;
      bool emissive_;
      float d;//refractable
      float Kr;//refraction rate
      float alpha_; // shininess constant
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
      //ע�⣬���Ƿ�����Ч���������侵�淴����Ȼ�Ƿǲ�ɫ
      float clearcoat;//����ǿ�ȣ����Ƶڶ������淴�䲨���γɼ���Ӱ�췶Χ
      float clearcoatGloss;//�������ȣ�����͸��Ϳ��ĸ߹�ǿ�ȣ�����ȣ�
      //�涨����(satin)Ϊ0������(gloss)Ϊ1��
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
