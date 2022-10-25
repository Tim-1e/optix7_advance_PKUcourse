#pragma once
#include <vector>
#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;
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
}