# 图形学大作业——光线跟踪类

2021&2022年春季学期计算机图形学课程大作业

报告视频详见[北大网盘](https://disk.pku.edu.cn:443/link/49FE386806470FF4C5BEAC990718B0BF)
具体项目文件请见:[Github仓库](https://github.com/Tim-1e/optix7_advance_PKUcourse)

> :warning: **Warning**
> 
请确保您cmake时使用的cuda与optix版本,CUDA 10.1(10.x,切勿使用cuda11), OptiX 7 SDK(7.x)
> 
## 作业构成

本作业以*optix7course*的代码框架作为基础，实现了*相机控制*,*optix管线搭建*,*材质参数输入重构*,*蒙特卡洛光线追踪*,*折射/反射材质光追*

+ `common/`为封装好的头文件和第三方库
    - `3rdParty`包括`stb_image`图形加载库,`tiny_obj_loader`模型加载库,`glfw`的opengl渲染库
    - `gdt`为`GPU_Development_Tools`库,用于调用`optix`,使用`optix`等相关函数及流水线
    - `glfWindow`为交互窗口库,在其中我们实现了鼠标及键盘交互
+ `models/`为用到的模型，`textures/`为用到的纹理
+ `raytrace_work/`包含主程序,流水线与场景渲染程序,光线追踪程序
    - `main`为主程序,我们在其中利用`glfWindow`继承的虚函数完成鼠标及键盘交互功能,并且利用opengl库完成渲染工作
    - `Model`为模型加载,完成了相关材质,顶点,网格的加载
    - `SampleRenderer`完成了pipeline的搭建,及利用cuda实现的GPU相关数据导入
    - `devicePrograms.cu`为主要的光线追踪程序,在其中完成`__raygen__renderFrame`,`__closesthit__radiance`,`__miss__radiance`,完成了*蒙特卡洛光线追踪*,*折射/反射材质光追*的具体工作

## 基本功能

+ 使用WASD移动摄像机，使用鼠标滑动控制摄像机方向
+ 在sponza场景下感受optix7带来的强大实时交互的光追程序
+ 根据命令行提示自由的调整移速,光追量,降噪等参数

## 编译

代码使用**CMake**配置，请使用**Cmake**生成可执行文件
