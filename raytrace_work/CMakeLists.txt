# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(OpenGL REQUIRED)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

cuda_add_library(toneMap
  toneMap.cu)
add_executable(raytrace_work
  ${embedded_ptx_code}
  devicePrograms.cu
  toneMap.cu
  config.h
  CUDABuffer.h
  BDPT.h
  BDPT_function.h
  LaunchParams.h
  LightParams.h
  Model.h
  optix7.h
  PRD.h
  SampleRenderer.h
  tool_function.h
  SBTdata.h
  main.cpp
  Model.cpp
  SampleRenderer.cpp
  )

target_link_libraries(raytrace_work
  toneMap
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  # glfw and opengl, for display
  glfWindow
  glfw
  ${OPENGL_gl_LIBRARY}
  )
