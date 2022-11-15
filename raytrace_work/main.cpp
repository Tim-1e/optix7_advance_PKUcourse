//the main render library
#include "SampleRenderer.h"

// program configuration
#include "config.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace osc {

  struct SampleWindow : public GLFCameraWindow
  {
    SampleWindow(const std::string &title,
                 const Model *model,
                 const Camera &camera,
                 const  std::vector<LightParams> light,
                 const float worldScale)
      : GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale, VISIBLE_MOUSE),
        sample(model,light)
    {
      sample.setCamera(camera);
      cameraFrame.motionSpeed=5.0f;
      myTime.startTime = clock();
    }
    
    virtual void render() override
    {
      if (cameraFrame.modified && !FIXED_CAMERA) {
        sample.setCamera(Camera{ cameraFrame.get_from(),
                                 cameraFrame.get_at(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
      }
      sample.render();
    }
    
    virtual void draw() override
    {
      sample.downloadPixels(pixels.data(), raw_pixels.data());

      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA;
      GLenum texelType = GL_UNSIGNED_BYTE;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();

      // Warning: this function will reverse vector "pixels" ! 
      if (DOWNLOAD) savePicture();
    }
    
    void savePicture() 
    {
        std::reverse(pixels.begin(), pixels.end());
        int currentTime = clock();
        int deltaTime = currentTime - myTime.startTime;
        for (int i = 0; i < myTime.len; ++i) {
            if (!myTime.timeFlag[i] && deltaTime > myTime.timeStamp[i]) {
                myTime.timeFlag[i] = true;
                std::string file_dir = std::string(DOWNLOAD_DIR).append(myTime.timeStampStr[i]);
                std::string file_dir_1 = file_dir;
                stbi_write_png(file_dir_1.append(".png").c_str(),
                    fbSize.x, fbSize.y, 4, pixels.data(), fbSize.x * sizeof(uint32_t));
                std::cout << "Saving picture " << myTime.timeStampStr[i] << std::endl;
                if (DOWNLOAD_RAW) {
                    std::ofstream myFile;
                    myFile.open(file_dir.append(".rawFrame"), std::ios::out);
                    for (auto p : raw_pixels) {
                        myFile << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;
                    }
                    myFile.close();
                }
                
            }
        }
    }

    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
      raw_pixels.resize(newSize.x * newSize.y);
    }

    virtual void key(int key, int mods)
    {
      if (key == 'Z' || key == 'z') {
        sample.denoiserOn = !sample.denoiserOn;
        std::cout << "denoising now " << (sample.denoiserOn?"ON":"OFF") << std::endl;
      }
      if (key == 'X' || key == 'x') {
        sample.accumulate = !sample.accumulate;
        std::cout << "accumulation/progressive refinement now " << (sample.accumulate?"ON":"OFF") << std::endl;
      }
      if (key == 'e' || key == 'E') {
          sample.move_available = !sample.move_available;
          std::cout << "Move mode now is" << sample.move_available << std::endl;
      }
      if (key == ',') {
        sample.launchParams.numPixelSamples
          = std::max(1,sample.launchParams.numPixelSamples-1);
        std::cout << "num samples/pixel now "
                  << sample.launchParams.numPixelSamples << std::endl;
      }
      if (key == '.') {
        sample.launchParams.numPixelSamples
          = std::max(1,sample.launchParams.numPixelSamples+1);
        std::cout << "num samples/pixel now "
                  << sample.launchParams.numPixelSamples << std::endl;
      }
      if (key=='q' || key=='Q') {
          std::cout << "(C)urrent camera:" << std::endl;
          std::cout << "- from :" << cameraFrame.position << std::endl;
          std::cout << "- poi  :" << cameraFrame.getPOI() << std::endl;
          std::cout << "- upVec:" << cameraFrame.upVector << std::endl;
          std::cout << "- frame:" << cameraFrame.frame << std::endl;
          std::cout << "- lookat:" << cameraFrame.get_at() << std::endl;
      }

      if (key == 334) {
          cameraFrame.motionSpeed += 1.0f;
          std::cout<<"Speed up now the speed is"<<cameraFrame.motionSpeed<<std::endl;
      }
      if (key == 333) {
          cameraFrame.motionSpeed /= 2.0f;
          std::cout << "Speed down now the speed is" << cameraFrame.motionSpeed << std::endl;
      }
    }
    
    virtual void processInput() {
        if (glfwGetKey(handle, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(handle, true);
        if (sample.move_available)
            return;
        if (glfwGetKey(handle, GLFW_KEY_W) == GLFW_PRESS) {
            cameraFrame.position -= cameraFrame.frame.vz * cameraFrame.motionSpeed;
            cameraFrame.modified = true;
        }
        if (glfwGetKey(handle, GLFW_KEY_S) == GLFW_PRESS) {
            cameraFrame.position += cameraFrame.frame.vz * cameraFrame.motionSpeed;
            cameraFrame.modified = true;
        }
        if (glfwGetKey(handle, GLFW_KEY_A) == GLFW_PRESS) {
            cameraFrame.position -= cameraFrame.frame.vx * cameraFrame.motionSpeed;
            cameraFrame.modified = true;
        }
        if (glfwGetKey(handle, GLFW_KEY_D) == GLFW_PRESS) {
            cameraFrame.position += cameraFrame.frame.vx * cameraFrame.motionSpeed;
            cameraFrame.modified = true;
        }
        if (glfwGetKey(handle, GLFW_KEY_SPACE) == GLFW_PRESS) {
            cameraFrame.position += cameraFrame.get_up() * cameraFrame.motionSpeed;
            cameraFrame.modified = true;
        }
        if (glfwGetKey(handle, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            cameraFrame.position -= cameraFrame.get_up() * cameraFrame.motionSpeed;
            cameraFrame.modified = true;
        }
    };

    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
    std::vector<float4> raw_pixels;
    struct timeStruct
    {
#define TIME_LEN 9
        int startTime;
        const int len = TIME_LEN;
        bool timeFlag[TIME_LEN] = { 0 };
        std::string timeStampStr[TIME_LEN] = { "3s", "10s", "30s", "60s", "180s", "300s", "600s", "1200s", "2400s"};
        int timeStamp[TIME_LEN] = { 3000, 10000, 30000, 60000, 180000, 300000, 600000, 1200000, 2400000 };
    } myTime;
  };

  extern "C" int main(int ac, char **av)
  {
    try {
        
        // the light 
        std::vector<LightParams> All_Lights;

        Model* model = loadOBJ("../../models/sponza2.obj" , All_Lights);
        Camera camera;
        switch (CAMERA_SET)
        {
        case 0:
        default:
            camera = { /*from*/vec3f(-1293.07f, 154.681f, -0.7304f),
            /* at */model->bounds.center() - vec3f(0,400,0),
            /* up */vec3f(0.f,1.f,0.f) };
            break;
        case 1:
             camera = { vec3f(-994.395, 890.632, 161.383),
                vec3f(42.0908, 398.834, -371.494),
                vec3f(0.f, 1.f, 0.f)};
            break;
        }
        


       //Model *model = loadOBJ("../../models/easy.obj", All_Lights); 
       //Camera camera = { /*from*/vec3f(5.3196f,7.24334f,5.39261f),
       //                               /* at */vec3f(-224.429f,-202.79f,-244.616f),
       //                               /* up */vec3f(0.f,1.f,0.f) };

      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:

      const float worldScale = length(model->bounds.span());

      SampleWindow *window = new SampleWindow("BDPT",
                                              model, camera, All_Lights, worldScale);
      window->enableFlyMode();
      
      std::cout << "Press 'Z' to enable/disable accumulation/progressive refinement" << std::endl;
      std::cout << "Press 'X' to enable/disable denoising" << std::endl;
      std::cout << "Press ',' to reduce the number of paths/pixel" << std::endl;
      std::cout << "Press '.' to increase the number of paths/pixel" << std::endl;
      std::cout << "Press 'Q' to get your camera position" << std::endl;

      // begin the rendering
      window->run();
      
    } catch (std::runtime_error& e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
	  exit(1);
    }
    return 0;
  }
  
} 
