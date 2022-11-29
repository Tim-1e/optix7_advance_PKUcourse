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
      load_GT();
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
      float mse = cal_mse();
      mse_file << mse << std::endl;

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
                if (DOWNLOAD_RAW && i >  4) {
                    std::cout << "Saving rawFrame..." << std::endl;
                    std::ofstream myFile;
                    myFile.open(file_dir.append(".rawFrame"), std::ios::out);
                    for (auto p : raw_pixels) {
                        myFile << p.x << " " << p.y << " " << p.z << std::endl;
                    }
                    myFile.close();
                }
            }
        }
    }

    void load_GT() {
        if (GT_ENABLE) {
            std::ifstream myFile;
            myFile.open(GT_DIR, std::ios::in);
            for (auto p : ground_truth) {
                myFile >> p.x >> p.y >> p.z ;
            }
            myFile.close();

            mse_file.open(std::string(DOWNLOAD_DIR).append("mseList"), std::ios::app);
        }
    }

    float pow2(float x) { return x * x; }

    float cal_mse() {
        double sum = 0;
        if (GT_ENABLE) {
            int size1 = raw_pixels.size();
            int size2 = ground_truth.size();
            if (size1 != size2) {
                std::cout << GDT_TERMINAL_RED << "FATAL ERROR:  ground truth "<< size2
                    <<" and rendered image "<< size1 <<" don't match size" << GDT_TERMINAL_DEFAULT << std::endl;
                exit(1);
            }
            for (int i = 0; i < size1; ++i) {
                sum += pow2(raw_pixels[i].x - ground_truth[i].x);
                sum += pow2(raw_pixels[i].y - ground_truth[i].y);
                sum += pow2(raw_pixels[i].z - ground_truth[i].z);
            }
            sum = sum / (3 * size1);
        }
        return float(sum);
    }

    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
      raw_pixels.resize(newSize.x * newSize.y);
      ground_truth.resize(newSize.x * newSize.y);
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
    std::vector<float4> ground_truth;
    std::ofstream mse_file;

    struct timeStruct
    {
#define TIME_LEN 10
        int startTime;
        const int len = TIME_LEN;
        bool timeFlag[TIME_LEN] = { 0 };
        std::string timeStampStr[TIME_LEN] = { "3s", "10s", "30s", "60s", "180s", "300s", "600s", "1200s", "2400s", "3600s"};
        int timeStamp[TIME_LEN] = { 3000, 10000, 30000, 60000, 180000, 300000, 600000, 1200000, 2400000, 3600000 };
    } myTime;
  };

  extern "C" int main(int ac, char **av)
  {
    try {
        
        // the light 
        std::vector<LightParams> All_Lights;

        Model* model = loadOBJ(
            "../../models/threemirror.obj", All_Lights
        );
        Camera threemirror_camera = { /*from*/vec3f(7.55735,11.741,-6.49642),
            /* at */vec3f(-163.544,-301.196,172.938),
            /* up */vec3f(0.f,1.f,0.f) };
        Camera onemirror_camera = { /*from*/vec3f(-2.41854, 9.48831, 5.85088),
            /* at */vec3f(145.705, -242.668, -265.953),
            /* up */vec3f(0.f,1.f,0.f) };
        Camera camera = threemirror_camera;

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
