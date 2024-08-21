#include "PBB.h"
#include <stb_image.h>
#include <CL/cl.h>
#include "builderCL.h"
#include <stb_image_write.h>
#include <iostream>


void pbb_postprocess(std::string image)
{
    int width, height, channels;
    int mipLevel = 5;


    stbi_uc* image_data = stbi_load("1.jpg", &width, &height, &channels, 0);
    std::vector<stbi_uc*> mip;
    for (uint32_t i = 0; i < mipLevel; ++i) {
       mip.push_back(new stbi_uc[(width/((2*(i+1)))) * (height/((2 * (i + 1)))) *channels]);
    }
    stbi_uc* finalImage = new stbi_uc[(width) * (height) * channels];
    if (image_data == NULL) {
        fprintf(stderr, "Error loading image\n");
    }


    // Inicjalizacja OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    clInit::CLInitializeInput input = {};
    input.platform = &platform;
    input.device = &device;
    input.context = &context;
    input.queue = &queue;

    clInit::Initialize(input);

    cl_mem image_buffer;
    cl_mem final_image_buffer;

    clInit::CLCreateBufferInput<stbi_uc> bufferInput = {};
    bufferInput.buffer = &image_buffer;
    bufferInput.context = context;
    bufferInput.queue = queue;
    bufferInput.size = width * height * channels;
    bufferInput.data = image_data;
    bufferInput.readOnly = true;
    clInit::CreateBuffer(bufferInput);
    bufferInput.readOnly = false;
    bufferInput.buffer = &final_image_buffer;
    bufferInput.data = nullptr;
    clInit::CreateBuffer(bufferInput);
    std::vector<cl_mem> mipBuffer;
    
    for (size_t i = 0; i < mipLevel; ++i)
    {
        mipBuffer.push_back(cl_mem());
    }
    size_t sizebuffer = 1;
    for (cl_mem buffer : mipBuffer) {
        bufferInput.size = width/(2*sizebuffer) * height/(2*sizebuffer) * channels;
        bufferInput.buffer = &buffer;
        clInit::CreateBuffer(bufferInput);
        ++sizebuffer;
    }
    


    cl_int err;


    cl_program down_scale;
    cl_program up_scale;

    clInit::CLCreateProgramInput programInfo = {};
    programInfo.context = context;
    programInfo.code = "downScale.CL";
    programInfo.program = &down_scale;
    programInfo.device = device;
    clInit::CreateProgram(programInfo);
    programInfo.code = "upScale.CL";
    programInfo.program = &up_scale;
    clInit::CreateProgram(programInfo);



    cl_kernel down_scale_kernel;
    cl_kernel up_scale_kernel;

    clInit::CreateKernel(&down_scale_kernel, down_scale);
    clInit::CreateKernel(&up_scale_kernel, up_scale);

    int *widthMip = new int[mipLevel];
    int* heightMip = new int[mipLevel];

    for (int i = 0; i < mipLevel; ++i) {
        widthMip[i] = width / (2 * (i + 1));
        heightMip[i] = height / (2 * (i + 1));
    }
    size_t global_work_size[2] = { widthMip[0],heightMip[0] };

    clSetKernelArg(down_scale_kernel, 0, sizeof(cl_mem), &image_buffer);
    clSetKernelArg(down_scale_kernel, 1, sizeof(cl_mem), &mipBuffer[0]);
    clSetKernelArg(down_scale_kernel, 2, sizeof(int), &widthMip[0]);
    clSetKernelArg(down_scale_kernel, 3, sizeof(int), &heightMip[0]);
    std::cout << global_work_size[0] << std::endl;
    std::cout << global_work_size[1] << std::endl;
    clEnqueueNDRangeKernel(queue, down_scale_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    /*
    for (size_t i = 0; i < mipLevel; ++i) {
        size_t global_work_size[2] = { widthMip[i],heightMip[i]};
        clSetKernelArg(down_scale_kernel, 2, sizeof(int), &widthMip[i]);
        clSetKernelArg(down_scale_kernel, 3, sizeof(int), &heightMip[i]);
        if (i == 0) {
            clSetKernelArg(down_scale_kernel, 0, sizeof(cl_mem), &image_buffer);
            clSetKernelArg(down_scale_kernel, 1, sizeof(cl_mem), &mipBuffer[i]);

            clEnqueueNDRangeKernel(queue, down_scale_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
            clFinish(queue);
        }
        else {

            clSetKernelArg(down_scale_kernel, 0, sizeof(cl_mem), &mipBuffer[i - 1]);
            clSetKernelArg(down_scale_kernel, 1, sizeof(cl_mem), &mipBuffer[i]);
            
            clEnqueueNDRangeKernel(queue, down_scale_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
            clFinish(queue);
        }
        
    }
    */

    

    //err = clEnqueueReadBuffer(queue, final_image_buffer, CL_TRUE, 0, widthMip[0] * heightMip[0] * channels, finalImage, 0, NULL, NULL);

    //if (err != CL_SUCCESS) {
      //  fprintf(stderr, "Error reading image data from buffer\n");
   // }

    // Zapisz przetworzony obraz
   // stbi_write_png("DownScale.jpg", widthMip[0], heightMip[0], channels, mipBuffer[0], widthMip[0] * channels);

    // Zwolnij zasoby
    clReleaseMemObject(image_buffer);
    clReleaseMemObject(final_image_buffer);
    clReleaseKernel(down_scale_kernel);
    clReleaseProgram(down_scale);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    stbi_image_free(image_data);
    stbi_image_free(finalImage);
    for (stbi_uc *image : mip) stbi_image_free(image);
    for (cl_mem buffer :mipBuffer) clReleaseMemObject(buffer);
    delete widthMip;
    delete heightMip;
}

