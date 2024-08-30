#include "Toon.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "./../builderCL.h"

void toon_postprocess(std::string image, std::string output)
{
    int width, height, channels;
    stbi_uc* image_data = stbi_load(image.c_str(), &width, &height, &channels, 0);
    int levels = 15;
    stbi_uc* finalImage = new stbi_uc[width * height * channels];
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
    cl_mem gray_buffer;
    cl_mem edge_buffer;
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
    bufferInput.buffer = &gray_buffer;
    bufferInput.size = width * height;
    clInit::CreateBuffer(bufferInput);
    bufferInput.buffer = &edge_buffer;
    clInit::CreateBuffer(bufferInput);

    cl_int err;


    cl_program gray_scale_program;
    cl_program edges_scale_program;
    cl_program toon_scale_program;

    clInit::CLCreateProgramInput programInfo = {};
    programInfo.context = context;
    programInfo.code = "grayScale.CL";
    programInfo.program = &gray_scale_program;
    programInfo.device = device;
    clInit::CreateProgram(programInfo);
    programInfo.code = "edge.CL";
    programInfo.program = &edges_scale_program;
    clInit::CreateProgram(programInfo);
    programInfo.code = "toon.CL";
    programInfo.program = &toon_scale_program;
    clInit::CreateProgram(programInfo);




    cl_kernel gray_kernel;
    cl_kernel edges_kernel;
    cl_kernel toon_kernel;
    clInit::CreateKernel(&gray_kernel, gray_scale_program);
    clInit::CreateKernel(&edges_kernel, edges_scale_program);
    clInit::CreateKernel(&toon_kernel, toon_scale_program);


    clSetKernelArg(gray_kernel, 0, sizeof(cl_mem), &image_buffer);
    clSetKernelArg(gray_kernel, 1, sizeof(cl_mem), &gray_buffer);
    clSetKernelArg(gray_kernel, 2, sizeof(int), &width);
    clSetKernelArg(gray_kernel, 3, sizeof(int), &height);

    clSetKernelArg(edges_kernel, 0, sizeof(cl_mem), &gray_buffer);
    clSetKernelArg(edges_kernel, 1, sizeof(cl_mem), &edge_buffer);
    clSetKernelArg(edges_kernel, 2, sizeof(int), &width);
    clSetKernelArg(edges_kernel, 3, sizeof(int), &height);

    clSetKernelArg(toon_kernel, 0, sizeof(cl_mem), &image_buffer);
    clSetKernelArg(toon_kernel, 1, sizeof(cl_mem), &edge_buffer);
    clSetKernelArg(toon_kernel, 2, sizeof(cl_mem), &final_image_buffer);
    clSetKernelArg(toon_kernel, 3, sizeof(int), &levels);
    clSetKernelArg(toon_kernel, 4, sizeof(int), &width);
    clSetKernelArg(toon_kernel, 5, sizeof(int), &height);


    size_t global_work_size[2] = { width, height };

    clEnqueueNDRangeKernel(queue, gray_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueNDRangeKernel(queue, edges_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueNDRangeKernel(queue, toon_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, final_image_buffer, CL_TRUE, 0, width * height * channels, finalImage, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading image data from buffer\n");
    }

    // Zapisz przetworzony obraz
    stbi_write_png(output.c_str(), width, height, channels, finalImage, width * channels);

    // Zwolnij zasoby
    clReleaseMemObject(image_buffer);
    clReleaseMemObject(gray_buffer);
    clReleaseMemObject(final_image_buffer);
    //clReleaseKernel(kernel);
    clReleaseKernel(gray_kernel);
    clReleaseKernel(edges_kernel);
    clReleaseKernel(toon_kernel);
    // clReleaseProgram(program);
    clReleaseProgram(gray_scale_program);
    clReleaseProgram(edges_scale_program);
    clReleaseProgram(toon_scale_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    stbi_image_free(image_data);
    stbi_image_free(finalImage);
}

