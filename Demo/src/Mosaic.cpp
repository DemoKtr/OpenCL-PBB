#include "Mosaic.h"
#include <stb_image.h>
#include <iostream>
#include "builderCL.h"
#include <stb_image_write.h>
#include <random>
void mosaic_effect(std::string image, std::string output, int step,int rand) {

        int width, height, channels;
        stbi_uc* image_data = stbi_load(image.c_str(), &width, &height, &channels, 0);
        stbi_uc* finalImage = new stbi_uc[width * height * channels];
        if (image_data == NULL) {
            fprintf(stderr, "Error loading image\n");
        }
        int realWidth = width / step;
        int realHeight = height / step;
        int realStep = step;
        int* radomValues = new int[realWidth * realHeight];
        std::random_device rd; // uzyskaj losowe ziarno z urz¹dzenia
        std::mt19937 gen(rd()); // u¿yj Mersenne Twister jako generatora

        // Utwórz rozk³ad liczbowy w zadanym zakresie
        std::uniform_int_distribution<> distrib(-rand, rand);

        for (uint32_t i = 0; i < realWidth * realHeight; ++i) {
            radomValues[i] = distrib(gen);

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
        cl_mem random_values_buffer;
        cl_mem final_image_buffer;

        cl_int err;
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

        clInit::CLCreateBufferInput<int> randomBufferInput = {};
        randomBufferInput.buffer = &random_values_buffer;
        randomBufferInput.context = context;
        randomBufferInput.queue = queue;
        randomBufferInput.size = realHeight * realWidth * sizeof(int);
        randomBufferInput.data = radomValues;
        randomBufferInput.readOnly = true;
        clInit::CreateBuffer(randomBufferInput);

        cl_program mosaic;


        clInit::CLCreateProgramInput programInfo = {};
        programInfo.context = context;
        programInfo.code = "mosaic.CL";
        programInfo.program = &mosaic;
        programInfo.device = device;
        clInit::CreateProgram(programInfo);


        cl_kernel mosaic_kernel;

        clInit::CreateKernel(&mosaic_kernel, mosaic);

        clSetKernelArg(mosaic_kernel, 0, sizeof(cl_mem), &image_buffer);
        clSetKernelArg(mosaic_kernel, 1, sizeof(cl_mem), &final_image_buffer);
        clSetKernelArg(mosaic_kernel, 2, sizeof(cl_mem), &random_values_buffer);
        clSetKernelArg(mosaic_kernel, 3, sizeof(int), &realStep);
        clSetKernelArg(mosaic_kernel, 4, sizeof(int), &realWidth);
        clSetKernelArg(mosaic_kernel, 5, sizeof(int), &realHeight);
        clSetKernelArg(mosaic_kernel, 6, sizeof(int), &width);
        clSetKernelArg(mosaic_kernel, 7, sizeof(int), &height);
        ;

        size_t global_work_size[2] = { width, height};

        err = clEnqueueNDRangeKernel(queue, mosaic_kernel,2, NULL, global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error execute kernel\n");
        }
        clFinish(queue);

        
        err = clEnqueueReadBuffer(queue, final_image_buffer, CL_TRUE, 0, width * height * channels, finalImage, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading image data from buffer\n");
        }
        stbi_write_png(output.c_str(), width, height, channels, finalImage, width * channels);

        clReleaseMemObject(image_buffer);
        clReleaseMemObject(random_values_buffer);
        clReleaseMemObject(final_image_buffer);
        clReleaseKernel(mosaic_kernel);
        clReleaseProgram(mosaic);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        stbi_image_free(image_data);
        stbi_image_free(finalImage);
        delete[] radomValues;


    }

