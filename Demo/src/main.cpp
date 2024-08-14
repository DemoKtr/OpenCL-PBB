#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define VECTOR_SIZE 4

// OpenCL kernel kodu
const char* kernelSource = R"(
    __kernel void process_image(__global uchar *image,
                                const int width,
                                const int height) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x < width && y < height) {
            int idx = (y * width + x) * 3; // Zakładamy obraz RGB
            image[idx] = 255 - image[idx]; // Inwersja koloru czerwonego
            image[idx + 1] = 255 - image[idx + 1]; // Inwersja koloru zielonego
            image[idx + 2] = 255 - image[idx + 2]; // Inwersja koloru niebieskiego
        }
    }
)";

int main() {

    int width, height, channels;
    stbi_uc* image_data = stbi_load("wp.jpg", &width, &height, &channels, 0);
    if (image_data == NULL) {
        fprintf(stderr, "Error loading image\n");
        return -1;
    }

    // Inicjalizacja danych
    float a[VECTOR_SIZE], b[VECTOR_SIZE], c[VECTOR_SIZE];
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = (i+1);
    }

    // Inicjalizacja OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_mem image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * channels, NULL, NULL);


    cl_int err;
    err = clEnqueueWriteBuffer(queue, image_buffer, CL_TRUE, 0, width * height * channels, image_data, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error copying image data to buffer\n");
        return -1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "process_image", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &width);
    clSetKernelArg(kernel, 2, sizeof(int), &height);

    size_t global_work_size[2] = { width, height };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, image_buffer, CL_TRUE, 0, width * height * channels, image_data, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading image data from buffer\n");
        return -1;
    }

    // Zapisz przetworzony obraz
    stbi_write_png("output.png", width, height, channels, image_data, width * channels);

    // Zwolnij zasoby
    clReleaseMemObject(image_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    stbi_image_free(image_data);


    return 0;
}
