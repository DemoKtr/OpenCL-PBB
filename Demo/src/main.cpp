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


const char* gray = R"(
    __kernel void process_image(__global uchar *image, // Obraz RGB
                            __global uchar *gray_image, // Obraz szaro-skalowy
                            const int width,
                            const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // Indeks w obrazie RGB
        uchar r = image[idx];
        uchar g = image[idx + 1];
        uchar b = image[idx + 2];
        
        uchar gray = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);

        // Zapisywanie wartości szarości w nowym buforze
        gray_image[y * width + x] = gray;
    }
}


)";

const char* edges = R"(
__kernel void edges_detection(__global uchar *gray_image, // Obraz szaro-skalowy
                              __global uchar *final_image, // Obraz z krawędziami
                              const int width,
                              const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Kernel matrices for Sobel operator
        const int horizontal_kernel_matrix[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
        const int vertical_kernel_matrix[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
        
        float gradientX = 0.0f;
        float gradientY = 0.0f;
        
        for(int i = -1; i <= 1; ++i) {
            for(int j = -1; j <= 1; ++j) {
                int pixel_value = gray_image[(y + i) * width + (x + j)];
                gradientX += pixel_value * horizontal_kernel_matrix[(i + 1) * 3 + (j + 1)];
                gradientY += pixel_value * vertical_kernel_matrix[(i + 1) * 3 + (j + 1)];
            }
        }
        
        float edge_strength = sqrt(gradientX * gradientX + gradientY * gradientY);
        final_image[y * width + x] = (uchar)min(255.0f, edge_strength); // Clamping the value
    } else {
        final_image[y * width + x] = 0;
    }
}
)";


const char* toon = R"(
__kernel void toon_shading(__global uchar *original_image, // Obraz RGB
                           __global uchar *edge_image,
                           __global uchar *final_image, // Obraz z krawędziami
                           const int levels,
                           const int width,
                           const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int index = (y * width + x) * 3;

        // Odczyt krawędzi
        int edge = (int)edge_image[index/3];
        
        if (edge < 245) {
            // Konwersja kolorów z uchar do float
            float R = (float)original_image[index] / 255.0f;
            float G = (float)original_image[index + 1] / 255.0f;
            float B = (float)original_image[index + 2] / 255.0f;

            // Kwantyzacja kolorów
            R = floor(R * levels) / levels;
            G = floor(G * levels) / levels;
            B = floor(B * levels) / levels;

            // Konwersja z powrotem do uchar
            final_image[index] = (uchar)clamp(R * 255.0f, 0.0f, 255.0f);
            final_image[index + 1] = (uchar)clamp(G * 255.0f, 0.0f, 255.0f);
            final_image[index + 2] = (uchar)clamp(B * 255.0f, 0.0f, 255.0f);
        } else {
            // Ustawienie koloru czarnego
            final_image[index] = 0;
            final_image[index + 1] = 0;
            final_image[index + 2] = 0;
        }
    }
    
   
}
)";




int main() {

    int width, height, channels;
    stbi_uc* image_data = stbi_load("wp.jpg", &width, &height, &channels, 0);
    int levels = 20;
    stbi_uc* finalImage = new stbi_uc[width * height * channels];
    if (image_data == NULL) {
        fprintf(stderr, "Error loading image\n");
        return -1;
    }


    // Inicjalizacja OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_mem image_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * channels, NULL, NULL);
    cl_mem gray_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height, NULL, NULL);
    cl_mem edge_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height, NULL, NULL);
    cl_mem final_image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * channels, NULL, NULL);


    cl_int err;
    err = clEnqueueWriteBuffer(queue, image_buffer, CL_TRUE, 0, width * height * channels, image_data, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error copying image data to buffer\n");
        return -1;
    }

    //cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    cl_program gray_scale_program = clCreateProgramWithSource(context, 1, &gray, NULL, NULL);
    cl_program edges_scale_program = clCreateProgramWithSource(context, 1, &edges, NULL, NULL);
    cl_program toon_scale_program = clCreateProgramWithSource(context, 1, &toon, NULL, NULL);
    
   
   // clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    clBuildProgram(gray_scale_program, 1, &device, NULL, NULL, NULL);
    clBuildProgram(edges_scale_program, 1, &device, NULL, NULL, NULL);
    err = clBuildProgram(toon_scale_program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error build image data to buffer\n");
        return -1;
    }

   // cl_kernel kernel = clCreateKernel(program, "process_image", NULL);
    cl_kernel gray_kernel = clCreateKernel(gray_scale_program, "process_image", NULL);
    cl_kernel edges_kernel = clCreateKernel(edges_scale_program, "edges_detection", NULL);
    cl_kernel toon_kernel = clCreateKernel(toon_scale_program, "toon_shading", NULL);

    //clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_buffer);
   // clSetKernelArg(kernel, 1, sizeof(int), &width);
   // clSetKernelArg(kernel, 2, sizeof(int), &height);

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
    //clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    //clFinish(queue);

    clEnqueueNDRangeKernel(queue, gray_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueNDRangeKernel(queue, edges_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueNDRangeKernel(queue, toon_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, final_image_buffer, CL_TRUE, 0, width * height * channels, finalImage, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading image data from buffer\n");
        return -1;
    }

    // Zapisz przetworzony obraz
    stbi_write_png("output.jpg", width, height, channels, finalImage, width*channels);

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



    return 0;
}