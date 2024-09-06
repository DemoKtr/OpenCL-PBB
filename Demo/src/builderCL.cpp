#include "builderCL.h"
#include "fileLoader/Loader.h"
#include <iostream>
//#include "fileLoader/Loader.h"


void clInit::Initialize(CLInitializeInput input)
{
    clGetPlatformIDs(1, input.platform, NULL);

    clGetDeviceIDs(*input.platform, CL_DEVICE_TYPE_GPU, 1, input.device, NULL);
    clGetDeviceIDs(*input.platform, CL_DEVICE_TYPE_GPU, 1, input.device, NULL);

    *input.context = clCreateContext(NULL, 1, input.device, NULL, NULL, NULL);
    *input.queue = clCreateCommandQueue(*input.context, *input.device, 0, NULL);

}

void clInit::CreateProgram(CLCreateProgramInput input)
{
    const char* code = loader::loadCL(input.code);
    cl_int err;
    *input.program = clCreateProgramWithSource(input.context, 1, &code, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error create %d program\n", input.code);
    }
    Build(*input.program, input.device,input.code);
}
void clInit::Build(cl_program program, cl_device_id device, std::string code)
{
    cl_int err;
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error build %d program\n", code);
    }
}

void clInit::CreateKernel(cl_kernel* kernel, cl_program program)
{
    // cl_kernel kernel = clCreateKernel(program, "process_image", NULL);
    *kernel = clCreateKernel(program, "execute", NULL);

}

void clInit::DebugProgram(cl_device_id device, cl_program program){
   
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
    std::cerr << "Build log:\n" << log.data() << std::endl;
}
