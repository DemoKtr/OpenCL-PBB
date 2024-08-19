#pragma once
#include <CL/cl.h>
#include <vector>
#include <string>
namespace clInit {
	struct CLInitializeInput {
		cl_platform_id *platform;
		cl_device_id *device;
		cl_context *context;
		cl_command_queue *queue;
	};
	template<typename T>
	struct CLCreateBufferInput {
		cl_context context;
		cl_mem *buffer;
		cl_command_queue queue;
		size_t size;
		bool readOnly;
		T* data;
	};

	struct CLCreateProgramInput {
		cl_context context;
		std::string code;
		cl_program *program;
		cl_device_id device;
	};


	void Initialize(CLInitializeInput input);

	template<typename T>
	void CreateBuffer(CLCreateBufferInput<T> input) {


		if (input.readOnly)
			*input.buffer = clCreateBuffer(input.context, CL_MEM_READ_ONLY, input.size, NULL, NULL);
		else *input.buffer = clCreateBuffer(input.context, CL_MEM_READ_WRITE, input.size, NULL, NULL);



		if (input.data != nullptr) {
			cl_int err;
				err = clEnqueueWriteBuffer(input.queue, *input.buffer, CL_TRUE, 0, input.size, input.data, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				fprintf(stderr, "Error copying image data to buffer\n");
			}
		}
	}


	void CreateProgram(CLCreateProgramInput input);

	void Build(cl_program program, cl_device_id device);
	void CreateKernel(cl_kernel *kernel, cl_program program);
}
