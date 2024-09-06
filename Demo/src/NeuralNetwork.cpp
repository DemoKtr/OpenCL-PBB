#include "NeuralNetwork.h"
#include "fileLoader/DataLoader.h"
#include <random>
#include "builderCL.h"
#include <cassert>
void run(int neuronNumber,int epochsNumber)
{
    float learningRate = 0.01f;
    int outputNeuronNumber = 3;
    int featuresNumber = 4;
	std::vector<IrisLoadData> loadData = readCSV("iris.csv");
	removePrefix(loadData,"Iris-");
    IrisData* data = new IrisData[loadData.size()];
    convertData(data, loadData);
    int irisNumber = loadData.size();
    float* hiden_weights;
    float* output_weights;
    float* hiden_biases = new float[neuronNumber];
    float* output_biases = new float[outputNeuronNumber];
    hiden_weights = new float[neuronNumber * featuresNumber];
    //
   // for (int i = 0; i < loadData.size();++i) {
   //     std::cout << data[i].petalLength << "  " << data[i].petalWidth << "  " << data[i].sepalLength << "  " << data[i].sepalWidth<<std::endl;
   // }
    
    output_weights = new float[outputNeuronNumber*neuronNumber];//////////////////////////////////////////////
    std::vector<int> uniqueNumbersX = generateUniqueRandomNumbers(0, 49, 1);
    std::vector<int> uniqueNumbersY = generateUniqueRandomNumbers(50, 99,1);
    std::vector<int> uniqueNumbersZ = generateUniqueRandomNumbers(100, 149, 1);
    int learningDataSize = uniqueNumbersX.size() + uniqueNumbersY.size() + uniqueNumbersZ.size();
    IrisData* learningData = new IrisData[learningDataSize];
    finalResult* finale = new finalResult[irisNumber];
    outputLayerOutput* read_HW = new outputLayerOutput[learningDataSize];


    

    for (uint32_t i = 0; i < learningDataSize; ++i) {
        if (i < uniqueNumbersX.size()) {
            learningData[i].petalLength = data[uniqueNumbersX[i]].petalLength;
            learningData[i].petalWidth = data[uniqueNumbersX[i]].petalWidth;
            learningData[i].sepalLength = data[uniqueNumbersX[i]].sepalLength;
            learningData[i].sepalWidth = data[uniqueNumbersX[i]].sepalWidth;
            learningData[i].type = data[uniqueNumbersX[i]].type;

        }
            
        else if (i < uniqueNumbersY.size()+ uniqueNumbersX.size()) {
        learningData[i].petalLength = data[uniqueNumbersY[i- uniqueNumbersX.size()]].petalLength;
        learningData[i].petalWidth = data[uniqueNumbersY[i - uniqueNumbersX.size()]].petalWidth;
        learningData[i].sepalLength = data[uniqueNumbersY[i - uniqueNumbersX.size()]].sepalLength;
        learningData[i].sepalWidth = data[uniqueNumbersY[i - uniqueNumbersX.size()]].sepalWidth;
        learningData[i].type = data[uniqueNumbersY[i - uniqueNumbersX.size()]].type;
        }
        else if (i < (uniqueNumbersZ.size()+ uniqueNumbersY.size() + uniqueNumbersX.size())) {
            learningData[i].petalLength = data[uniqueNumbersZ[i - uniqueNumbersX.size() - uniqueNumbersY.size()]].petalLength;
            learningData[i].petalWidth = data[uniqueNumbersZ[i - uniqueNumbersX.size() - uniqueNumbersY.size()]].petalWidth;
            learningData[i].sepalLength = data[uniqueNumbersZ[i - uniqueNumbersX.size() - uniqueNumbersY.size()]].sepalLength;
            learningData[i].sepalWidth = data[uniqueNumbersZ[i - uniqueNumbersX.size() - uniqueNumbersY.size()]].sepalWidth;
            learningData[i].type = data[uniqueNumbersZ[i - uniqueNumbersX.size() - uniqueNumbersY.size()]].type;
        }
            
    }


    hidenLayerOutput* hidenLayerOutputs = new hidenLayerOutput[irisNumber];

    std::random_device rd; // uzyskaj losowe ziarno z urzπdzenia
    std::mt19937 gen(rd()); // uøyj Mersenne Twister jako generatora

    float limit = sqrt(2 / ((float)featuresNumber));

    std::uniform_real_distribution<> distrib(-limit, limit);
    std::normal_distribution<> normdist(-0.01, 0.01);

    for (uint32_t i = 0; i < neuronNumber; ++i) {
        for (uint32_t j = 0; j < featuresNumber; ++j) {
            hiden_weights[i*featuresNumber + j] = distrib(gen);
        }
        hiden_biases[i] = normdist(gen);

    }

    for (uint32_t i = 0; i < outputNeuronNumber * neuronNumber; ++i) {
        output_weights[i] = distrib(gen);
       

    }
    for (uint32_t i = 0; i < outputNeuronNumber; ++i) {
        output_biases[i] = normdist(gen);

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

    cl_mem iris_data_buffer;//
    cl_mem iris_learning_data_buffer;//
    cl_mem weights_hiden_buffer;//
    cl_mem weights_output_buffer;//
    cl_mem biases_hiden_buffer;//
    cl_mem biases_output_buffer;//
    cl_mem hiden_result_buffer;//
    cl_mem final_output_result_buffer;//
    cl_mem output_result_buffer;//
    cl_mem local_logits_buffer;
    cl_mem final_local_logits_buffer;
    cl_mem hiden_gradients_buffer;//
    cl_mem output_gradients_buffer;//
    cl_mem debug_buffer;//
    cl_mem final_hiden_result_buffer;//


    cl_int err;
    clInit::CLCreateBufferInput<IrisData> irisDataBufferInput = {};
    irisDataBufferInput.buffer = &iris_data_buffer;
    irisDataBufferInput.context = context;
    irisDataBufferInput.queue = queue;
    irisDataBufferInput.size = sizeof(IrisData) * irisNumber;
    irisDataBufferInput.data = data;
    irisDataBufferInput.readOnly = true;
    clInit::CreateBuffer(irisDataBufferInput);

    irisDataBufferInput.buffer = &iris_learning_data_buffer;
    irisDataBufferInput.size = sizeof(IrisData) * learningDataSize;
    irisDataBufferInput.data = learningData;
    clInit::CreateBuffer(irisDataBufferInput);


    clInit::CLCreateBufferInput<float> weightBufferInput = {};
    weightBufferInput.buffer = &weights_hiden_buffer;
    weightBufferInput.context = context;
    weightBufferInput.queue = queue;
    weightBufferInput.size = sizeof(float)* neuronNumber * featuresNumber;
    weightBufferInput.data = hiden_weights;
    weightBufferInput.readOnly = false;
    clInit::CreateBuffer(weightBufferInput);

    weightBufferInput.size = sizeof(float) * outputNeuronNumber * neuronNumber;
    weightBufferInput.data = output_weights;
    weightBufferInput.buffer = &weights_output_buffer;
    clInit::CreateBuffer(weightBufferInput);

    weightBufferInput.size = sizeof(float) * neuronNumber;
    weightBufferInput.data = hiden_biases;
    weightBufferInput.buffer = &biases_hiden_buffer;
    clInit::CreateBuffer(weightBufferInput);

    weightBufferInput.size = sizeof(float) * outputNeuronNumber;
    weightBufferInput.data = output_biases;
    weightBufferInput.buffer = &biases_output_buffer;
    clInit::CreateBuffer(weightBufferInput);

    weightBufferInput.size = sizeof(float) * learningDataSize *3;
    weightBufferInput.data = nullptr;
    weightBufferInput.buffer = &hiden_gradients_buffer;
    clInit::CreateBuffer(weightBufferInput);

    weightBufferInput.size = sizeof(float) * learningDataSize* outputNeuronNumber;
    weightBufferInput.buffer = &output_gradients_buffer;
    clInit::CreateBuffer(weightBufferInput);

    
    weightBufferInput.buffer = &hiden_result_buffer;
    weightBufferInput.size = sizeof(hidenLayerOutput) * learningDataSize;
    clInit::CreateBuffer(weightBufferInput);
    weightBufferInput.buffer = &final_hiden_result_buffer;
    weightBufferInput.size = sizeof(hidenLayerOutput) * irisNumber;
    clInit::CreateBuffer(weightBufferInput);
    weightBufferInput.buffer = &final_output_result_buffer;
    weightBufferInput.size = sizeof(finalResult) * irisNumber;
    clInit::CreateBuffer(weightBufferInput);
    weightBufferInput.buffer = &debug_buffer;
    weightBufferInput.size = sizeof(DebugStruct) * learningDataSize;
    clInit::CreateBuffer(weightBufferInput);
    weightBufferInput.buffer = &output_result_buffer;
    weightBufferInput.size = sizeof(outputLayerOutput) * learningDataSize;
    clInit::CreateBuffer(weightBufferInput);

    weightBufferInput.buffer = &local_logits_buffer;
    weightBufferInput.size = sizeof(float) * learningDataSize * outputNeuronNumber;
    clInit::CreateBuffer(weightBufferInput);
    weightBufferInput.buffer = &final_local_logits_buffer;
    weightBufferInput.size = sizeof(float) * irisNumber * outputNeuronNumber;
    clInit::CreateBuffer(weightBufferInput);

   
    cl_program hidenLayer;
    cl_program hidenLayerDetection;
    cl_program outputLayer;
    cl_program outputLayerDetection;
    cl_program backPropagation;



    
    clInit::CLCreateProgramInput programInfo = {};
    programInfo.context = context;
    programInfo.code = "hidenLayer.CL";
    programInfo.program = &hidenLayer;
    programInfo.device = device;
    clInit::CreateProgram(programInfo);
    programInfo.code = "outputLayer.CL";
    programInfo.program = &outputLayer;
    clInit::CreateProgram(programInfo);
    programInfo.code = "hidenLayerDetection.CL";
    programInfo.program = &hidenLayerDetection;
    clInit::CreateProgram(programInfo);
    programInfo.code = "outputLayerDetection.CL";
    programInfo.program = &outputLayerDetection;
    clInit::CreateProgram(programInfo);
    programInfo.code = "backPropagation.CL";
    programInfo.program = &backPropagation;
    clInit::CreateProgram(programInfo);
    clInit::DebugProgram(device, outputLayerDetection);
    
   
    cl_kernel hidenLayer_kernel;
    cl_kernel outputLayer_kernel;
    cl_kernel backPropagation_kernel;
    cl_kernel hidenLayerDetection_kernel;
    cl_kernel outputLayerDetection_kernel;


    clInit::CreateKernel(&hidenLayer_kernel, hidenLayer);
    clInit::CreateKernel(&hidenLayerDetection_kernel, hidenLayerDetection);
    clInit::CreateKernel(&outputLayer_kernel, outputLayer);
    clInit::CreateKernel(&outputLayerDetection_kernel, outputLayerDetection);
    clInit::CreateKernel(&backPropagation_kernel, backPropagation);

   // clInit::CreateKernel(&test_kernel, test);

    
    clSetKernelArg(hidenLayer_kernel, 0, sizeof(cl_mem), &iris_learning_data_buffer);
    clSetKernelArg(hidenLayer_kernel, 1, sizeof(cl_mem), &weights_hiden_buffer);
    clSetKernelArg(hidenLayer_kernel, 2, sizeof(cl_mem), &biases_hiden_buffer);
    clSetKernelArg(hidenLayer_kernel, 3, sizeof(cl_mem), &hiden_result_buffer);
    clSetKernelArg(hidenLayer_kernel, 4, sizeof(int), &learningDataSize);
    clSetKernelArg(hidenLayer_kernel, 5, sizeof(int), &neuronNumber);

    clSetKernelArg(hidenLayerDetection_kernel, 0, sizeof(cl_mem), &iris_data_buffer);
    clSetKernelArg(hidenLayerDetection_kernel, 1, sizeof(cl_mem), &weights_hiden_buffer);
    clSetKernelArg(hidenLayerDetection_kernel, 2, sizeof(cl_mem), &biases_hiden_buffer);
    clSetKernelArg(hidenLayerDetection_kernel, 3, sizeof(cl_mem), &final_hiden_result_buffer);
    clSetKernelArg(hidenLayerDetection_kernel, 4, sizeof(int), &irisNumber);
    clSetKernelArg(hidenLayerDetection_kernel, 5, sizeof(int), &neuronNumber);


    clSetKernelArg(outputLayer_kernel, 0, sizeof(cl_mem), &hiden_result_buffer);
    clSetKernelArg(outputLayer_kernel, 1, sizeof(cl_mem), &weights_output_buffer);
    clSetKernelArg(outputLayer_kernel, 2, sizeof(cl_mem), &biases_output_buffer);
    clSetKernelArg(outputLayer_kernel, 3, sizeof(cl_mem), &output_result_buffer);
    clSetKernelArg(outputLayer_kernel, 4, sizeof(cl_mem), &local_logits_buffer);
    clSetKernelArg(outputLayer_kernel, 5, sizeof(int), &learningDataSize);
    clSetKernelArg(outputLayer_kernel, 6, sizeof(int), &neuronNumber);
    clSetKernelArg(outputLayer_kernel, 7, sizeof(int), &outputNeuronNumber);
    clSetKernelArg(outputLayer_kernel, 8, sizeof(cl_mem), &hiden_gradients_buffer);


    clSetKernelArg(outputLayerDetection_kernel, 0, sizeof(cl_mem), &final_hiden_result_buffer);
    clSetKernelArg(outputLayerDetection_kernel, 1, sizeof(cl_mem), &weights_output_buffer);
    clSetKernelArg(outputLayerDetection_kernel, 2, sizeof(cl_mem), &biases_output_buffer);
    clSetKernelArg(outputLayerDetection_kernel, 3, sizeof(cl_mem), &final_output_result_buffer);
    clSetKernelArg(outputLayerDetection_kernel, 4, sizeof(cl_mem), &final_local_logits_buffer);
    clSetKernelArg(outputLayerDetection_kernel, 5, sizeof(int), &irisNumber);
    clSetKernelArg(outputLayerDetection_kernel, 6, sizeof(int), &neuronNumber);
    clSetKernelArg(outputLayerDetection_kernel, 7, sizeof(int), &outputNeuronNumber);



    clSetKernelArg(backPropagation_kernel, 0, sizeof(cl_mem), &output_result_buffer);
    clSetKernelArg(backPropagation_kernel, 1, sizeof(cl_mem), &hiden_gradients_buffer);
    clSetKernelArg(backPropagation_kernel, 2, sizeof(cl_mem), &output_gradients_buffer);
    clSetKernelArg(backPropagation_kernel, 3, sizeof(cl_mem), &weights_hiden_buffer);
    clSetKernelArg(backPropagation_kernel, 4, sizeof(cl_mem), &weights_output_buffer);
    clSetKernelArg(backPropagation_kernel, 5, sizeof(cl_mem), &biases_hiden_buffer);
    clSetKernelArg(backPropagation_kernel, 6, sizeof(cl_mem), &biases_output_buffer);
    clSetKernelArg(backPropagation_kernel, 7, sizeof(int), &neuronNumber);
    clSetKernelArg(backPropagation_kernel, 8, sizeof(int), &outputNeuronNumber);
    clSetKernelArg(backPropagation_kernel, 9, sizeof(int), &learningDataSize);
    clSetKernelArg(backPropagation_kernel, 10, sizeof(float), &learningRate);
    clSetKernelArg(backPropagation_kernel, 11, sizeof(cl_mem), &debug_buffer);

    for (uint32_t i = 0; i < epochsNumber; ++i) {
        size_t hiden_global_work_size[2] = { irisNumber ,neuronNumber };

        err = clEnqueueNDRangeKernel(queue, hidenLayer_kernel, 2, NULL, hiden_global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error execute kernel\n");
        }
        clFinish(queue);
        size_t output_global_work_size[2] = { irisNumber ,outputNeuronNumber };

        err = clEnqueueNDRangeKernel(queue, outputLayer_kernel, 2, NULL, output_global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error execute kernel\n");
        }
        clFinish(queue);
     

        err = clEnqueueNDRangeKernel(queue, backPropagation_kernel, 2, NULL, output_global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error execute kernel\n");
        }
        clFinish(queue);

        
      
    }


    size_t hiden_global_work_size[2] = { irisNumber ,neuronNumber };

    err = clEnqueueNDRangeKernel(queue, hidenLayerDetection_kernel, 2, NULL, hiden_global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error execute kernel\n");
    }
    clFinish(queue);
    size_t output_global_work_size[2] = { irisNumber ,outputNeuronNumber };

    err = clEnqueueNDRangeKernel(queue, outputLayerDetection_kernel, 2, NULL, output_global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error execute kernel\n");
    }
    clFinish(queue);
   
    err = clEnqueueReadBuffer(queue, final_output_result_buffer, CL_TRUE, 0, irisNumber * sizeof(finalResult), finale, 0, NULL, NULL);
    
  
    


    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading image data from buffer\n");
    }
    
    

    for (uint32_t i = 0; i < irisNumber; ++i) {
        std::cout << "Epoka :" << i << std::endl;
        std::cout << "X: " <<finale[i].value[0] << "Y: " << finale[i].value[1] << "Z: " << finale[i].value[2] << std::endl;
    }
    
  


    clReleaseMemObject(iris_data_buffer);
    clReleaseMemObject(iris_learning_data_buffer);
    clReleaseMemObject(weights_hiden_buffer);
    clReleaseMemObject(weights_output_buffer);
    clReleaseMemObject(biases_hiden_buffer);
    clReleaseMemObject(biases_output_buffer);
    clReleaseMemObject(hiden_result_buffer);
    clReleaseMemObject(final_output_result_buffer);
    clReleaseMemObject(output_result_buffer);
    clReleaseMemObject(final_hiden_result_buffer);
    clReleaseMemObject(local_logits_buffer);
    clReleaseMemObject(final_local_logits_buffer);
    clReleaseMemObject(hiden_gradients_buffer);
    clReleaseMemObject(output_gradients_buffer);
    clReleaseMemObject(debug_buffer);

    clReleaseKernel(hidenLayer_kernel);
    clReleaseKernel(hidenLayerDetection_kernel);

    clReleaseKernel(outputLayer_kernel);
    clReleaseKernel(outputLayerDetection_kernel);
    clReleaseKernel(backPropagation_kernel);
    clReleaseProgram(hidenLayer);
    clReleaseProgram(outputLayer);
    clReleaseProgram(outputLayerDetection);
    clReleaseProgram(hidenLayerDetection);
    clReleaseProgram(backPropagation);
  

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    delete[] hiden_weights;

    delete[] output_weights;

    delete[] data;

    delete[] hiden_biases;

    delete[] output_biases;

    delete[] hidenLayerOutputs;

    delete[] finale;

    delete[] read_HW;
 

}


std::vector<int> generateUniqueRandomNumbers(int k, int z, int n) {
    // Sprawdü, czy jest wystarczajπco liczb w przedziale
    assert(n <= (z - k + 1) && "Nie moøna wylosowaÊ wiÍcej unikalnych liczb niø dostÍpnych w przedziale.");

    // Tworzenie wektora z liczbami z przedzia≥u <k, z>
    std::vector<int> numbers;
    for (int i = k; i <= z; ++i) {
        numbers.push_back(i);
    }

    // Uøycie generatora losowego
    std::random_device rd;  // uzyskanie losowego ürÛd≥a
    std::mt19937 g(rd());   // inicjalizacja generatora

    // Tasowanie wektora
    std::shuffle(numbers.begin(), numbers.end(), g);

    // Wybranie pierwszych n elementÛw
    std::vector<int> result(numbers.begin(), numbers.begin() + n);

    return result;
}