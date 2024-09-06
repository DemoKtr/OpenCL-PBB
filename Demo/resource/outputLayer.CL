


__kernel void execute(__global float *result,
					  __global float *weights,
					  __global float *biases,
					  __global float *output,
					  __local float *localLogits,
					  const int hidenNeuronNumber,
					  const int outputNeuronNumber
					  ) {

    int neuron = get_global_id(0); // neuron

	if( neuron<outputNeuronNumber){
	float logit =0.0f;
	for(int i=0;i<hidenNeuronNumber; ++i) {
	logit+=result[i]*weights[(hidenNeuronNumber * neuron) +i];
	}
	logit+= biases[neuron];
	

    localLogits[neuron] = logit;

    barrier(CLK_LOCAL_MEM_FENCE);

    float exp_sum = 0.0f;
    float exp_logit = exp(logit);


    for (int i = 0; i < outputNeuronNumber; ++i) {
        exp_sum += exp(localLogits[i]);
    }


    float softmax_value = exp_logit / exp_sum;

    output[neuron] = softmax_value;
	}
}




