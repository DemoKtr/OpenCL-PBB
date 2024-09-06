typedef struct{
    float sepalLength;
    float sepalWidth;
    float petalLength;
    float petalWidth;
    int type;
}IrisData;


float reluActivation(float x){
	if(x<=0) x=0; 
	return x;
}




__kernel void execute(__global IrisData *data,
					  __global float *weight,
					  __global float *bias,
					  __global float *result,
                      const int irisNumber,
					  const int neuronNumber
					  ) {

    int iris = get_global_id(0); // Irys 
    int neuron = get_global_id(1); // neuron
	int featureNumber = 4;
	if( iris<irisNumber && neuron<neuronNumber){
	float value = 0;
	value += weight[(neuron*featureNumber)]*data[iris].sepalLength;
	value += weight[(neuron*featureNumber)+1]*data[iris].sepalWidth;
	value += weight[(neuron*featureNumber)+2]*data[iris].petalLength;
	value += weight[(neuron*featureNumber)+3]*data[iris].petalWidth;
	value += bias[neuron];
	result[neuron] = reluActivation(value);
	
	}
}




