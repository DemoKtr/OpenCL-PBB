#pragma once
#include <vector>

struct hidenLayerOutput {
	float values[255];
	float type[3];
	float basic_feature[4];
};
struct outputLayerOutput {
	float value[3];
	float type[3];
	float hidenValue[255];
	float basic_feature[4];
};
struct DebugStruct {
	float value[3];
	float expect[3];
	float diff[3];

};

struct finalResult{
	float value[3];
};

void run(int neuronNumber, int epochsNumber);

std::vector<int> generateUniqueRandomNumbers(int k, int z, int n);