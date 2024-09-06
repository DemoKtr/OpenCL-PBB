#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

struct IrisLoadData {
    float sepalLength;
    float sepalWidth;
    float petalLength;
    float petalWidth;
    std::string species;
};

struct IrisData {
    float sepalLength;
    float sepalWidth;
    float petalLength;
    float petalWidth;
    int type;
};

std::vector<IrisLoadData> readCSV(const std::string& filename);

void printData(const std::vector<IrisLoadData> irisData);
void removePrefix(std::vector<IrisLoadData>& data, const std::string& prefix);
void convertData(IrisData* data ,const std::vector<IrisLoadData>& loadData);