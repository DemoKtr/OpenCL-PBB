#include "DataLoader.h"

std::vector<IrisLoadData> readCSV(const std::string& filename)
{
        std::vector<IrisLoadData> data;
        std::ifstream file(filename);
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            IrisLoadData iris;

            try {
                std::getline(ss, item, ',');
                iris.sepalLength = std::stof(item);
                std::getline(ss, item, ',');
                iris.sepalWidth = std::stof(item);
                std::getline(ss, item, ',');
                iris.petalLength = std::stof(item);
                std::getline(ss, item, ',');
                iris.petalWidth = std::stof(item);
                std::getline(ss, item, ',');
                iris.species = item;

                data.push_back(iris);
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument error: " << e.what() << " in line: " << line << std::endl;
            }
            catch (const std::out_of_range& e) {
                std::cerr << "Out of range error: " << e.what() << " in line: " << line << std::endl;
            }
        }

        for (IrisLoadData iris : data) {

         
        }
        return data;
}

void printData(const std::vector<IrisLoadData> irisData)
{
    
    for (const auto& iris : irisData) {
        std::cout << "Sepal Length: " << iris.sepalLength
            << ", Sepal Width: " << iris.sepalWidth
            << ", Petal Length: " << iris.petalLength
            << ", Petal Width: " << iris.petalWidth
            << ", Species: " << iris.species << std::endl;
    }
}

void removePrefix(std::vector<IrisLoadData>& data, const std::string& prefix)
{
    size_t prefixLength = prefix.length();
    for (auto& iris : data) {
        // SprawdŸ, czy ci¹g zaczyna siê od prefiksu
        if (iris.species.compare(0, prefixLength, prefix) == 0) {
            // U¿yj substr, aby usun¹æ prefiks
            iris.species = iris.species.substr(prefixLength);
        }
    }
}

void convertData(IrisData* data,const std::vector<IrisLoadData>& loadData)
{
    
    size_t i = 0;
    float minsepalLength = FLT_MAX;
    float minsepalWidth = FLT_MAX;
    float minpetalLength = FLT_MAX;
    float minpetalWidth = FLT_MAX;
    float maxsepalLength = FLT_MIN;
    float maxsepalWidth = FLT_MIN;
    float maxpetalLength = FLT_MIN;
    float maxpetalWidth = FLT_MIN;


    for (IrisLoadData loadedIris : loadData) {

        if (minsepalLength > loadedIris.sepalLength) minsepalLength = loadedIris.sepalLength;
        if (maxsepalLength < loadedIris.sepalLength) maxsepalLength = loadedIris.sepalLength;

        if (minsepalWidth > loadedIris.sepalWidth) minsepalWidth = loadedIris.sepalWidth;
        if (maxsepalWidth < loadedIris.sepalWidth) maxsepalWidth = loadedIris.sepalWidth;

        if (minpetalLength > loadedIris.petalLength) minpetalLength = loadedIris.petalLength;
        if (maxpetalLength < loadedIris.petalLength) maxpetalLength = loadedIris.petalLength;

        if (minpetalWidth > loadedIris.petalWidth) minpetalWidth = loadedIris.petalWidth;
        if (maxpetalWidth < loadedIris.petalWidth) maxpetalWidth = loadedIris.petalWidth;
    }

    float minmaxsepalLength = maxsepalLength - minsepalLength;
    float minmaxsepalWidth = maxsepalWidth - minsepalWidth;
    float minmaxpetalLength = maxpetalLength - minpetalLength;
    float minmaxpetalWidth = maxpetalWidth - minpetalWidth;

    for (IrisLoadData loadedIris : loadData) {
        IrisData iris;
        iris.petalLength = ((loadedIris.petalLength-minpetalLength)/minmaxpetalLength);
        iris.petalWidth = ((loadedIris.petalWidth - minpetalWidth) / minmaxpetalWidth);
        iris.sepalLength = ((loadedIris.sepalLength - minsepalLength) / minmaxsepalLength);
        iris.sepalWidth = ((loadedIris.sepalWidth - minsepalWidth) / minmaxsepalWidth);
        if (loadedIris.species == "setosa") {
            iris.type = 0;

           
        }
        else if (loadedIris.species == "versicolor") {
            iris.type = 1;
            
        }
        else{
            iris.type = 2;
        }
        data[i++] = iris;
    }

}
