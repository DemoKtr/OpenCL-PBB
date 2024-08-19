#pragma once
#include <string>
#include <fstream>


namespace loader {
	const char* loadCL(const std::string filename) {
		std::ifstream file(filename);
		if (!file.is_open()) {
			throw std::runtime_error("Nie mo¿na otworzyæ pliku z kodem OpenCL: " + filename);
		}
		std::string sourceCode((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		char* sourceCodeCStr = new char[sourceCode.size() + 1];  // +1 dla null-terminatora
		std::strcpy(sourceCodeCStr, sourceCode.c_str());  // Kopiowanie danych do bufora
		return sourceCodeCStr;  // Zwracamy wskaŸnik
	}
}