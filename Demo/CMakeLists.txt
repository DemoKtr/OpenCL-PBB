# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project (OpenCLPPB VERSION 0.0.1 LANGUAGES CXX)

set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/thirdParty/include/OpenCL")
set(OpenCL_LIBRARY "${CMAKE_SOURCE_DIR}/thirdParty/lib/OpenCL.lib")

# Dodaj ścieżki do nagłówków OpenCL
include_directories(${OpenCL_INCLUDE_DIR})



# Add source to this project's executable.
add_executable (OpenCLPPB src/main.cpp  "src/Toon.h" "src/fileLoader/Loader.h" "src/Toon.cpp" "src/builderCL.h" "src/builderCL.cpp" "src/NeuralNetwork.h" "src/NeuralNetwork.cpp" "src/PBB.h" "src/PBB.cpp"  "src/Mosaic.h" "src/Mosaic.cpp" "src/fileLoader/DataLoader.h" "src/fileLoader/DataLoader.cpp")

# Zdefiniuj ścieżkę do folderu 'resource' w katalogu źródłowym
set(RESOURCE_DIR "${CMAKE_SOURCE_DIR}/resource")

# Zdefiniuj ścieżkę do katalogu 'resource' w folderze build
set(RESOURCE_DEST_DIR "${CMAKE_BINARY_DIR}/resource")

# Skopiuj folder 'resource' do folderu build
file(COPY ${RESOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})

target_link_libraries(OpenCLPPB ${OpenCL_LIBRARY})
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET OpenCLPPB PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
