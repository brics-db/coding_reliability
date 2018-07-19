cmake_minimum_required(VERSION 2.8)
project(codrel CXX)

set(CMAKE_CXX_STANDARD 11)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "--std=gnu++11")
endif ()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA)
include(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS} ${helper_INCLUDE_DIRS})
#set(CUDA_VERBOSE_BUILD ON)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CODREL_CUDA_ARCH "30" CACHE STRING "Target Compute Capability, e.g. '52'.")
# https://stackoverflow.com/questions/35656294/cuda-how-to-use-arch-and-code-and-sm-vs-compute
list(APPEND CUDA_NVCC_FLAGS  "-gencode arch=compute_30,code=compute_30")
list(APPEND CUDA_NVCC_FLAGS  "-gencode arch=compute_60,code=compute_60")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${CODREL_CUDA_ARCH}") # real architecture
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -Xcompiler -fopenmp")

list(APPEND SOURCES main.cpp globals.cpp an_coding.cu an_coding_grid.cu)
cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} helper rt)
