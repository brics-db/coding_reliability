CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(codrel)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 11)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  LIST (APPEND CMAKE_CXX_FLAGS "--std=gnu++11")
endif ()


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

FIND_PACKAGE(CUDA)
INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} ${helper_INCLUDE_DIRS})
set(CUDA_VERBOSE_BUILD ON)

#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_37,code=compute_37 --default-stream per-thread")
LIST(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_37,code=compute_37")
LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11 -Xcompiler -fopenmp")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

LIST(APPEND SOURCES main.cpp globals.cpp hamming.cpp hamming_native.cpp hamming_native_short.cpp hamming_native_short_ordering.cpp hamming_grid.cu hamming.cu hamming_mc.cu an_coding.cu an_coding.cpp an_coding_mc.cu an_coding_grid.cu test_curand.cu rand_gen.cu)
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} helper rt)
