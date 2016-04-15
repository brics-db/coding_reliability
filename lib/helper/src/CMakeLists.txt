CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(helper CXX)

find_package(CUDA)
set(PROJECT_DIR ${PROJECT_SOURCE_DIR}/..)
include_directories(${PROJECT_DIR}/inc ${CUDA_INCLUDE_DIRS})
set(SOURCES ${SOURCES} statistics.cpp timestatistics.cpp timer.cpp)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

set_target_properties(${PROJECT_NAME}
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${PROJECT_NAME} rt)

set(${PROJECT_NAME}_INCLUDE_DIRS "${PROJECT_DIR}" "${CUDA_INCLUDE_DIRS}"
    CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)