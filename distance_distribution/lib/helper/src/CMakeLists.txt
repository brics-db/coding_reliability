# Copyright 2016 Matthias Werner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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