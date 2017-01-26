# Copyright 2016 Till Kolditz
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)

SET(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
SET(CMAKE_DISABLE_SOURCE_CHANGES  ON)

SET(CMAKE_COLOR_MAKEFILE ON)
# And optionally
#SET(CMAKE_VERBOSE_MAKEFILE ON)

IF(POLICY CMP0048)
    CMAKE_POLICY(SET CMP0048 NEW)
    PROJECT(ANDecodingIsErrorDetection VERSION 1.1 LANGUAGES CXX)
ELSE()
    PROJECT(ANDecodingIsErrorDetection LANGUAGES CXX)
ENDIF()

ADD_DEFINITIONS(-Wall)
ADD_DEFINITIONS(-pedantic)
ADD_DEFINITIONS(-march=native)
ADD_DEFINITIONS(-Wno-ignored-attributes)

IF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  MESSAGE(SEND_ERROR "In-source builds are not allowed.")
ENDIF ()
IF(WIN32)
  SET(CMAKE_SHARED_LIBRARY_PREFIX "")
ENDIF ()

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
# set project build type
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
MESSAGE(">> Setting up ${CMAKE_BUILD_TYPE} build")

ADD_DEFINITIONS(-Wall)
ADD_DEFINITIONS(-pedantic)
ADD_DEFINITIONS(-march=native)

ADD_EXECUTABLE(an_decoding_is_error_detection src/an_decoding_is_error_detection.cpp src/stopwatch.cpp)

SET_PROPERTY(TARGET an_decoding_is_error_detection PROPERTY CXX_STANDARD 14)
SET_PROPERTY(TARGET an_decoding_is_error_detection PROPERTY CXX_STANDARD_REQUIRED ON)
# Force -O2, because -O3 seems to produce erroneous code on GCC 6.2
TARGET_COMPILE_OPTIONS(an_decoding_is_error_detection PRIVATE "-O2")
