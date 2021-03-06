# Copyright 2017 Till Kolditz
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

IF(POLICY CMP0048)
    CMAKE_POLICY(SET CMP0048 NEW)
    PROJECT(Legacy VERSION 1.0 LANGUAGES CXX)
ELSE()
    PROJECT(Legacy LANGUAGES CXX)
ENDIF()

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_COLOR_MAKEFILE ON)
# And optionally
#set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
message(">> Setting up ${CMAKE_BUILD_TYPE} build")

FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Boost REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

ADD_DEFINITIONS(-Wall)
ADD_DEFINITIONS(-pedantic)
ADD_DEFINITIONS(-march=native)
#ADD_DEFINITIONS(-mavx2)
#ADD_DEFINITIONS(-march=knl)
ADD_DEFINITIONS(-Wno-ignored-attributes)
ADD_DEFINITIONS(-Weffc++)

MACRO(append_target_property arg_target arg_property arg_setting)
	GET_TARGET_PROPERTY(TEMP ${arg_target} ${arg_property})
	IF(TEMP STREQUAL "TEMP-NOTFOUND")
		SET(TEMP "") # set to empty string
	ELSE()
		SET(TEMP "${TEMP} ") # a space to cleanly separate from existing content
	ENDIF()
	# append our values
	SET(TEMP "${TEMP}${arg_setting}")
	SET_TARGET_PROPERTIES(${arg_target} PROPERTIES ${arg_property} ${TEMP})
ENDMACRO(append_target_property)

MACRO( add_algorithm target_name source_files )
    MESSAGE(STATUS "Add executable ${target_name}")
    STRING(REPLACE " " ";" src_files "${source_files}")
    ADD_EXECUTABLE(${target_name} ${src_files})
    SET_PROPERTY(TARGET ${target_name} PROPERTY CXX_STANDARD 17)
    SET_PROPERTY(TARGET ${target_name} PROPERTY CXX_STANDARD_REQUIRED ON)
ENDMACRO( add_algorithm )

MACRO( add_algorithm_OMP target_name source_files )
	add_algorithm(${target_name} ${source_files})
    append_target_property(${target_name} "COMPILE_FLAGS" "${OpenMP_CXX_FLAGS}")
    append_target_property(${target_name} "LINK_FLAGS" "${OpenMP_CXX_FLAGS}")
ENDMACRO( add_algorithm_OMP )

add_algorithm_OMP(hamming "hamming_count_undetectable_errors.cpp Util/Statistics.cpp Util/Binom.cpp Util/StopWatch.cpp Hamming/Scalar.cpp \
Hamming/ExtHamming08.cpp Hamming/ExtHamming16.cpp Hamming/ExtHamming32.cpp Hamming/ExtHamming64.cpp \
Hamming/SSE_08.cpp Hamming/SSE_16.cpp Hamming/SSE_32.cpp Hamming/SSE_64.cpp \
Hamming/AVX2_08.cpp Hamming/AVX2_16.cpp Hamming/AVX2_32.cpp Hamming/AVX2_64.cpp")
add_algorithm_OMP(ancoding_count_undetectable_errors "ancoding_count_undetectable_errors.cpp")
add_algorithm_OMP(ancoding_count_undetectable_errors2 "ancoding_count_undetectable_errors2.cpp")
add_algorithm(print_codeword_matrix "print_codeword_matrix.cpp")
