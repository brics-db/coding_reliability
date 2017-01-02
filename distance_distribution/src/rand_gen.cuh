// Copyright 2016 Matthias Werner
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RAND_GEN_CUH_
#define RAND_GEN_CUH_

#include <curand.h>
#include <curand_kernel.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

template<typename RandGenType>
struct RandGen {
  RandGenType *devStates;

  void init(dim3 blocks, dim3 threads, uint_t offset, uint_t dim, uint_t dev);
  void free();
};

/// -

template<>
struct RandGen<curandStateSobol32_t> {
  curandStateSobol32_t *devStates;
  curandDirectionVectors32_t *hostVectors;
  curandDirectionVectors32_t *devDirectionVectors;
  
  void init(dim3 blocks, dim3 threads, uint_t offset, uint_t dim, uint_t dev);
  void free();
};

template<>
struct RandGen<curandStateSobol64_t> {
  curandStateSobol64_t *devStates;
  curandDirectionVectors64_t *hostVectors;
  curandDirectionVectors64_t *devDirectionVectors;
  
  void init(dim3 blocks, dim3 threads, uint_t offset, uint_t dim, uint_t dev);
  void free();
};

template<>
struct RandGen<curandStateScrambledSobol32_t> {
  curandStateScrambledSobol32_t *devStates;
  curandDirectionVectors32_t *hostVectors;
  curandDirectionVectors32_t *devDirectionVectors;
  uint_t *devScrambleConstants;
  uint_t *hostScrambleConstants;
  void init(dim3 blocks, dim3 threads, uint_t offset, uint_t dim, uint_t dev);
  void free();
};

//typedef curandStateScrambledSobol32_t RAND_GEN;
typedef curandStateSobol32_t RAND_GEN;
//typedef curandStateSobol64_t RAND_GEN;
//typedef curandState_t RAND_GEN;


#endif /* RAND_GEN_CUH_ */
