/*
 * rand_gen.cuh
 *
 *  Created on: 17.03.2016
 *      Author: gcoe02
 */

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

#endif /* RAND_GEN_CUH_ */
