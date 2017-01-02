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

#include "globals.h"
#include "rand_gen.cuh"

#include <helper.h>

#define OFFSET_MULT 8137 // should be a higher number
/*
Notes:
i * 32 * curandDirectionVectors32 === curandDirectionVectors32_t[i]

offset in curand_init should be high between threads (e.g. offset=tid*XXXXX)
(first 'random' numbers are nonsense)

does every threads needs his own dimension aka own direction vector (aka own stream) ?
=> you are limited to 20k threads
=> better to play with offsets
(1 random number stream has sequence of sobol32 is 2^32)

curand() does not deliver random number (but index of next element), use curand_*() instead

better to use fewer threads with longer runnings within random number stream
 */

template<uint_t DIM>
__global__ void init_rand_gen(curandState_t *state,
    uint_t seed)
{
  uint_t tid = DIM*(threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x));

  curand_init(seed,
              tid,
              1,
              &state[tid]);
  if(DIM==2)
    curand_init(seed+1,
              tid+1,
              2,
              &state[tid]);
  if(DIM==3)
    curand_init(seed+2,
              tid+2,
              3,
              &state[tid]);
}


template<uint_t DIM, typename curandDirectionVectors_sz, typename curandStateSobol_sz>
__global__ void init_rand_gen(curandDirectionVectors_sz * sobolDirectionVectors,
    curandStateSobol_sz *state,
    uint_t offset)
{
  //uint_t tid = DIM*(threadIdx.x + blockDim.x * blockIdx.x);
  uint_t tid = DIM*(threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x));

  curand_init(sobolDirectionVectors[0],
              (offset+tid)*OFFSET_MULT,
              &state[tid]);
  if(DIM==2)
    curand_init(sobolDirectionVectors[1], // sobol direction vectors consists of 32 vectors (curandStateScrambledSobol32)
              (offset+tid)*OFFSET_MULT,
              &state[tid+1]);
  if(DIM==3)
      curand_init(sobolDirectionVectors[2], // sobol direction vectors consists of 32 vectors (curandStateScrambledSobol32)
              (offset+tid)*OFFSET_MULT,
              &state[tid+2]);
}

template<uint_t DIM>
__global__ void init_rand_gen(curandDirectionVectors32_t * sobolDirectionVectors,
    uint_t *sobolScrambleConstants,
    curandStateScrambledSobol32_t *state,
    uint_t offset)
{
  //uint_t tid = DIM*(threadIdx.x + blockDim.x * blockIdx.x);
  uint_t tid = DIM*(threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x));

  curand_init(sobolDirectionVectors[0],
              sobolScrambleConstants[0],
              (offset+tid)*OFFSET_MULT,
              &state[tid]);
  if(DIM==2)
    curand_init(sobolDirectionVectors[1], // sobol direction vectors consists of 32 vectors (curandStateScrambledSobol32)
              sobolScrambleConstants[1],
              (offset+tid)*OFFSET_MULT,
              &state[tid+1]);
  if(DIM==3)
      curand_init(sobolDirectionVectors[2], // sobol direction vectors consists of 32 vectors (curandStateScrambledSobol32)
              sobolScrambleConstants[2],
              (offset+tid)*OFFSET_MULT,
              &state[tid+2]);
}


void RandGen<curandStateScrambledSobol32_t>::init(dim3 blocks, dim3 threads, uint_t offset, uint_t dim, uint_t dev)
{
  uint_t max_threads = threads.x * blocks.x * blocks.y;

  CHECK_ERROR(cudaSetDevice(dev));
  CHECK_ERROR_LIB(curandGetDirectionVectors32( &hostVectors, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
  CHECK_ERROR_LIB(curandGetScrambleConstants32( &hostScrambleConstants));

  CHECK_ERROR(cudaMalloc(&devStates, dim*max_threads * sizeof(curandStateScrambledSobol32_t)));
  CHECK_ERROR(cudaMalloc(&devDirectionVectors, dim * sizeof(curandDirectionVectors32_t)));
  CHECK_ERROR(cudaMalloc(&devScrambleConstants, dim * sizeof(uint_t)));

  CHECK_ERROR(cudaMemcpy(devDirectionVectors, hostVectors, dim * sizeof(curandDirectionVectors32_t),
                        cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(devScrambleConstants, hostScrambleConstants, dim * sizeof(uint_t),
                        cudaMemcpyHostToDevice));

  switch(dim)
  {
    case 2:
      init_rand_gen<2><<<blocks, threads>>>(devDirectionVectors, devScrambleConstants, devStates, offset);
      break;
    case 3:
      init_rand_gen<3><<<blocks, threads>>>(devDirectionVectors, devScrambleConstants, devStates, offset);
      break;
    default:
    init_rand_gen<1><<<blocks, threads>>>(devDirectionVectors, devScrambleConstants, devStates, offset);
  }
}

void RandGen<curandStateScrambledSobol32_t>::free()
{
  CHECK_ERROR(cudaFree(devStates));
  CHECK_ERROR(cudaFree(devDirectionVectors));
  CHECK_ERROR(cudaFree(devScrambleConstants));
}

// --------

void RandGen<curandStateSobol32_t>::init(dim3 blocks, dim3 threads, uint_t offset, uint_t dim, uint_t dev)
{
  uint_t max_threads = threads.x * blocks.x * blocks.y;

  CHECK_ERROR(cudaSetDevice(dev));
  CHECK_ERROR_LIB(curandGetDirectionVectors32( &hostVectors, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
  CHECK_ERROR(cudaMalloc(&devStates, dim*max_threads * sizeof(curandStateSobol32_t)));
  CHECK_ERROR(cudaMalloc(&devDirectionVectors, dim * sizeof(curandDirectionVectors32_t)));
  CHECK_ERROR(cudaMemcpy(devDirectionVectors, hostVectors, dim * sizeof(curandDirectionVectors32_t),
                        cudaMemcpyHostToDevice));

  switch(dim)
  {
    case 2:
      init_rand_gen<2><<<blocks, threads>>>(devDirectionVectors, devStates, offset);
      break;
    case 3:
      init_rand_gen<3><<<blocks, threads>>>(devDirectionVectors, devStates, offset);
      break;
    default:
    init_rand_gen<1><<<blocks, threads>>>(devDirectionVectors, devStates, offset);
  }
}

void RandGen<curandStateSobol32_t>::free()
{
  CHECK_ERROR(cudaFree(devStates));
  CHECK_ERROR(cudaFree(devDirectionVectors));
}

// --------

void RandGen<curandStateSobol64_t>::init(dim3 blocks, dim3 threads, uint_t offset, uint_t dim, uint_t dev)
{
  uint_t max_threads = threads.x * blocks.x * blocks.y;

  CHECK_ERROR(cudaSetDevice(dev));
  CHECK_ERROR_LIB(curandGetDirectionVectors64( &hostVectors, CURAND_DIRECTION_VECTORS_64_JOEKUO6));
  CHECK_ERROR(cudaMalloc(&devStates, dim*max_threads * sizeof(curandStateSobol64_t)));
  CHECK_ERROR(cudaMalloc(&devDirectionVectors, dim * sizeof(curandDirectionVectors64_t)));
  CHECK_ERROR(cudaMemcpy(devDirectionVectors, hostVectors, dim * sizeof(curandDirectionVectors64_t),
                        cudaMemcpyHostToDevice));

  switch(dim)
  {
    case 2:
      init_rand_gen<2><<<blocks, threads>>>(devDirectionVectors, devStates, offset);
      break;
    case 3:
      init_rand_gen<3><<<blocks, threads>>>(devDirectionVectors, devStates, offset);
      break;
    default:
    init_rand_gen<1><<<blocks, threads>>>(devDirectionVectors, devStates, offset);
  }
}

void RandGen<curandStateSobol64_t>::free()
{
  CHECK_ERROR(cudaFree(devStates));
  CHECK_ERROR(cudaFree(devDirectionVectors));
}

// --------
template<>
void RandGen<curandState_t>::init(dim3 blocks, dim3 threads, uint_t seed, uint_t dim, uint_t dev)
{
  uint_t max_threads = threads.x * blocks.x * blocks.y;

  CHECK_ERROR(cudaSetDevice(dev));
  CHECK_ERROR(cudaMalloc(&devStates, dim*max_threads * sizeof(curandStateSobol32_t)));

  switch(dim)
  {
    case 2:
      init_rand_gen<2><<<blocks, threads>>>(devStates, seed);
      break;
    case 3:
      init_rand_gen<3><<<blocks, threads>>>(devStates, seed);
      break;
    default:
    init_rand_gen<1><<<blocks, threads>>>(devStates, seed);
  }
}

template<>
void RandGen<curandState_t>::free()
{
  CHECK_ERROR(cudaFree(devStates));
}
