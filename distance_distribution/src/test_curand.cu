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
#include "algorithms.h"
#include "rand_gen.cuh"

#include <helper.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>
#include <ostream>

#include <curand.h>
#include <curand_kernel.h>


const int USE_ORDER=0; // for 2d with y>x mapping

using namespace std;
template<uint_t N, int USE_ORDER, typename RandGenType>
__global__
void dtest(uint_t *const test, RandGenType *const state, const uint_t iterations, const uint_t max_threads)
{
  uint_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid>=max_threads)
    return;

  uintll_t v=0;
  uintll_t w=0;
  uint_t it = 0;
  RandGenType local_state_x = state[2*tid];
  RandGenType local_state_y = state[2*tid+1];
  uint_t index = 0;
  while(it<iterations)
  {
    index = 2*(it*max_threads+tid);
    if(USE_ORDER)
      do{
        //v = static_cast<uintll_t>( curand(&local_state)) & ((1ull<<N)-1); // @todo early collisions possible for N<32
        v = static_cast<uintll_t>( (static_cast<float>(1ull<<N)-2) * curand_uniform(&local_state_x) + 0.5f);
        w = static_cast<uintll_t>( (static_cast<float>(1ull<<N)-2) * curand_uniform(&local_state_y) + 0.5f);
        w+=1;
        test[index]   = v;
        test[index+1] = w;
      }while(v>=w);
    else{
      v = static_cast<uintll_t>( (static_cast<float>(1ull<<N)-1) * curand_uniform(&local_state_x) + 0.5f);
      w = static_cast<uintll_t>( (static_cast<float>(1ull<<N)-1) * curand_uniform(&local_state_y) + 0.5f);
      test[index]   = v;
      test[index+1] = w;
    }
    ++it;
  }
  state[2*tid] = local_state_x;
  state[2*tid+1] = local_state_y;
}


template<uint_t N, int USE_ORDER, typename RandGenType>
__global__
void dtest_1d(uint_t *const test, RandGenType *const state, const uint_t iterations, const uint_t max_threads)
{
  uint_t tid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x);
  if(tid>=max_threads)
    return;
  uint_t it = 0;
  uint_t index=0;
  uintll_t z=0;
  uintll_t v=0;
  uintll_t w=0;
  const uintll_t MSGS=(1ull<<N);
  const uintll_t nr_pairs = USE_ORDER==1 ? (MSGS-1)*(1ull<<(N-1)) : uintll_t(uint128_t(1)<<(2*N))-1;
  RandGenType local_state_x = state[tid];

  while(it<iterations)
  {
    index = 2*(it*max_threads+tid);
    z = static_cast<uintll_t>( static_cast<float>(nr_pairs) * curand_uniform(&local_state_x) );
    if(USE_ORDER)
    {
      double val = MSGS;
      double val_z = z;
      // todo: fix (v does not fill to 2^N)
      v = uintll_t(val+0.5-sqrt((val+0.5)*(val+0.5)+2.0*(1.0-val_z)));
      w = z+v+v*(v-1)/2-MSGS*v;
    }else{
      // todo: zeros appear too often
      v = z/MSGS;
      w = z&(MSGS-1);
    }
    test[index]   = v;
    test[index+1] = w;
    ++it;
  }
  state[tid] = local_state_x;
}

template<uint_t N, typename RandGenType>
__global__
void dtest_raw(uint_t *const test, RandGenType *const state, const uint_t iterations, const uint_t max_threads)
{
  uint_t tid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x);
  uint_t it = 0;
  uint_t z=0;
  if(tid>=max_threads)
    return;

  const uintll_t MSGS=(1ull<<N);
  RandGenType local_state_x = state[tid];

  while(it<iterations)
  {    
    z = static_cast<uint_t>( static_cast<float>(MSGS) * curand_uniform(&local_state_x) );
   
    test[it*max_threads + tid]   = z;

    ++it;
  }
  state[tid] = local_state_x;
}

extern void test_curand(uintll_t n, uintll_t iterations, int max_nr_dev)
{
  int tmp_nr_dev;
  RandGen<RAND_GEN> gen;
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats,GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);


  printf("Start cuRAND Test with %zu iterations\n", iterations);
  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = max_nr_dev==0 ? tmp_nr_dev : min(tmp_nr_dev,max_nr_dev);
  cudaDeviceProp prop;
  printf("Found %d CUDA devices.\n", nr_dev);
  results_cpu.start(i_totaltime);

  // curand sobol generator parameters give limit max_threads<20k/dims
  dim3 threads(128, 1, 1);
  uintll_t nr_pairs = USE_ORDER==1 ? ((1ull<<n)-1)*(1ull<<(n-1)) : uintll_t(uint128_t(1)<<(2*n))-1;
  //dim3 blocks(min(10000ull,nr_pairs/nr_dev/iterations+threads.x-1)/threads.x, 1, 1);
  dim3 blocks((nr_pairs/nr_dev/iterations+threads.x-1)/threads.x, 1, 1);
  uint_t max_threads = threads.x * blocks.x;
  //printf("%u\n", max_threads);
  uint_t numbers_length = 2*iterations*max_threads;
  uint_t* numbers = new uint_t[nr_dev*numbers_length];
  memset(numbers,0,nr_dev*numbers_length*sizeof(uint_t));

#pragma omp parallel for num_threads(nr_dev) schedule(dynamic)
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );

    gen.init(blocks, threads, 1337, 2, dev);

    printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
    if(dev==0)
      results_gpu.start(i_runtime);

    uint_t* d_test = NULL;
    CHECK_ERROR(cudaMalloc(&d_test,numbers_length*sizeof(uint_t)));
    CHECK_ERROR(cudaMemset(d_test,0,numbers_length*sizeof(uint_t)));

    switch(n)
    {
      case 16:
        dtest<16,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
        break;
      case 24:
        dtest<24,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
        break;
      case 32:
        dtest<32,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
        break;
      default:
      dtest<8,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
    }

    CHECK_LAST("Kernel failed.");
    CHECK_ERROR(cudaMemcpy(numbers+dev*numbers_length,d_test,numbers_length*sizeof(uint_t),cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_test));

    if(dev==0)
      results_gpu.stop(i_runtime);

    gen.free();
  }
  results_cpu.stop(i_totaltime);


  FILE* fp;
  fp = fopen("sobol_out.csv","w");

  uintll_t offset;
  uint_t nr_rands = 0;
  for(uintll_t i=0; i<iterations; ++i)
  {
    for(int dev=0; dev<nr_dev; ++dev)
    {
      offset = 2*(i * max_threads) + uintll_t(dev)*numbers_length;
      for(uintll_t k=0; k<max_threads; ++k,offset+=2){
        fprintf(fp,"%zd,%u,%u\n", k, numbers[offset], numbers[1+offset]);
        ++nr_rands;
      }
    }
    fprintf(fp,"\n\n");
  }
  fclose(fp);
  printf("%u/%llu random pairs generated%s.\n", nr_rands, nr_pairs,USE_ORDER?" (sorted)":"");

  for(int i=0; i<stats.getLength(); ++i)
    printf("%s %7.2lf %s\n", stats.getLabel(i).c_str(), stats.getAverage(i), stats.getUnit(i).c_str());

  CHECK_ERROR( cudaDeviceReset() );
}




extern void test_curand_1d(uintll_t n, uintll_t iterations, int max_nr_dev)
{
  int tmp_nr_dev;
  RandGen<RAND_GEN> gen;
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats,GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);


  printf("Start cuRAND Test with %zu iterations\n", iterations);
  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = max_nr_dev==0 ? tmp_nr_dev : min(tmp_nr_dev,max_nr_dev);
  cudaDeviceProp prop;
  printf("Found %d CUDA devices.\n", nr_dev);
  results_cpu.start(i_totaltime);

  // curand sobol generator parameters give limit max_threads<20k/dims
  dim3 threads(128, 1, 1);
  uintll_t nr_pairs = USE_ORDER==1 ? ((1ull<<n)-1)*(1ull<<(n-1)) : uintll_t(uint128_t(1)<<(2*n))-1;
  //dim3 blocks(min(20000ull,nr_pairs/nr_dev/iterations+threads.x-1)/threads.x, 1, 1);
  dim3 blocks(min(16384ull,(nr_pairs/nr_dev/iterations+threads.x-1)/threads.x), 1, 1);
  blocks.y = floor(sqrt(blocks.x)+0.5);
  blocks.x = blocks.y;

  uint_t max_threads = threads.x * blocks.x * blocks.y;
  //printf("%u\n", max_threads);
  uint_t numbers_length = 2*iterations*max_threads;
  uint_t* numbers = new uint_t[nr_dev*numbers_length];
  memset(numbers,0,nr_dev*numbers_length*sizeof(uint_t));

#pragma omp parallel for num_threads(nr_dev) schedule(dynamic)
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );

    gen.init(blocks, threads, 1337, 1, dev);

    printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
    printf("Dev %d: Blocks: %d %d, max threads %u\n", dev, blocks.x, blocks.y, max_threads);

    if(dev==0)
      results_gpu.start(i_runtime);

    uint_t* d_test = NULL;
    CHECK_ERROR(cudaMalloc(&d_test,numbers_length*sizeof(uint_t)));
    CHECK_ERROR(cudaMemset(d_test,0,numbers_length*sizeof(uint_t)));

    switch(n)
    {
      case 16:
        dtest_1d<16,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
        break;
      case 24:
        dtest_1d<24,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
        break;
      case 32:
        dtest_1d<32,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
        break;
      default:
        dtest_1d<8,USE_ORDER><<<blocks, threads>>>(d_test,gen.devStates,iterations,max_threads);
    }

    CHECK_LAST("Kernel failed.");
    CHECK_ERROR(cudaMemcpy(numbers+dev*numbers_length,d_test,numbers_length*sizeof(uint_t),cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_test));

    if(dev==0)
      results_gpu.stop(i_runtime);

    gen.free();
  }
  results_cpu.stop(i_totaltime);


  FILE* fp;
  fp = fopen("sobol_out.csv","w");

  uintll_t offset;
  uint_t nr_rands = 0;
  for(uintll_t i=0; i<iterations; ++i)
  {
    for(int dev=0; dev<nr_dev; ++dev)
    {
      offset = 2*(i * max_threads) + uintll_t(dev)*numbers_length;
      for(uintll_t k=0; k<max_threads; ++k, offset+=2){
        fprintf(fp,"%zd,%u,%u\n", k, numbers[offset], numbers[offset+1]);
        ++nr_rands;
      }
    }
    fprintf(fp,"\n\n");
  }
  fclose(fp);
  printf("%u/%llu random pairs generated%s.\n", nr_rands, nr_pairs,USE_ORDER?" (sorted)":"");

  for(int i=0; i<stats.getLength(); ++i)
    printf("%s %7.2lf %s\n", stats.getLabel(i).c_str(), stats.getAverage(i), stats.getUnit(i).c_str());

  CHECK_ERROR( cudaDeviceReset() );
}


extern void test_curand_raw(uintll_t n, uintll_t iterations, int max_nr_dev)
{
  int tmp_nr_dev;
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats,GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);


  printf("Start cuRAND Test Raw with %zu iterations\n", iterations);
  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = max_nr_dev==0 ? tmp_nr_dev : min(tmp_nr_dev,max_nr_dev);
  cudaDeviceProp prop;
  printf("Found %d CUDA devices.\n", nr_dev);
  results_cpu.start(i_totaltime);

  const dim3 threads(128, 1, 1);
  const uintll_t msgs = 1ull<<n;
  const uint_t xblocks = ceil(sqrt(msgs/nr_dev/threads.x));
  const dim3 blocks(xblocks, xblocks);
  const uint_t max_threads = min(msgs, (uintll_t)threads.x * blocks.x * blocks.y);

  const uint_t numbers_length = iterations*max_threads;
  uint_t* numbers = new uint_t[nr_dev*numbers_length];
  memset(numbers,0,nr_dev*numbers_length*sizeof(uint_t));


#pragma omp parallel for num_threads(nr_dev) schedule(dynamic)
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );

    RandGen<RAND_GEN> gen;
    gen.init(blocks, threads, 1337+8137*xblocks*xblocks*threads.x*dev, 1, dev);

    printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
    printf("Dev %d: Blocks: %d %d, max threads %u\n", dev, blocks.x, blocks.y, max_threads);
    if(dev==0)
      results_gpu.start(i_runtime);

    uint_t* d_test = NULL;
    CHECK_ERROR(cudaMalloc(&d_test,numbers_length*sizeof(uint_t)));
    CHECK_ERROR(cudaMemset(d_test,0,numbers_length*sizeof(uint_t)));

    switch(n)
    {
      case 16:
        dtest_raw<16><<<blocks, threads>>>(d_test,gen.devStates,iterations, max_threads);
        break;
      case 24:
        dtest_raw<24><<<blocks, threads>>>(d_test,gen.devStates,iterations, max_threads);
        break;
      case 32:
        dtest_raw<32><<<blocks, threads>>>(d_test,gen.devStates,iterations, max_threads);
        break;
      default:
        dtest_raw<8><<<blocks, threads>>>(d_test,gen.devStates,iterations, max_threads);
    }

    CHECK_LAST("Kernel failed.");
    CHECK_ERROR(cudaMemcpy(numbers+dev*numbers_length,d_test,numbers_length*sizeof(uint_t),cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_test));

    if(dev==0)
      results_gpu.stop(i_runtime);

    gen.free();
  }
  results_cpu.stop(i_totaltime);


  FILE* fp;
  fp = fopen("sobol_out.csv","w");

  uintll_t offset;
  uint_t nr_rands = 0;
  for(uintll_t i=0; i<iterations; ++i)
  {
    for(int dev=0; dev<nr_dev; ++dev)
    {
      offset = i * max_threads + uintll_t(dev)*numbers_length;
      for(uintll_t k=0; k<max_threads; ++k){
        fprintf(fp,"%zd,%u\n", k, numbers[k+offset]);
        ++nr_rands;
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
  printf("%u/%llu random values generated.\nsobol_out.csv written.\n", nr_rands, msgs*msgs);

  for(int i=0; i<stats.getLength(); ++i)
    printf("%s %7.2lf %s\n", stats.getLabel(i).c_str(), stats.getAverage(i), stats.getUnit(i).c_str());

  CHECK_ERROR( cudaDeviceReset() );
}
