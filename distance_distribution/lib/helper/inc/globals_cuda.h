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

#ifndef GLOBALS_CUDA_H_
#define GLOBALS_CUDA_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/*---------------------------------------------------------------------------*/
#define CUDA_ERROR_CHECKING 1
#define LAST_CUDA_ERROR_CHECKING 1
/*---------------------------------------------------------------------------*/

#if CUDA_ERROR_CHECKING==1
  #define CHECK_ERROR(ans) globals_check_error((ans), #ans, __FILE__, __LINE__)
  #define CHECK_ERROR_LIB(ans) globals_check_error_lib((ans), #ans, __FILE__, __LINE__)
#else
  #define CHECK_ERROR(ans) {}
  #define CHECK_ERROR_LIB(ans) {}
#endif
#if CUDA_ERROR_CHECKING==1 && LAST_CUDA_ERROR_CHECKING==1
#define CHECK_LAST(msg) globals_check_error_last(msg, __FILE__, __LINE__)
#else
#define CHECK_LAST(msg) {}
#endif

/*---------------------------------------------------------------------------*/

inline
void globals_check_error(cudaError_t code, const char *func, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error '%s' at %s:%d (%s)\n", cudaGetErrorString(code), file, line, func);
      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
   }
}
inline
void globals_check_error_last(const char *msg, const char *file, int line)
{
  cudaError_t code = cudaGetLastError();
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error '%s' at %s:%d (%s)\n", cudaGetErrorString(code), file, line, msg);

      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
   }
}
template<typename T>
void globals_check_error_lib(T code, const char *func, const char *file, int line)
{
   if (code)
   {
      fprintf(stderr,"CUDA Lib Error '%d' at %s:%d (%s)\n", static_cast<unsigned int>(code), file, line, func);

      cudaDeviceReset();
      exit(static_cast<unsigned int>(code));
   }
}


#endif /* GLOBALS_CUDA_H_ */
