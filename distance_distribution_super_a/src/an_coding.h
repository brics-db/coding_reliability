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

#ifndef AN_CODING_H_
#define AN_CODING_H_

namespace ANCoding {

namespace traits {
  // trait for Shards sizes
  template<uintll_t N>
  struct Shards: std::integral_constant<uintll_t,1>{};
  template<> struct Shards<8>: std::integral_constant<uintll_t,1>{};
  template<> struct Shards<16>: std::integral_constant<uintll_t,1>{};
  template<> struct Shards<24>: std::integral_constant<uintll_t,2>{};
  template<> struct Shards<32>: std::integral_constant<uintll_t,256>{};
  template<> struct Shards<40>: std::integral_constant<uintll_t,512>{};
  template<> struct Shards<48>: std::integral_constant<uintll_t,1024>{};


  // trait for size of array counts[]
  template<uintll_t N>
  struct CountCounts: std::integral_constant<int,16>{};
  template<> struct CountCounts<8>: std::integral_constant<int,32>{};
  template<> struct CountCounts<16>: std::integral_constant<int,32>{};
  template<> struct CountCounts<24>: std::integral_constant<int,32>{};
  template<> struct CountCounts<32>: std::integral_constant<int,64>{};
  template<> struct CountCounts<40>: std::integral_constant<int,64>{};
  template<> struct CountCounts<48>: std::integral_constant<int,64>{};

} // traits

template<template<uintll_t> class TFunctor, typename... Types>
inline void bridge(uint_t n, Types... args)
{
  uint_t n_up = n<=8 ? 8 : n<=16 ? 16 : n<=24 ? 24 : n<=32 ? 32 : n<=40 ? 40 : 48;
  switch(n_up){
  case 8:  return TFunctor<8>()(n, args...);
  case 16: return TFunctor<16>()(n, args...); 
  case 24: return TFunctor<24>()(n, args...); 
  case 32: return TFunctor<32>()(n, args...); 
  case 40: return TFunctor<40>()(n, args...); 
  case 48: return TFunctor<48>()(n, args...); 
  }
  throw std::runtime_error("Unsupported n.");
}
template<uintll_t N>
struct GetShardsSize
{
  void operator()(uint_t n, uintll_t* result){ *result = traits::Shards<N>::value; }
};
inline uintll_t getShardsSize(uint_t n)
{
  uintll_t result = 0;
  bridge< GetShardsSize >(n, &result);
  return result;
}
template<uintll_t N>
struct GetCountCounts
{
  void operator()(uint_t n, uintll_t* result){ *result = traits::CountCounts<N>::value; }
};
inline uintll_t getCountCounts(uint_t n)
{
  uintll_t result = 0;
  bridge< GetCountCounts >(n, &result);
  return result;
}


__device__
inline uint_t dbitcount(uint_t v) {
  return __popc(v);
}
__device__
inline uint_t dbitcount(uintll_t v) {
  return __popcll(v);
}


} // ANCoding


#endif
