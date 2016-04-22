#ifndef AN_CODING_H_
#define AN_CODING_H_
#include "hamming.h"
namespace ANCoding {
namespace traits {
  
  // trait for Shards sizes
  using Hamming::traits::Shards;
  // trait for size of array counts[]
  template<uintll_t N>
  struct CountCounts: std::integral_constant<int,16>{};
  template<> struct CountCounts<8>: std::integral_constant<int,16>{};
  template<> struct CountCounts<16>: std::integral_constant<int,32>{};
  template<> struct CountCounts<24>: std::integral_constant<int,48>{};
  template<> struct CountCounts<32>: std::integral_constant<int,64>{};
  template<> struct CountCounts<40>: std::integral_constant<int,64>{};
  template<> struct CountCounts<48>: std::integral_constant<int,64>{};

} // traits

using Hamming::bridge;
using Hamming::getShardsSize;

} // ANCoding


#endif
