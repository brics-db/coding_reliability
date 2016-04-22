#ifndef HAMMING_H_
#define HAMMING_H_

#include "globals.h"
#include <stdexcept>
namespace Hamming {

inline uintll_t computeHamming08(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (bitcount(value & 0x0000005B) & 0x1) << 1;
  hamming |= (bitcount(value & 0x0000006D) & 0x1) << 2;
  hamming |= (bitcount(value & 0x0000008E) & 0x1) << 3;
  hamming |= (bitcount(value & 0x000000F0) & 0x1) << 4;
  hamming |= (bitcount(value & 0x000000FF) + bitcount(hamming)) & 0x1;
  return (value << 5) | hamming;
}

inline uintll_t computeHamming16(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (bitcount(value & 0x0000AD5B) & 0x1) << 1;
  hamming |= (bitcount(value & 0x0000366D) & 0x1) << 2;
  hamming |= (bitcount(value & 0x0000C78E) & 0x1) << 3;
  hamming |= (bitcount(value & 0x000007F0) & 0x1) << 4;
  hamming |= (bitcount(value & 0x0000F800) & 0x1) << 5;
  hamming |= (bitcount(value & 0x0000FFFF) + bitcount(hamming)) & 0x1;
  return (value << 6) | hamming;
}

inline uintll_t computeHamming24(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (bitcount(value & 0x00AAAD5B) & 0x1) << 1;
  hamming |= (bitcount(value & 0x0033366D) & 0x1) << 2;
  hamming |= (bitcount(value & 0x00C3C78E) & 0x1) << 3;
  hamming |= (bitcount(value & 0x00FC07F0) & 0x1) << 4;
  hamming |= (bitcount(value & 0x00FFF800) & 0x1) << 5;
  hamming |= (bitcount(value & 0x00FFFFFF) + bitcount(hamming)) & 0x1;
  return (value << 6) | hamming;
}

inline uintll_t computeHamming32(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (bitcount(value & 0x56AAAD5B) & 0x1) << 1;
  hamming |= (bitcount(value & 0x9B33366D) & 0x1) << 2;
  hamming |= (bitcount(value & 0xE3C3C78E) & 0x1) << 3;
  hamming |= (bitcount(value & 0x03FC07F0) & 0x1) << 4;
  hamming |= (bitcount(value & 0x03FFF800) & 0x1) << 5;
  hamming |= (bitcount(value & 0xFC000000) & 0x1) << 6;
  hamming |= (bitcount(value & 0xFFFFFFFF) + bitcount(hamming)) & 0x1;
  return (value << 7) | hamming;
}

template<typename T>
inline uintll_t computeDistance(const T &value1, const T &value2) {
  return static_cast<uintll_t>(bitcount(value1 ^ value2));
}

namespace traits {
  
  // trait for Shards sizes
  template<uintll_t N>
  struct Shards: std::integral_constant<uintll_t,1>{};
  template<> struct Shards<8>: std::integral_constant<uintll_t,1>{};
  template<> struct Shards<16>: std::integral_constant<uintll_t,16>{};
  template<> struct Shards<24>: std::integral_constant<uintll_t,128>{};
  template<> struct Shards<32>: std::integral_constant<uintll_t,256>{};
  template<> struct Shards<40>: std::integral_constant<uintll_t,512>{};
  template<> struct Shards<48>: std::integral_constant<uintll_t,1024>{};
  // trait for size of array counts[]
  template<uintll_t N>
  struct CountCounts: std::integral_constant<int,14>{};
  template<> struct CountCounts<8>: std::integral_constant<int,14>{};
  template<> struct CountCounts<16>: std::integral_constant<int,23>{};
  template<> struct CountCounts<24>: std::integral_constant<int,31>{};
  template<> struct CountCounts<32>: std::integral_constant<int,40>{};
  template<> struct CountCounts<40>: std::integral_constant<int,48>{};
  template<> struct CountCounts<48>: std::integral_constant<int,56>{};

} // traits

template<template<uintll_t> class TFunctor, typename... Types>
inline void bridge(uint_t n, Types... args)
{
  switch(n){
  case 8:  return TFunctor<8>()(args...);
  case 16: return TFunctor<16>()(args...); 
  case 24: return TFunctor<24>()(args...); 
  case 32: return TFunctor<32>()(args...); 
  case 40: return TFunctor<40>()(args...); 
  case 48: return TFunctor<48>()(args...); 
  }
  throw std::runtime_error("Unsupported n.");
}
template<uintll_t N>
struct GetShardsSize
{
  void operator()(uintll_t* result){ *result = traits::Shards<N>::value; }
};
inline uintll_t getShardsSize(uint_t n)
{
  uintll_t result = 0;
  bridge< GetShardsSize >(n, &result);
  return result;
}
} // Hamming
#endif
