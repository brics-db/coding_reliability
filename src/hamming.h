#ifndef HAMMING_H_
#define HAMMING_H_

#include "globals.h"

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
#endif
