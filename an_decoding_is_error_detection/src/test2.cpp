#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <immintrin.h>

template<typename T>
T
ext_euklidean(T b0, size_t codewidth)
{
	T a0(1);
	a0 <<= codewidth;
	std::vector<T> a, b, q, r, s, t;
	a.push_back(a0), b.push_back(b0), s.push_back(T(0)), t.push_back(T(0));
	size_t i = 0;
	do
	{
		q.push_back(a[i] / b[i]);
		r.push_back(a[i] % b[i]);
		a.push_back(b[i]);
		b.push_back(r[i]);
		s.push_back(0);
		t.push_back(0);
	}
	while (b[++i] > 0);
	s[i] = 1;
	t[i] = 0;

	for (size_t j = i; j > 0; --j)
	{
		s[j - 1] = t[j];
		t[j - 1] = s[j] - q[j - 1] * t[j];
	}

	T result = ((b0 * t.front()) % a0);
	result += result < 0 ? a0 : 0;
	if (result == 1)
	{
		return t.front();
	}
	else
	{
		return 0;
	}
}

int main() {
	const ssize_t A = 3;
	const size_t Awidth = 2;
	const size_t Dwidth = 4;
	const ssize_t Dmin = (-1) * (1ll << (Dwidth - 1));
	const ssize_t Dmax = (1ll << (Dwidth - 1)) - 1;
	const ssize_t Cwidth = Awidth + Dwidth;
	const ssize_t AInv = ext_euklidean(A, Cwidth) & ((1ll << Cwidth) - 1);
	const size_t numCodewords = (1ull << Cwidth);
	std::cout << "numCodewords: " << numCodewords << std::endl;
	const size_t numMasks = numCodewords / (sizeof(int) * 4); // How many masks will we generate?
	int * pNonCodewordMasks = new int[numMasks];
	const int16_t c = ~((1ll << (Cwidth - 1)) - 1);
	std::cout << "c = 0x" << std::hex << c << std::dec << std::endl;
	for (ssize_t i = 0, cw = c, posMask = 0; i < numCodewords; ++posMask) {
		int tmpMask = 0;
		for (ssize_t k = 0; k < 16; ++k, ++cw, ++i) {
			if ((cw % A) != 0) { // we want the non-codewords
				// std::cout << "cw % A != 0: " << cw << std::endl;
				tmpMask |= (1ll << (k * 2)) | (1ll << (k * 2 + 1)); // expand to 32 bits, because AVX2 cannot movemask across lanes to 16 bits
			}
		}
		pNonCodewordMasks[posMask] = tmpMask;
	}
	std::cout << "numMasks: " << numMasks << std::endl;
	std::cout << "non-codeword-masks: 0x" << std::hex << std::setfill('0');
	for (size_t posMask = 0; posMask < numMasks; ++posMask) {
		std::cout << std::setw(8) << pNonCodewordMasks[posMask] << ':';
	}
	std::cout << std::dec << std::endl << std::setfill(' ');
	auto mmCodewords = _mm256_set_epi16(c+15, c+14, c+13, c+12, c+11, c+10, c+9, c+8, c+7, c+6, c+5, c+4, c+3, c+2, c+1, c);
	auto mmAddUp = _mm256_set1_epi16(16);
	auto mmAinv = _mm256_set1_epi16(AInv);
	auto mmDmin = _mm256_set1_epi16(Dmin);
	auto mmDmax = _mm256_set1_epi16(Dmax);
	const size_t posEnd = (1ull << Cwidth);
	__m256i mmFillUp[] = {_mm256_set1_epi16(0), _mm256_set1_epi16(~((1ll << Cwidth) - 1))}; // fill up all non-codeword bits with 1's if necessary
	std::cout << "posEnd = 0x" << std::hex << posEnd << std::dec << std::endl;
	std::cout << std::setfill('0') << std::hex;
	for(size_t pos = 15, posMask = 0; pos < posEnd; pos += 16, ++posMask) {
		auto isNeg = 0x1 & _mm256_movemask_epi8(_mm256_cmpgt_epi16(mmFillUp[0], mmCodewords));
		auto mm1 = _mm256_or_si256(_mm256_mullo_epi16(mmCodewords, mmAinv), mmFillUp[isNeg]);
		auto mm2 = _mm256_cmpgt_epi16(mm1, mmDmin);
		auto mm3 = _mm256_cmpgt_epi16(mmDmax, mm1);
		auto mm4 = _mm256_cmpeq_epi16(mmDmax, mm1);
		auto mm5 = _mm256_or_si256(mm3, mm4);
		auto mm6 = _mm256_and_si256(mm2, mm5);
		auto mask = _mm256_movemask_epi8(mm6);
		if (mask & pNonCodewordMasks[posMask]) {
			std::cout << "BAD @0x" << std::setw((Cwidth + 7) / 8) << pos << ": 0x" << mask << " & 0x" << pNonCodewordMasks[posMask] << " = 0x" << (mask & pNonCodewordMasks[posMask]) << std::endl;
		} else {
			std::cout << "OK @0x" << std::setw((Cwidth + 7) / 8) << pos << ": 0x" << mask << " & 0x" << pNonCodewordMasks[posMask] << " = 0x" << (mask & pNonCodewordMasks[posMask]) << std::endl;
		}
		mmCodewords = _mm256_add_epi16(mmCodewords, mmAddUp);
	}
	std::cout << std::setfill(' ') << std::dec;
}

