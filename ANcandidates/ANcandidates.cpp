/*
 * Compile e.g. with
 *   g++ -std=c++17 -O3 -march=native -o ANcandidates_d2-24_A2-24 ANcandidates.cpp -fopenmp -lpthread
 * Run e.g. with
 *   /usr/bin/time /bin/bash -c "OMP_THREAD_LIMIT=\$(nproc) ./ANcandidates_d2-24_A2-24 1>ANcandidates_d2-24_A2-24.out 2>ANcandidates_d2-24_A2-24.err"
 *   /usr/bin/time /bin/bash -c "OMP_THREAD_LIMIT=\$(echo \"\$(nproc)-2\"|bc) ./ANcandidates_d2-24_A2-24 1>ANcandidates_d2-24_A2-24.out 2>ANcandidates_d2-24_A2-24.err"
 */

#include <iostream>
#include <cstdint>
#include <limits>
#include <vector>
#include <array>
#include <sstream>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>

#ifdef __GNUC__
#define popcount__ __builtin_popcountll
#endif

struct data_t {
	size_t lengthSignedDigitsRepresentation;
	std::vector<ssize_t> elements;
	data_t() : lengthSignedDigitsRepresentation(std::numeric_limits<size_t>::min()), elements() {
		elements.reserve(64);
	}
};

/**
 * This is the actual worker and computes the signed digit representation for a given code in a 4-unrolled loop.
 */
void worker(data_t & data, size_t bitWidthData, size_t bitWidthA, ssize_t currentA) {
	constexpr const size_t MAX_SSDR = std::numeric_limits<size_t>::max();
	volatile auto lenShortestSignedDigitsRepresentation = MAX_SSDR;
	ssize_t min = currentA;
	const ssize_t max = ((0x1ll << bitWidthData) - 1ll) * currentA;
	const size_t currentDataLengthSignedDigitsRepresentation = data.lengthSignedDigitsRepresentation;
/*
#ifdef __AVX512F__
	if constexpr(sizeof(ssize_t) == 8) {
		const ssize_t num = (max - min) / currentA;
		constexpr const size_t numSSIZEperM512 = sizeof(__m512i) / sizeof(ssize_t);
		const size_t numMM512 = (num / numSSIZEperM512) / 2; // we will unroll the loop twice
		auto mm512_current = _mm512_set_epi64(8 * currentA, 7 * currentA, 6 * currentA, 5 * currentA, 4 * currentA, 3 * currentA, 2 * currentA, 1 * currentA);
		auto mm512_incA = _mm512_set1_epi64(numSSIZEperM512 * currentA);
		for (size_t i = 0; i < numMM512; ++i) { // we already account for two mm operations in computation of numMM512
			auto mm1 = _mm512_xor_si512(mm512_current, _mm512_add_epi64(mm512_current, _mm512_slli_epi64(mm512_current, 1)));
			mm512_current = _mm512_add_epi64(mm512_current, mm512_incA);
			auto mm2 = _mm512_xor_si512(mm512_current, _mm512_add_epi64(mm512_current, _mm512_slli_epi64(mm512_current, 1)));
			mm512_current = _mm512_add_epi64(mm512_current, mm512_incA);
			bool noEarlyBreak = true;
			auto pMM = reinterpret_cast<ssize_t*>(&mm1);
			auto popcnt0 = popcount__(pMM[0]);
			auto popcnt1 = popcount__(pMM[1]);
			auto popcnt2 = popcount__(pMM[2]);
			auto popcnt3 = popcount__(pMM[3]);
			auto popcnt4 = popcount__(pMM[4]);
			auto popcnt5 = popcount__(pMM[5]);
			auto popcnt6 = popcount__(pMM[6]);
			auto popcnt7 = popcount__(pMM[7]);
			pMM = reinterpret_cast<ssize_t*>(&mm2);
			auto popcnt8 = popcount__(pMM[8]);
			auto popcnt9 = popcount__(pMM[9]);
			auto popcnt10 = popcount__(pMM[10]);
			auto popcnt11 = popcount__(pMM[11]);
			auto popcnt12 = popcount__(pMM[12]);
			auto popcnt13 = popcount__(pMM[13]);
			auto popcnt14 = popcount__(pMM[14]);
			auto popcnt15 = popcount__(pMM[15]);
			size_t popcnt = std::min({popcnt0, popcnt1, popcnt2, popcnt3, popcnt4, popcnt5, popcnt6, popocnt7, popcnt8,
				popcnt9, popcnt10, popcnt11, popcnt12, popcnt13, popcnt14, popcnt15});
			if (popcnt < lenShortestSignedDigitsRepresentation) {
				lenShortestSignedDigitsRepresentation = popcnt;
				noEarlyBreak = popcnt >= currentDataLengthSignedDigitsRepresentation;
			}
			if (!noEarlyBreak) {
				break;
			}
		}
		min = _mm256_extract_epi64(_mm512_extracti64x4_epi64(mm512_current, 0), 0) + currentA; // to start with next number in the scalar loop
	}
#endif
#ifdef __AVX2__
	if constexpr(sizeof(ssize_t) == 8) {
		const ssize_t num = (max - min) / currentA;
		constexpr const size_t numSSIZEperM256 = sizeof(__m256i) / sizeof(ssize_t);
		const size_t numMM256 = (num / numSSIZEperM256) / 2; // we will unroll the loop twice
		auto mm256_current = _mm256_set_epi64x(4 * currentA, 3 * currentA, 2 * currentA, 1 * currentA);
		auto mm256_incA = _mm256_set1_epi64x(numSSIZEperM256 * currentA);
		for (size_t i = 0; i < numMM256; ++i) { // we already account for two mm operations in computation of numMM256
			auto mm1 = _mm256_xor_si256(mm256_current, _mm256_add_epi64(mm256_current, _mm256_slli_epi64(mm256_current, 1)));
			mm256_current = _mm256_add_epi64(mm256_current, mm256_incA);
			auto mm2 = _mm256_xor_si256(mm256_current, _mm256_add_epi64(mm256_current, _mm256_slli_epi64(mm256_current, 1)));
			mm256_current = _mm256_add_epi64(mm256_current, mm256_incA);
			bool noEarlyBreak = true;
			auto pMM = reinterpret_cast<ssize_t*>(&mm1);
			auto popcnt0 = popcount__(pMM[0]);
			auto popcnt1 = popcount__(pMM[1]);
			auto popcnt2 = popcount__(pMM[2]);
			auto popcnt3 = popcount__(pMM[3]);
			pMM = reinterpret_cast<ssize_t*>(&mm2);
			auto popcnt4 = popcount__(pMM[0]);
			auto popcnt5 = popcount__(pMM[1]);
			auto popcnt6 = popcount__(pMM[2]);
			auto popcnt7 = popcount__(pMM[3]);
			size_t popcnt = std::min({popcnt0, popcnt1, popcnt2, popcnt3, popcnt4, popcnt5, popcnt6, popcnt7});
			if (popcnt < lenShortestSignedDigitsRepresentation) {
				lenShortestSignedDigitsRepresentation = popcnt;
				noEarlyBreak = popcnt >= currentDataLengthSignedDigitsRepresentation;
			}
			if (!noEarlyBreak) {
				break;
			}
		}
		min = _mm256_extract_epi64(mm256_current, 0); // to start with next number in the scalar loop
	}
#endif
*/
	const auto factorA = 4 * currentA;
	ssize_t current = min;
	size_t tmp[2]{lenShortestSignedDigitsRepresentation, 0};
	while ((current <= (max - factorA)) && (tmp[0] >= currentDataLengthSignedDigitsRepresentation)) {
		auto current0 = current;
		auto popcnt0 = popcount__(current0 ^ (3ll * current0));
		auto current1 = current0 + currentA;
		auto popcnt1 = popcount__(current1 ^ (3ll * current1));
		auto current2 = current1 + currentA;
		auto popcnt2 = popcount__(current2 ^ (3ll * current2));
		auto current3 = current2 + currentA;
		auto popcnt3 = popcount__(current3 ^ (3ll * current3));
		current += factorA;
		size_t popcnt = std::min({popcnt0, popcnt1, popcnt2, popcnt3});
		tmp[1] = popcnt;
		tmp[0] = tmp[popcnt < tmp[0]];
	}
	for (; (current <= max) && (tmp[0] >= currentDataLengthSignedDigitsRepresentation); current += currentA) {
		size_t popcnt = popcount__(current ^ (3ll * current));
		tmp[1] = popcnt;
		tmp[0] = tmp[popcnt < tmp[0]];
	}

	if ((tmp[0] != MAX_SSDR) && (tmp[0] >= data.lengthSignedDigitsRepresentation))
	{ // only enter the following critical section iff we want to alter data
#pragma omp critical
		{
			if (tmp[0] > data.lengthSignedDigitsRepresentation) {
				data.lengthSignedDigitsRepresentation = tmp[0];
				data.elements.clear();
			}
			if (tmp[0] == data.lengthSignedDigitsRepresentation) {
				data.elements.push_back(currentA);
			}
		}
	}
}

/*
 * Use the Symmetric Signed-Digit Representation With Base 2 to find the candidates for Super As
 * in a given inclusive range.
 */
int main() {
	constexpr const size_t minBitWidthData = 2ull;
	constexpr const size_t maxBitWidthData = 18ull;
	constexpr const size_t minBitWidthA = 2ull;
	constexpr const size_t maxBitWidthA = 18ull;
	constexpr const size_t numWidthsData = maxBitWidthData - minBitWidthData + 1ull;
	constexpr const size_t numWidthsA = maxBitWidthA - minBitWidthA + 1ull;
	constexpr const size_t maxCells = numWidthsData * numWidthsA;

	std::cout << "# bit width data = [" << minBitWidthData << ',' << maxBitWidthData << "]\n";
	std::cout << "# bit width A = [" << minBitWidthA << ',' << maxBitWidthA << "]\n";
	std::cout << "# n=|C|\tk=|D|\t|A|\t|SDR|\t|candidates|\tcandidates\n";

	for (size_t cell = 0ull; cell < maxCells; ++cell) {
		const size_t rowData = cell % numWidthsData;
		const size_t colA = cell / numWidthsData;
		data_t data;
		const size_t bitWidthData = rowData + minBitWidthData;
		const size_t bitWidthA = colA + minBitWidthA;
		const ssize_t minA = (0x1ll << (bitWidthA - 1ll)) + 1ll;
		const ssize_t maxA = (0x1ll << bitWidthA) - 1ll;
		const size_t numA = (maxA - minA) / 2ull + 1ull;
		const size_t numCombinations = numA * (0x1ll << bitWidthData);

//		if ((bitWidthData == 28) && (bitWidthA == 17)) {
//			std::cout << "# Skipping |D|=" << bitWidthData << " and |A|=" << bitWidthA << std::endl;
//			continue;
//		}

		if (numCombinations >= (0x1ull << 16ull)) { // from n=16 on, we use nested parallelization (the value is arbitrarily chosen)
#pragma omp parallel for schedule(dynamic) shared(data)
			for (ssize_t currentA = minA; currentA <= maxA; currentA += 2ull) {
				worker(data, bitWidthData, bitWidthA, currentA);
			} // nested parallel for (single nesting level)
		} else { // not-paralleled inner loop for too small units
			for (ssize_t currentA = minA; currentA <= maxA; currentA += 2ull) {
				worker(data, bitWidthData, bitWidthA, currentA);
			}
		}

		std::stringstream ss;
		ss << (bitWidthData + bitWidthA) << '\t' << bitWidthData << '\t' << bitWidthA << '\t' << data.lengthSignedDigitsRepresentation << '\t' << data.elements.size() << '\t';
		bool first = true;
		for (auto A : data.elements) {
			if (!first) {
				ss << ',';
			} else {
				first = false;
			}
			ss << A;
		}
		ss << '\n';
		std::cout << ss.str();
	}
} // main
