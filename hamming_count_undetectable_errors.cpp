#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <omp.h>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt64
#endif

using namespace std;
using namespace std::chrono;

class StopWatch {
	high_resolution_clock::time_point startNS, stopNS;

public:
	StopWatch() {
	}

	void start() {
		startNS = high_resolution_clock::now();
	}

	high_resolution_clock::rep stop() {
		stopNS = high_resolution_clock::now();
		return duration();
	}

	high_resolution_clock::rep duration() {
		return duration_cast<nanoseconds>(stopNS - startNS).count();
	}
};
ostream& operator<<(ostream& stream, StopWatch& sw) {
	high_resolution_clock::rep dura = sw.duration();
	size_t max = 1000;
	while (dura / max > 0) {
		max *= 1000;
	}
	max /= 1000;
	stream << setfill('0') << (dura / max);
	while (max > 1) {
		dura %= max;
		max /= 1000;
		stream << '.' << setw(3) << (dura / max);
	}
	stream << setfill(' ');
	return stream;
}

/*
 * Count undetectable bit flips for Hamming encoded data words.
 *
 * Extended Hamming cannot detect bit flips which lead
 *  * either exactly to another codeword,
 *  * or to any other word in the 1-distance sphere around another codeword.
 * 
 * We essentially only count all transitions between codewords and other codewords,
 * as well as codewords and non-codewords which are 1 bit away from the codewords (1-distance sphere),
 * because we know that these transistions cannot be detected.
 * 
 * Therefore, we model the space of possible messages as a graph. The edges are the transitions / bit flips between words,
 * while the weights are the number of bits required to flip to get to the other message.
 * Since we start from codewords only, the graph is directed, from all codewords to all other codewords an all words in the
 * corresponding 1-distance sphears.
 * 
 * The 1-distance sphere is only virtual in this implementation, realized by an inner for-loop over all 22 possible 1-bitflips
 *
 * The weights (W) are counted, grouped by the weight. This gives the corresponding number of undetectable W-bitflips.
 */

inline size_t computeHamming08(const size_t &value) {
	size_t hamming = 0;
	hamming |= (__builtin_popcount(value & 0x0000005B) & 0x1) << 1;
	hamming |= (__builtin_popcount(value & 0x0000006D) & 0x1) << 2;
	hamming |= (__builtin_popcount(value & 0x0000008E) & 0x1) << 3;
	hamming |= (__builtin_popcount(value & 0x000000F0) & 0x1) << 4;
	hamming |= (__builtin_popcount(value & 0x000000FF) + __builtin_popcount(hamming)) & 0x1;
	return (value << 5) | hamming;
}

inline size_t computeHamming16(const size_t &value) {
	size_t hamming = 0;
	hamming |= (__builtin_popcount(value & 0x0000AD5B) & 0x1) << 1;
	hamming |= (__builtin_popcount(value & 0x0000366D) & 0x1) << 2;
	hamming |= (__builtin_popcount(value & 0x0000C78E) & 0x1) << 3;
	hamming |= (__builtin_popcount(value & 0x000007F0) & 0x1) << 4;
	hamming |= (__builtin_popcount(value & 0x0000F800) & 0x1) << 5;
	hamming |= (__builtin_popcount(value & 0x0000FFFF) + __builtin_popcount(hamming)) & 0x1;
	return (value << 6) | hamming;
}

inline size_t computeHamming24(const size_t &value) {
	size_t hamming = 0;
	hamming |= (__builtin_popcount(value & 0x00AAAD5B) & 0x1) << 1;
	hamming |= (__builtin_popcount(value & 0x0033366D) & 0x1) << 2;
	hamming |= (__builtin_popcount(value & 0x00C3C78E) & 0x1) << 3;
	hamming |= (__builtin_popcount(value & 0x00FC07F0) & 0x1) << 4;
	hamming |= (__builtin_popcount(value & 0x00FFF800) & 0x1) << 5;
	hamming |= (__builtin_popcount(value & 0x00FFFFFF) + __builtin_popcount(hamming)) & 0x1;
	return (value << 6) | hamming;
}

inline size_t computeHamming32(const size_t &value) {
	size_t hamming = 0;
	hamming |= (__builtin_popcount(value & 0x56AAAD5B) & 0x1) << 1;
	hamming |= (__builtin_popcount(value & 0x9B33366D) & 0x1) << 2;
	hamming |= (__builtin_popcount(value & 0xE3C3C78E) & 0x1) << 3;
	hamming |= (__builtin_popcount(value & 0x03FC07F0) & 0x1) << 4;
	hamming |= (__builtin_popcount(value & 0x03FFF800) & 0x1) << 5;
	hamming |= (__builtin_popcount(value & 0xFC000000) & 0x1) << 6;
	hamming |= (__builtin_popcount(value & 0xFFFFFFFF) + __builtin_popcount(hamming)) & 0x1;
	return (value << 7) | hamming;
}

template<typename T>
inline size_t computeDistance(const T &value1, const T &value2) {
	return static_cast<size_t>(__builtin_popcount(value1 ^ value2));
}

typedef size_t (*computeHamming_ft)(const size_t &);

template<typename T, size_t BITCNT_DATA, size_t SZ_SHARDS = 64ull, computeHamming_ft func, size_t BITCNT_HAMMING = (
		BITCNT_DATA == 8 ? 5 : ((BITCNT_DATA == 16) | (BITCNT_DATA == 24) ? 6 : 7)), size_t BITCNT_MSG = BITCNT_DATA + BITCNT_HAMMING, size_t CNT_COUNTS =
		BITCNT_MSG + 1ull, size_t CNT_EDGES_SHIFT = BITCNT_DATA + BITCNT_MSG, size_t CNT_EDGES = 0x1ull << CNT_EDGES_SHIFT, size_t CNT_MESSAGES = 0x1ull
		<< BITCNT_DATA, size_t MUL_1DISTANCE = BITCNT_MSG, size_t MUL_2DISTANCE = BITCNT_MSG * (BITCNT_MSG - 1ull) / 2ull, size_t CNT_SLICES = CNT_MESSAGES
		/ SZ_SHARDS, size_t CNT_SHARDS = CNT_SLICES * CNT_SLICES>
void countHammingUndetectableErrors() {
	StopWatch sw;
	sw.start();

	size_t counts[CNT_COUNTS] = { 0 };
	double shardsDone = 0.0;

#pragma omp parallel
	{
#pragma omp master
		{
			cout << "OpenMP using " << omp_get_num_threads() << endl;
		}
#pragma omp for schedule(dynamic,1)
#ifdef _MSC_VER
		for (__int64 shardX = 0; shardX < CNT_MESSAGES; shardX += SZ_SHARDS) {
#else
		for (size_t shardX = 0; shardX < CNT_MESSAGES; shardX += SZ_SHARDS) {
#endif
			size_t counts_local[CNT_COUNTS] = { 0 };
			T messages[SZ_SHARDS] = { 0 };
			T messages2[SZ_SHARDS] = { 0 };
			size_t distance;

			// 1) precompute Hamming values for this slice, i.e. the "originating" codewords
			for (size_t x = shardX, k = 0; k < SZ_SHARDS; ++x, ++k) {
				messages[k] = func(x);
			}

			// 2) Triangle for main diagonale
			for (size_t x = 0; x < SZ_SHARDS; ++x) {
				distance = computeDistance(messages[x], messages[x]);
				++counts_local[distance];
				for (size_t y = (x + 1); y < SZ_SHARDS; ++y) {
					distance = computeDistance(messages[x], messages[y]);
					counts_local[distance] += 2;
				}
			}

			// 3) Remainder of the slice
			for (size_t shardY = shardX + SZ_SHARDS; shardY < CNT_MESSAGES; shardY += SZ_SHARDS) {
				// 3.1) Precompute other code words
				for (size_t y = shardY, k = 0; k < SZ_SHARDS; ++y, ++k) {
					messages2[k] = func(y);
				}

				// 3.2) Do the real work
				for (size_t x = 0; x < SZ_SHARDS; ++x) {
					for (size_t y = 0; y < SZ_SHARDS; ++y) {
						distance = computeDistance(messages[x], messages2[y]);
						counts_local[distance] += 2;
					}
				}
			}

			// 4) Sum the counts
			for (size_t i = 0; i < CNT_COUNTS; ++i) {
#pragma omp atomic
				counts[i] += counts_local[i];
			}

			size_t shardsComputed = CNT_SLICES - (static_cast<float>(shardX) / SZ_SHARDS);
			float inc = static_cast<float>(shardsComputed * 2 - 1) / CNT_SHARDS * 100;

#pragma omp atomic
			shardsDone += inc;

			if (omp_get_thread_num() == 0) {
				cout << "\b\b\b\b\b\b\b\b\b\b" << right << setw(9) << setprecision(5) << shardsDone << '%' << flush;
			}
		}
	}
	sw.stop();

	cout << "\b\b\b\b\b\b\b\b\b\b" << right << setw(9) << setprecision(5) << shardsDone << '%' << flush;

	counts[1] = counts[0] * BITCNT_MSG;
	cout << dec << "\n#Distances:\n";
	for (size_t i = 3; i < CNT_COUNTS; i += 2) {
		counts[i] = (counts[i - 1] * (BITCNT_MSG - i + 1)) + ((i + 1) < CNT_COUNTS ? (counts[i + 1] * (i + 1)) : 0);
	}
	for (size_t i = 0; i < CNT_COUNTS; ++i) {
		cout << "  " << right << setw(2) << i << ": " << setw(13) << counts[i] << '\n';
	}

	cout << "Computation took " << sw << "ns." << endl;
}

int main() {
	countHammingUndetectableErrors<size_t, 8, 8, computeHamming08>();
	countHammingUndetectableErrors<size_t, 16, 512, computeHamming16>();
	countHammingUndetectableErrors<size_t, 24, 512, computeHamming24>();
	// countHammingUndetectableErrors<uint64_t, 32>();

	return 0;
}
