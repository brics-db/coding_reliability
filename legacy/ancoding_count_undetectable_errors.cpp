/*
 * Copyright (C) 2015, 2016 Till Kolditz
 *
 * File: ancoding_count_undetectable_errors.cpp
 * Authors:
 *     Till Kolditz - till.kolditz@gmail.com
 *
 * This file is distributed under the Apache License Version 2.0; you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *  http://www.apache.org/licenses/LICENSE-2.0
 */
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
 * Count undetectable bit flips for AN encoded data words.
 *
 * AN coding cannot detect bit flips which result in multiples of A.
 * 
 * We essentially only count all transitions from codewords to other possible codewords, i.e. multiples of A.
 * All actual bit flips (i.e. distance > 0) are undetectable bit flips.
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

template<typename T>
inline size_t computeDistance(const T &value1, const T &value2) {
	return static_cast<size_t>(__builtin_popcountll(value1 ^ value2));
}

template<size_t A, size_t BITCNT_A, size_t BITCNT_DATA, typename T = size_t, size_t SZ_SHARDS = 64, size_t CNT_MESSAGES = 0x1ull << BITCNT_DATA,
		size_t CNT_COUNTS = BITCNT_A + BITCNT_DATA + 1, size_t CNT_SLICES = CNT_MESSAGES / SZ_SHARDS, size_t CNT_SHARDS = CNT_SLICES * CNT_SLICES>
void countANCodingUndetectableErrors() {
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

			// 1) Triangle for main diagonale
			T m1, m2;
			for (size_t x = 0; x < SZ_SHARDS; ++x) {
				m1 = (shardX + x) * A;
				++counts_local[computeDistance(m1, m1)];
				for (size_t y = (x + 1); y < SZ_SHARDS; ++y) {
					m2 = (shardX + y) * A;
					counts_local[computeDistance(m1, m2)] += 2;
				}
			}

			// 2) Remainder of the slice
			for (size_t shardY = shardX + SZ_SHARDS; shardY < CNT_MESSAGES; shardY += SZ_SHARDS) {
				for (size_t x = 0; x < SZ_SHARDS; ++x) {
					m1 = (shardX + x) * A;
					for (size_t y = 0; y < SZ_SHARDS; ++y) {
						m2 = (shardY + y) * A;
						counts_local[computeDistance(m1, m2)] += 2;
					}
				}
			}

			// 3) Sum the counts
			for (size_t i = 0; i < CNT_COUNTS; ++i) {
#pragma omp atomic
				counts[i] += counts_local[i];
			}

			size_t shardsComputed = CNT_SLICES - (static_cast<float>(shardX) / SZ_SHARDS);
			float inc = static_cast<float>(shardsComputed * 2 - 1) / CNT_SHARDS * 100;
#pragma omp atomic
			shardsDone += inc;

			if (omp_get_thread_num() == 0) {
				// cout << setw(10) << setfill('\b') << '\b' << setfill(' ') << setprecision(5) << (static_cast<double>(cntSlicesDone) / CNT_SLICES * 100.0L) << '%' << flush;
				cout << "\b\b\b\b\b\b\b\b\b\b" << right << setw(9) << setprecision(5) << shardsDone << '%' << flush;
			}
		}
	}
	sw.stop();

	cout << "\b\b\b\b\b\b\b\b\b\b" << right << setw(9) << setprecision(5) << shardsDone << '%' << endl;

	cout << dec << A << " - " << BITCNT_DATA << " - #Distances:\n";
	for (size_t i = 0; i < CNT_COUNTS; ++i) {
		cout << "  " << right << setw(2) << i << ": " << setw(13) << counts[i] << '\n';
	}

	cout << "Computation took " << sw << "ns." << endl;
}

int main() {
	countANCodingUndetectableErrors<641, 10, 8>();
	countANCodingUndetectableErrors<641, 10, 16>();
	countANCodingUndetectableErrors<641, 10, 24>();
	// countANCodingUndetectableErrors<641, 10, 32>();

	countANCodingUndetectableErrors<965, 10, 8>();
	countANCodingUndetectableErrors<965, 10, 16>();
	countANCodingUndetectableErrors<965, 10, 24>();
	// countANCodingUndetectableErrors<965, 10, 32>();

	countANCodingUndetectableErrors<7567, 13, 8>();
	countANCodingUndetectableErrors<7567, 13, 16>();
	countANCodingUndetectableErrors<7567, 13, 24>();
	// countANCodingUndetectableErrors<7567, 13, 32>();

	countANCodingUndetectableErrors<58659, 16, 8>();
	countANCodingUndetectableErrors<58659, 16, 16>();
	countANCodingUndetectableErrors<58659, 16, 24>();
	// countANCodingUndetectableErrors<58659, 16, 32>();
	return 0;
}
