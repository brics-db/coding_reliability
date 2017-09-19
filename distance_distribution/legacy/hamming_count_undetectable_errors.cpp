// Copyright 2015,2016 Till Kolditz
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

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <array>
#include <utility>
#include <vector>
#include <type_traits>
#include <omp.h>
#include <immintrin.h>
#include <boost/multiprecision/cpp_int.hpp>

#include "Hamming/Scalar.hpp"
#include "Hamming/SSE.hpp"
#include "Hamming/AVX2.hpp"

/**
 * Taken and adapted from: https://gist.github.com/jeetsukumaran/5392166
 *
 * Calculates the binomial coefficient, $\choose{n, k}$, i.e., the number of
 * distinct sets of $k$ elements that can be sampled with replacement from a
 * population of $n$ elements.
 *
 * @tparam T
 *   Numeric type. Defaults to unsigned long.
 * @param n
 *   Population size.
 * @param k
 *   Number of elements to sample without replacement.
 *
 * @return
 *   The binomial coefficient, $\choose{n, k}$.
 *
 * @note
 *    Modified from: http://etceterology.com/fast-binomial-coefficients
 */
template<class T = unsigned long>
T binomial_coefficient(
        size_t n,
        size_t k) {
    unsigned long i;
    if (0 == k || n == k) {
        return 1;
    }
    if (k > n) {
        return 0;
    }
    if (k > (n - k)) {
        k = n - k;
    }
    if (1 == k) {
        return n;
    }
    T b = 1;
    for (i = 1; i <= k; ++i) {
        b *= (n - (k - i));
        if (b < 0)
            return -1; /* Overflow */
        b /= i;
    }
    return b;
}

class StopWatch {
    std::chrono::high_resolution_clock::time_point startNS, stopNS;

public:
    StopWatch()
            : startNS(std::chrono::nanoseconds(0)),
              stopNS(std::chrono::nanoseconds(0)) {
    }

    void start() {
        startNS = std::chrono::high_resolution_clock::now();
    }

    std::chrono::high_resolution_clock::rep stop() {
        stopNS = std::chrono::high_resolution_clock::now();
        return duration();
    }

    std::chrono::high_resolution_clock::rep duration() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(stopNS - startNS).count();
    }
};
std::ostream& operator<<(
        std::ostream& stream,
        StopWatch& sw) {
    std::chrono::high_resolution_clock::rep dura = sw.duration();
    size_t max = 1000;
    while (dura / max > 0) {
        max *= 1000;
    }
    max /= 1000;
    stream << std::setfill('0') << (dura / max);
    while (max > 1) {
        dura %= max;
        max /= 1000;
        stream << '.' << std::setw(3) << (dura / max);
    }
    stream << std::setfill(' ');
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

struct statistics {
    typedef boost::multiprecision::uint128_t counts_t;

    const size_t numDataBits;
    const size_t numHammingBits;
    const size_t numCodeBits;
    const counts_t numCodeWords;
    const size_t numCounts;
    std::unique_ptr<counts_t[]> counts;

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = 0;
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            boost::multiprecision::uint128_t * counts,
            boost::multiprecision::uint128_t * ext_counts,
            boost::multiprecision::uint128_t * cor_counts,
            boost::multiprecision::uint128_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint64_t * counts,
            uint64_t * ext_counts,
            uint64_t * cor_counts,
            uint64_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint32_t * counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint16_t * counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint8_t * counts,
            uint8_t * ext_counts,
            uint8_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
        }
    }

    statistics(
            statistics && other)
            : numDataBits(other.numDataBits),
              numHammingBits(other.numHammingBits),
              numCodeBits(other.numCodeBits),
              numCodeWords(other.numCodeWords),
              numCounts(other.numCounts),
              counts(std::move(other.counts)) {
    }

    statistics(
            statistics & other) = delete;

    statistics & operator=(
            statistics && other) {
        new (this) statistics(std::forward<statistics>(other));
        return *this;
    }

    statistics & operator=(
            statistics & other) = delete;
};

template<size_t ACTUAL_BITCNT_DATA>
statistics countHammingUndetectableErrors(
        statistics & former_stats) {
    typedef typename ExtHamming<ACTUAL_BITCNT_DATA>::hamming_impl_t T;
    typedef typename T::accumulator_t accumulator_t;
    typedef typename T::data_t data_t;

    const constexpr size_t BITCNT_HAMMINGBITS = ExtHamming < ACTUAL_BITCNT_DATA > ::BITCNT_HAMMING;
    const constexpr size_t BITCNT_CODEWORD = ACTUAL_BITCNT_DATA + BITCNT_HAMMINGBITS;
    const constexpr accumulator_t CNT_CODEWORDS = 1ull << ACTUAL_BITCNT_DATA;
    const constexpr size_t CNT_COUNTS = BITCNT_CODEWORD + 1ull;

#if defined (_MSC_VER) and not defined (size_t)
    typedef __int64 size_t;
#endif

    StopWatch sw;
    sw.start();

    statistics stats(ACTUAL_BITCNT_DATA, BITCNT_HAMMINGBITS, BITCNT_CODEWORD, CNT_CODEWORDS, CNT_COUNTS);
    // we reuse the former stats to half the number of code words we actually need to count
#ifndef NDEBUG
    std::cout << "# former stats given (" << former_stats.numCounts << " counts, " << former_stats.numCodeWords << " codewords):\n";
#endif
    for (size_t i = 0; i < former_stats.numCounts; ++i) {
        stats.counts[i] = former_stats.counts[i];
    }
    std::cout << std::flush;

#ifndef NDEBUG
    std::cout << "BITCNT_HAMMINGBITS=" << BITCNT_HAMMINGBITS << "\n";
    std::cout << "BITCNT_CODEWORD=" << BITCNT_CODEWORD << "\n";
    std::cout << "CNT_CODEWORDS=" << CNT_CODEWORDS << "\n";
    std::cout << "CNT_COUNTS=" << CNT_COUNTS << "\n";
#pragma omp parallel
    {
#pragma omp master
        {
            std::cout << "OpenMP using " << omp_get_num_threads() << " threads." << std::endl;
        }
    }
    size_t bucket_size_tmp;
    size_t remaining_tmp;
#pragma omp parallel shared(stats,former_stats,bucket_size_tmp,remaining_tmp)
#else
#pragma omp parallel shared(stats,former_stats)
#endif
    {
        const size_t num_threads = omp_get_num_threads();
        const size_t numCodeWordsToEnumerate = CNT_CODEWORDS - former_stats.numCodeWords;
        const size_t bucket_size = numCodeWordsToEnumerate / num_threads;
        const size_t remaining = numCodeWordsToEnumerate % num_threads;

#ifndef NDEBUG
#pragma omp master
        {
            bucket_size_tmp = bucket_size;
            remaining_tmp = remaining;
        }
#endif

#pragma omp for schedule(dynamic,1)
        for (size_t thread = 0; thread < num_threads; ++thread) {
            std::array<accumulator_t, CNT_COUNTS> counts_local {0};
            size_t x = former_stats.numCodeWords + bucket_size * thread;
            const size_t max = x + bucket_size + (thread == (num_threads - 1) ? remaining : 0); // the last thread does the few additional remaining code words

#ifdef __AVX2__
            const constexpr size_t values_per_mm256 = sizeof(__m256i) / sizeof(data_t);
            auto mm256 = SIMD<__m256i, data_t>::set_inc(x, 1);
            auto mm256inc = SIMD<__m256i, data_t>::set1(values_per_mm256);

            if (max >= values_per_mm256) {
                for (; x <= (max - values_per_mm256); x += values_per_mm256) {
                    auto popcount = SIMD<__m256i, data_t>::popcount(mm256);
                    auto hamming = SIMD<__m256i, data_t>::hamming(mm256);
                    auto hammingPopcount = SIMD<__m256i, data_t>::popcount(hamming);
                    popcount = SIMD<__m256i, data_t>::add(popcount, hammingPopcount);
                    auto * pPopcount = reinterpret_cast<data_t*>(&popcount);
                    for (size_t i = 0; i < values_per_mm256; ++i) {
                        counts_local[pPopcount[i]]++;
                    }
                    mm256 = SIMD<__m256i, data_t>::add(mm256, mm256inc);
                }
            }
#endif /* __AVX2__ */

#ifdef __SSE4_2__
            const constexpr size_t values_per_mm128 = sizeof(__m128i) / sizeof(data_t);
            auto mm128 = SIMD<__m128i, data_t>::set_inc(x, 1);
            auto mm128inc = SIMD<__m128i, data_t>::set1(values_per_mm128);

            if (max >= values_per_mm128) {
                for (; x <= (max - values_per_mm128); x += values_per_mm128) {
                    auto popcount = SIMD<__m128i, data_t>::popcount(mm128);
                    auto hamming = SIMD<__m128i, data_t>::hamming(mm128);
                    auto hammingPopcount = SIMD<__m128i, data_t>::popcount(hamming);
                    popcount = SIMD<__m128i, data_t>::add(popcount, hammingPopcount);
                    auto * pPopcount = reinterpret_cast<data_t*>(&popcount);
                    for (size_t i = 0; i < values_per_mm128; ++i) {
                        counts_local[pPopcount[i]]++;
                    }
                    mm128 = SIMD<__m128i, data_t>::add(mm128, mm128inc);
                }
            }
#endif /* __SSE4_2__ */

            for (; x < max; ++x) {
                counts_local[Scalar<data_t>::popcount(x) + Scalar<data_t>::popcount(T::compute(x))]++;
            }

            // 4) Sum the counts
#pragma omp critical
            for (size_t i = 0; i < CNT_COUNTS; ++i) {
                stats.counts[i] += counts_local[i];
            }
        }
    }
    sw.stop();

    typedef typename statistics::counts_t counts_t;
    std::vector<counts_t> ext_counts(stats.numCounts);
    std::vector<counts_t> cor_counts(stats.numCounts);
    std::vector<counts_t> act_counts(stats.numCounts);

    // extend the basic counts to extended Hamming
    counts_t maxCountsRaw = 0;
    counts_t maxCountsExt = 0;
    ext_counts[0] = stats.counts[0];
    for (size_t i = 1; i < CNT_COUNTS; ++i) {
        if (i & 0x1) {
            ext_counts[i] = 0;
        } else {
            ext_counts[i] = stats.counts[i] + stats.counts[i - 1];
        }
        if (stats.counts[i] > maxCountsRaw) {
            maxCountsRaw = static_cast<counts_t>(stats.counts[i]);
        }
        if (ext_counts[i] > maxCountsExt) {
            maxCountsExt = static_cast<counts_t>(ext_counts[i]);
        }
    }
    size_t maxWidthRaw = 0;
    do {
        maxCountsRaw /= 10;
        ++maxWidthRaw;
    } while (maxCountsRaw);
    size_t maxWidthExt = 0;
    do {
        maxCountsExt /= 10;
        ++maxWidthExt;
    } while (maxCountsExt);

    // the 1-bit sphere transitions
    counts_t maxCountsCor = 0;
    cor_counts[0] = ext_counts[0];
    for (size_t i = 1; i < CNT_COUNTS; ++i) {
        if (i & 0x1) {
            cor_counts[i] = (ext_counts[i - 1] * (BITCNT_CODEWORD - (i - 1))) + ((i + 1) < CNT_COUNTS ? (ext_counts[i + 1] * (i + 1)) : 0);
        } else {
            cor_counts[i] = ext_counts[i];
        }
        if (cor_counts[i] > maxCountsCor) {
            maxCountsCor = static_cast<counts_t>(cor_counts[i]);
        }
    }
    size_t maxWidthCor = 0;
    do {
        maxCountsCor /= 10;
        ++maxWidthCor;
    } while (maxCountsCor);

    // the transitions apply to all code words
    counts_t maxCountAct = 0;
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        act_counts[i] = cor_counts[i] * CNT_CODEWORDS;
        if (act_counts[i] > maxCountAct) {
            maxCountAct = static_cast<counts_t>(act_counts[i]);
        }
    }
    size_t maxWidthAct = 0;
    do {
        maxCountAct /= 10;
        ++maxWidthAct;
    } while (maxCountAct);

    counts_t maxTransitions2 = CNT_CODEWORDS * binomial_coefficient<counts_t>(BITCNT_CODEWORD, BITCNT_CODEWORD / 2);
    size_t maxWidthTra = 0;
    do {
        maxTransitions2 /= 10;
        ++maxWidthTra;
    } while (maxTransitions2 > 0);

    counts_t numTotal = 0;
    counts_t numTransitions = 0;
#ifndef NDEBUG
    std::cout << '(' << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : bucketsize(" << bucket_size_tmp << ") remaining(" << remaining_tmp << ")" << std::endl;
#endif
    const constexpr bool isCodeShortened = (BITCNT_CODEWORD & (~BITCNT_CODEWORD + 1)) == BITCNT_CODEWORD;
    std::cout << "# SDC probabilities for a (n,k) = (" << BITCNT_CODEWORD << ',' << ACTUAL_BITCNT_DATA << ") " << (isCodeShortened ? "shortened " : "") << "correcting Extended Hamming code\n";
    std::cout << "# The reported colunms are:\n";
    std::cout << "#   1) bit width\n";
    std::cout << "#   2) weight distribution of " << (isCodeShortened ? "shortened " : "") << "detecting-only Hamming\n";
    std::cout << "#   3) weight distribution of " << (isCodeShortened ? "shortened " : "") << "detecting-only Extended Hamming\n";
    std::cout << "#   4) weight distribution of " << (isCodeShortened ? "shortened " : "") << "correcting Extended Hamming\n";
    std::cout << "#   5) error pattern weight distribution of " << (isCodeShortened ? "shortened " : "") << "correcting Extended Hamming\n";
    std::cout << "#   6) maximum # transitions of " << (isCodeShortened ? "shortened " : "") << "extended Hamming (2^k * (n over col 1))\n";
    std::cout << "#   7) probability (col 5 / col 6)\n";
    std::cout << std::dec << "#Distances:\n" << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        counts_t maxTransitions = CNT_CODEWORDS * binomial_coefficient<counts_t>(BITCNT_CODEWORD, i);
        numTotal += stats.counts[i];
        numTransitions += act_counts[i];
        double probability_act = double(act_counts[i]) / double(maxTransitions);

        std::cout << std::right << std::setw(4) << i;
        std::cout << ',' << std::setw(maxWidthRaw) << stats.counts[i];
        std::cout << ',' << std::setw(maxWidthExt) << ext_counts[i];
        std::cout << ',' << std::setw(maxWidthCor) << cor_counts[i];
        std::cout << ',' << std::setw(maxWidthAct) << act_counts[i];
        std::cout << ',' << std::setw(maxWidthTra) << maxTransitions;
        std::cout << ',' << probability_act << '\n';
    }
    if (numTotal != CNT_CODEWORDS) {
        std::cerr << '(' << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : numTotal (" << numTotal << " != numCodeWords (" << CNT_CODEWORDS << ')' << std::endl;
    }

    std::cout << "# (" << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : Computation took " << sw << "ns for " << CNT_CODEWORDS << " code words and " << numTransitions << " transitions.\n\n"
            << std::endl;

    return stats;
}

int main() {
    std::vector<statistics> all_stats;
    all_stats.reserve(64);
    all_stats.push_back(statistics(0, 0, 0, 0, 1));
    all_stats.emplace_back(countHammingUndetectableErrors<1>(all_stats[0])); // all stats are zero-initialized
    all_stats.emplace_back(countHammingUndetectableErrors<2>(all_stats[1])); // now stats[0] is all set-up
    all_stats.emplace_back(countHammingUndetectableErrors<3>(all_stats[2]));
    all_stats.emplace_back(countHammingUndetectableErrors<4>(all_stats[3]));
    all_stats.emplace_back(countHammingUndetectableErrors<5>(all_stats[4]));
    all_stats.emplace_back(countHammingUndetectableErrors<6>(all_stats[5]));
    all_stats.emplace_back(countHammingUndetectableErrors<7>(all_stats[6]));
    all_stats.emplace_back(countHammingUndetectableErrors<8>(all_stats[7]));
    all_stats.emplace_back(countHammingUndetectableErrors<9>(all_stats[8]));
    all_stats.emplace_back(countHammingUndetectableErrors<10>(all_stats[9]));
    all_stats.emplace_back(countHammingUndetectableErrors<11>(all_stats[10]));
    all_stats.emplace_back(countHammingUndetectableErrors<12>(all_stats[11]));
    all_stats.emplace_back(countHammingUndetectableErrors<13>(all_stats[12]));
    all_stats.emplace_back(countHammingUndetectableErrors<14>(all_stats[13]));
    all_stats.emplace_back(countHammingUndetectableErrors<15>(all_stats[14]));
    all_stats.emplace_back(countHammingUndetectableErrors<16>(all_stats[15]));
    all_stats.emplace_back(countHammingUndetectableErrors<17>(all_stats[16]));
    all_stats.emplace_back(countHammingUndetectableErrors<18>(all_stats[17]));
    all_stats.emplace_back(countHammingUndetectableErrors<19>(all_stats[18]));
    all_stats.emplace_back(countHammingUndetectableErrors<20>(all_stats[19]));
    all_stats.emplace_back(countHammingUndetectableErrors<21>(all_stats[20]));
    all_stats.emplace_back(countHammingUndetectableErrors<22>(all_stats[21]));
    all_stats.emplace_back(countHammingUndetectableErrors<23>(all_stats[22]));
    all_stats.emplace_back(countHammingUndetectableErrors<24>(all_stats[23]));
    all_stats.emplace_back(countHammingUndetectableErrors<25>(all_stats[24]));
    all_stats.emplace_back(countHammingUndetectableErrors<26>(all_stats[25]));
    all_stats.emplace_back(countHammingUndetectableErrors<27>(all_stats[26]));
    all_stats.emplace_back(countHammingUndetectableErrors<28>(all_stats[27]));
    all_stats.emplace_back(countHammingUndetectableErrors<29>(all_stats[28]));
    all_stats.emplace_back(countHammingUndetectableErrors<30>(all_stats[29]));
    all_stats.emplace_back(countHammingUndetectableErrors<31>(all_stats[30]));
    all_stats.emplace_back(countHammingUndetectableErrors<32>(all_stats[31]));
    all_stats.emplace_back(countHammingUndetectableErrors<33>(all_stats[32]));
    all_stats.emplace_back(countHammingUndetectableErrors<34>(all_stats[33]));
    all_stats.emplace_back(countHammingUndetectableErrors<35>(all_stats[34]));
    all_stats.emplace_back(countHammingUndetectableErrors<36>(all_stats[35]));
    all_stats.emplace_back(countHammingUndetectableErrors<37>(all_stats[36]));
    all_stats.emplace_back(countHammingUndetectableErrors<38>(all_stats[37]));
    all_stats.emplace_back(countHammingUndetectableErrors<39>(all_stats[38]));
    all_stats.emplace_back(countHammingUndetectableErrors<40>(all_stats[39]));

    return 0;
}
