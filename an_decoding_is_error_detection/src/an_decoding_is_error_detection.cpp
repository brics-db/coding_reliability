/* Created: 2016/09/05
 * Author: Till Kolditz - till.kolditz@gmail.com
 * 
 * This files tests, whether the assumption holds that for all A (1 <= |A| <=16)
 * and data widths (1 <= |D| <= 16) a corrupt code word decodes into a d* > d_max
 */

#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <algorithm>
#include <iterator>

#include "stopwatch.hpp"

template<typename T>
T
ext_euklidean (T b0, size_t codewidth) {
    T a0(1);
    a0 <<= codewidth;
    vector<T> a, b, q, r, s, t;
    a.push_back(a0), b.push_back(b0), s.push_back(T(0)), t.push_back(T(0));
    size_t i = 0;
    do {
        q.push_back(a[i] / b[i]);
        r.push_back(a[i] % b[i]);
        a.push_back(b[i]);
        b.push_back(r[i]);
        s.push_back(0);
        t.push_back(0);
    } while (b[++i] > 0);
    s[i] = 1;
    t[i] = 0;

    for (size_t j = i; j > 0; --j) {
        s[j - 1] = t[j];
        t[j - 1] = s[j] - q[j - 1] * t[j];
    }

    T result = ((b0 * t.front()) % a0);
    result += result < 0 ? a0 : 0;
    if (result == 1) {
        return t.front();
    } else {
        return 0;
    }
}

inline void
checkCodeWord (ofstream & fout, const ssize_t & c, const ssize_t & A, const ssize_t & A_inv, const ssize_t & Dmin, const ssize_t & Dmax, const ssize_t Cmin, const ssize_t Cmax, const ssize_t Dwidth, const ssize_t Cwidth) {
    auto tmp = c * A_inv;
    if (Dmin <= tmp && tmp <= Dmax) { // if the decoding would yield a correct data word
        std::stringstream ss;
        ss << "BAD A=" << A << " A^-1=" << A_inv << " |D|=" << Dwidth << " |C|=" << Cwidth << " c=" << c << " (c_min, d_min, c*A^-1, d_max, c_max)=(0x" << hex << Cmin << ", 0x" << Dmin << ", 0x" << tmp << ", 0x" << Dmax << ", 0x" << Cmax << ')' << '\n';
        if (fout) {
            fout << ss.str();
        } else {
            cerr << ss.str();
        }
    }
}

const size_t AwidthMAX = 16;

int
main (int argc, char** argv) {
    std::vector<bool> checkedAs(1ull << AwidthMAX);
    if (argc > 1) {
        if (strncmp("-", argv[1], 1) == 0) {
            std::istreambuf_iterator<char> begin(std::cin), end;
            std::string str(begin, end);
            std::istringstream iss(str);
            std::vector<string> tokens{std::istream_iterator<string>{iss}, std::istream_iterator<string>{}};
            size_t i = 0;
            for (auto & str : tokens) {
                size_t A = strtoull(str.c_str(), nullptr, 10);
                if (A > 0 && A < std::numeric_limits<size_t>::max()) {
                    ++i;
                    checkedAs[A] = true;
                }
            }
            std::cerr << "Got " << i << " As to skip" << std::endl;
        } else {
            int i = 1;
            for (; i < argc; ++i) {
                size_t A = strtoull(argv[i], nullptr, 10);
                if (A > 0 && A < std::numeric_limits<size_t>::max()) {
                    checkedAs[A] = true;
                }
            }
            std::cerr << "Got " << (i - 1) << " As to skip" << std::endl;
        }
    }
#pragma omp parallel for schedule(dynamic)
    for (ssize_t A = 3; A < (1ll << AwidthMAX); A += 2) {
        // for (ssize_t A = 2059; A <= 2063; A += 2) {
        if (checkedAs[A]) {
            std::stringstream ss;
            ss << "skipping " << A << '\n';
            std::cout << ss.str() << std::flush;
            continue;
        } else {
            std::stringstream ss;
            ss << "computing " << A << '\n';
            std::cout << ss.str() << std::flush;
        }
        std::stringstream ss;
        ss << "an_decoding_is_error_detection_" << omp_get_thread_num() << ".out";
        std::ofstream fout(ss.str(), ios::out | ios::app | ios::ate);
        StopWatch sw;
        for (size_t Dwidth = 1; Dwidth <= 24; ++Dwidth) {
            // { size_t Dwidth = 24;
            sw.start();
            const ssize_t Dmin = (-1) * (1ll << (Dwidth - 1));
            const ssize_t Dmax = (1ll << (Dwidth - 1)) - 1;
            const uint32_t Awidth = (sizeof (size_t) * 8 - __builtin_clzll(A));
            const uint32_t Cwidth = Awidth + Dwidth;
            const ssize_t Cmin = Dmin * A;
            const ssize_t Cmax = Dmax * A;
            const ssize_t A_inv = ext_euklidean(A, Cwidth) & ((1ll << Cwidth) - 1);
            if (A_inv == 0) {
                std::stringstream ss;
                ss << "AINV No multiplicative inverse for A=" << A << " and |C|=" << Cwidth << '\n';
                if (fout) {
                    fout << ss.str();
                } else {
                    std::cerr << ss.str();
                }
            } else {
                ssize_t c = ~((1ll << (Cwidth - 1)) - 1); // smallest signed integer representable with Cwidth bits
                ssize_t count = (c % A);
                if (count < 0) {
                    count = -count;
                }
                // first, check all codewords up to the first one dividable by A (exclusive)
                for (; count > 0; --count, ++c) {
                    checkCodeWord(fout, c, A, A_inv, Dmin, Dmax, Cmin, Cmax, Dwidth, Cwidth);
                }
                ssize_t max = (1ll << (Cwidth - 1)) - 1;
                count = max % A;
                // now, check all codewords up to the largest one decodable by A (exclusive)
                for (++c; c < (max - count); ++c) { // jump over multiples of A
                    for (ssize_t i = 1; (i < A) && (c < Cmax); ++i, ++c) {
                        checkCodeWord(fout, c, A, A_inv, Dmin, Dmax, Cmin, Cmax, Dwidth, Cwidth);
                    }
                }
                max += 1;
                // now, check all remaining codewords
                for (++c; c < max; ++c) {
                    checkCodeWord(fout, c, A, A_inv, Dmin, Dmax, Cmin, Cmax, Dwidth, Cwidth);
                }
            }
            sw.stop();
            {
                std::stringstream ss;
                ss << A << '|' << std::hex << "0x" << A_inv << std::dec << '|' << Dwidth << '|' << sw.duration() << '\n';
                if (fout) {
                    fout << ss.str();
                } else {
                    std::cout << ss.str();
                }
            }
        }
        {
            std::stringstream ss;
            ss << "done " << A << "\n";
            std::cout << ss.str() << std::flush;
        }
        if (fout) {
            fout.close();
        }
    }
    return 0;
}

