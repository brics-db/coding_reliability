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
ext_euklidean(T b0, size_t codewidth)
{
	T a0(1);
	a0 <<= codewidth;
	vector<T> a, b, q, r, s, t;
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

const ssize_t AMIN = 3;
const size_t AwidthMAX = 16;
const ssize_t DwidthMIN = 1;
const ssize_t DwidthMAX = 24;

int
main(int argc, char** argv)
{
	std::vector<bool> checkedAs(1ull << AwidthMAX);
	if (argc > 1)
	{
		if (strncmp("-", argv[1], 1) == 0)
		{
			std::istreambuf_iterator<char> begin(std::cin), end;
			std::string str(begin, end);
			std::istringstream iss(str);
			std::vector<string> tokens{std::istream_iterator<string>{iss}, std::istream_iterator<string>{}};
			size_t i = 0;
			for (auto & str : tokens)
			{
				size_t A = strtoull(str.c_str(), nullptr, 10);
				if (A > 0 && A < std::numeric_limits<size_t>::max())
				{
					++i;
					checkedAs[A] = true;
				}
			}
			std::cerr << "Got " << i << " As to skip" << std::endl;
		}
		else
		{
			int i = 1;
			for (; i < argc; ++i)
			{
				size_t A = strtoull(argv[i], nullptr, 10);
				if (A > 0 && A < std::numeric_limits<size_t>::max())
				{
					checkedAs[A] = true;
				}
			}
			std::cerr << "Got " << (i-1) << " As to skip" << std::endl;
		}
	}
#pragma omp parallel for schedule(dynamic)
	for (ssize_t A = AMIN; A < (1ll << AwidthMAX); A += 2)
	{
		if (checkedAs[A])
		{
			std::stringstream ss;
			ss << "skipping " << A << '\n';
			std::cout << ss.str() << std::flush;
			continue;
		}
		else
		{
			std::stringstream ss;
			ss << "computing " << A << '\n';
			std::cout << ss.str() << std::flush;
		}
		std::stringstream ss;
		ss << "an_decoding_is_error_detection_" << omp_get_thread_num() << ".out";
		std::ofstream fout(ss.str(), ios::out | ios::app | ios::ate);
		StopWatch sw;
		for (size_t Dwidth = DwidthMIN; Dwidth <= DwidthMAX; ++Dwidth)
		{
			sw.start();
			const ssize_t Dmax = (1ll << (Dwidth - 1)) - 1;
			const ssize_t Dmin = (-1) * (1ll << (Dwidth - 1));
			const ssize_t Awidth = (sizeof (size_t) * 8 - __builtin_clzll(A));
			const ssize_t Cwidth = Awidth + Dwidth;
			const ssize_t Cmin = Dmin * A;
			const ssize_t Cmax = Dmax * A;
			const ssize_t AInv = ext_euklidean(A, Cwidth) & ((1ll << Cwidth) - 1);
			if (AInv == 0)
			{
				std::stringstream ss;
				ss << "AINV No multiplicative inverse for A=" << A << " and |C|=" << Cwidth << '\n';
				if (fout)
				{
					fout << ss.str();
				}
				else
				{
					std::cerr << ss.str();
				}
			}
			else
			{
				ssize_t c = ~((1ll << (Cwidth - 1)) - 1); // smallest signed integer representable with Cwidth bits
				ssize_t count = (c % A);
				if (count < 0)
				{
					count = -count;
				}
				if (Cwidth <= 16) {
					int mmIsNotACodeword = 0;
					for (ssize_t i = 0, tmp = c; i < 16; ++i, ++tmp) {
						if ((tmp % A) == 1) {
							mmIsNotACodeword |= (1ll << (i * 2)); // expand to 32 bits, because AVX2 cannot movemask across lanes to 16 bits
						}
					}
					auto mmCodewords = _mm256_set_epi16(c+15, c+14, c+13, c+12, c+11, c+10, c+9, c+8, c+7, c+6, c+5, c+4, c+3, c+2, c+1, c);
					auto mmAddUp = _mm256_set1_epi16(16);
					auto mmAinv = _mm256_set1_epi16(AInv);
					auto mmDmin = _mm256_set1_epi16(Dmin);
					auto mmDmax = _mm256_set1_epi16(Dmax);
					const size_t posMax = (1ull << CWidth);
					std::cout << setfill('0') << std::hex;
					for(size_t pos = 16; pos < posMax; pos += 16) {
						auto mm1 = _mm256_mullo_epi16(mmCodewords, mmAinv);
						auto mm2 = _mm256_cmpgt_epi16(mm1, mmDmin);
						auto mm3 = _mm256_cmpgt_epi16(mmDmax, mm1);
						auto mm4 = _mm256_cmpeq_epi16(mmDmax, mm1);
						auto mm5 = _mm256_or_si256(mm3, mm4);
						auto mm6 = _mm256_and_si256(mm2, mm5);
						auto mask = _mm256_movemask_epi8(mm6);
						if (mask & mmIsNotACodeword) {
							std::cout << "BAD @0x" << setw(16) << pos << ": 0x" << mask << " & 0x" << mmIsNotACodeword << " = 0x" << (mask & mmIsNotACodeword) << std::endl;
						}
						mmCodewords = _mm256_add_epi16(mmCodewords, mmAddUp);
					}
					std::cout << setfill(' ') << std::dec;
				} else if (Cwidth <= 32) {
				} else if (CWidth <= 64) {
				}
			}
			sw.stop();
			{
				std::stringstream ss;
				ss << A << '|' << std::hex << "0x" << AInv << std::dec << '|' << Dwidth << '|' << sw.duration() << '\n';
				if (fout)
				{
					fout << ss.str();
				}
				else
				{
					std::cout << ss.str();
				}
			}
		}
		{
			std::stringstream ss;
			ss << "done " << A << "\n";
			std::cout << ss.str() << std::flush;
		}
		if (fout)
		{
			fout.close();
		}
	}
	return 0;
}

