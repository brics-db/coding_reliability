// Copyright 2016 Matthias Werner
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

#include "globals.h"

#include <helper.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


stringstream printbits(uintll_t v, uintll_t n)
{
  stringstream ss;
  uintll_t mask = (1ull<<n)-1;
  uintll_t p=n;
  v &= mask;
  for (; v&mask; v <<= 1, --p)
    ss << char('0' + ((v&mask)>>(n-1)));
  for (; p; --p)
    ss << '0';
  return ss;
}

void process_result(stringstream& ss,
                    uintll_t A, uint_t h,
                    const Flags& flags,
                    double* times, uint128_t* counts)
{
  static unsigned int id = 0;
  const char sep = ',';
  // sum
  long double total = 0;
  uint_t n = flags.n;

  for (uint_t i = 0; i < n+h+1; ++i) {
    total += static_cast<long double>(counts[i]);
  }

  long double prob;
  long double base1 = pow(2.0,n);
  long double base2;
  long double base;
  double mxabs=0,mxrel=0;
  int cnts = flags.n + h + 1;

  if(id==0) {
    ss << "id, k, A, h, A_2, m_1, m_2, hlen";
    for(int i=1; i<=cnts; ++i)
      ss << ", p_" << i;
    for(int i=1; i<=cnts; ++i)
      ss << ", hist_" << i;
    ss << ", t_kernel, t, sum, minb, mincb, superA, minb2, mincb2, superA2\n";
  }


  // k,A,h,A_bit,g,m,M,time,density[50],histogram[50],times,sum
  ss << id++
     << sep << flags.n
     << sep << A
     << sep << h
     << sep << printbits(A, n).str()
     << sep << flags.mc_iterations
     << sep << flags.mc_iterations_2
     << sep << cnts
    ;
  // density + histogram

  for(uint_t i=1; i<=cnts; ++i)
  {
    base2 = binomialCoeff( static_cast<long double>(n+h), static_cast<long double>(i) );
    base = base1 * base2;
    prob = counts[i]/base;

    ss << sep << prob;
  }

  for(uint_t i=1; i<=cnts; ++i)
  {
    if(n>=32 && counts[i]>(uint128_t(1)<<64))
      ss << sep << setprecision(12) << static_cast<long double>(counts[i]);
    else
      ss << sep << static_cast<uintll_t>(counts[i]);
  }

  for(int i=0; i<2; ++i)
    ss << sep << times[i];

  ss << sep << total;

  if(flags.verbose==1) {
    std::cout << "Kernel: " << times[0] << " s\n"
              << "Runtime: " << times[1] << " s\n"
              << flags.n
              << sep << A
              << sep << h
              << sep << flags.mc_iterations
              << sep << flags.mc_iterations_2
              << "\n";

  }
}


// quick&dirty:
// http://stackoverflow.com/questions/25114597/how-to-print-int128-in-g
std::ostream&
operator<<( std::ostream& dest, uint128_t value )
{
  std::ostream::sentry s( dest );
  if ( s ) {
    uint128_t tmp = value;
    char buffer[ 128 ];
    char* d = std::end( buffer );
    do
    {
      -- d;
      *d = "0123456789"[ tmp % 10 ];
      tmp /= 10;
    } while ( tmp != 0 );

    int len = std::end( buffer ) - d;
    if ( dest.rdbuf()->sputn( d, len ) != len ) {
      dest.setstate( std::ios_base::badbit );
    }
  }
  return dest;
}
