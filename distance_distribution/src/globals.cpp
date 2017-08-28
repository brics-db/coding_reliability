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
#include "solutions.h"
#include <helper.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

static const int OFFSET = 2;

void printbits(uintll_t v, uintll_t n);

static long double process_result(uint128_t* counts, Statistics stats, uint_t n, uint_t h, const char* file_prefix, double* errors_abs=nullptr, double* errors_rel=nullptr, std::string app = "");

static void get_sol_hamming(uint128_t* sol, uint_t n, uint_t *h_out)
{
  const uint128_t* sol_base;
  *h_out = n==8?5:n==16||n==24?6:7;
  uint_t h = *h_out;
  switch(n)
  {
  case 8:
    sol_base = sol_hamming_n8;
    break;
  case 16:
    sol_base = sol_hamming_n16;
    break;
  case 24:
    sol_base = sol_hamming_n24;
    break;
  case 32:
    sol_base = sol_hamming_n32;
    break;
  default:
    throw std::runtime_error("Wrong n for hamming.");
  }
  sol[0] = 1ull<<n;
// 1-bit sphere  
  sol[1] = (n+h)*sol[0];  
  for (uint_t i = 3; i < n+h+1; i+=2)
  {
    if(i+1<n+h+1){
      sol[i+1] = sol_base[i+1]<<n;
      sol[i] = uint128_t(i+1)*sol[i+1] + uint128_t(n+h-i+1)*sol[i-1];
    }else
      sol[i] = uint128_t(n+h-i+1)*sol[i-1];
  }    
}

double get_rel_error_hamming(uint_t n, uint128_t* tgt, int offset, int w1bit, double* errors)
{
  uint128_t sol[64] = {0};
  double max_err=0;
  double err = 0;
  double p;
  uint_t h = 0;
  if(errors)
    memset(errors,0,(n+h+1)*sizeof(double));
  get_sol_hamming(sol, n, &h);
  if(w1bit==0 && (offset&1)==1) offset+=1;
  for(uint_t k=offset; k<n+h+1; k+= w1bit?1:2)
  {
    p = static_cast<double>(sol[k]);
    if(p>0.)
      err = fabs(static_cast<double>(tgt[k])/p-1.0);
    else if(tgt[k]>0)
      err = 1.0;
    else
      err = 0.0;
    if(errors)
      errors[k] = err;
    if(err>max_err)
      max_err = err;
  }
  return max_err;
}

double get_abs_error_hamming(uint_t n, uint128_t* tgt, int offset, int w1bit, double* errors)
{
  uint128_t sol[64] = {0};
  double max_err=0;
  double err = 0;
  double p;
  uint_t h=0;
  if(errors)
    memset(errors,0,(n+h+1)*sizeof(double));
  get_sol_hamming(sol,n, &h);
  if(w1bit==0 && (offset&1)==1) offset+=1;
  for(uint_t k=offset; k<n+h+1; k+= w1bit?1:2)
  {
    p = static_cast<double>(sol[k]>tgt[k] ? sol[k]-tgt[k] : tgt[k]-sol[k]);
    err = p/(pow(2.0,n)*binomialCoeff(static_cast<double>(n)+h,static_cast<double>(k)));
    if(errors)
      errors[k] = err;

    if(err>max_err)
      max_err = err;
  }
  return max_err;
}

double get_rel_error_AN(uintll_t A, uint_t n, uint128_t* tgt, int offset, double* errors)
{
  const uint128_t* sol = NULL;
  if(A!=641 && A!=61) 
    return -1.0;
  if(n==24)
    sol = A==641 ? solution_an24_A641 : solution_an24_A61;
  else if(n==16)
    sol = A==641 ? solution_an16_A641 : solution_an16_A61;
  else if(n==8)
    sol = A==641 ? solution_an8_A641 : solution_an8_A61;
  else
    return -1.0;
  uint_t h = A==641 ? 10 : 6;
  double max_err=0;
  double err = 0;
  double p;
  if(errors)
    memset(errors,0,(n+h+1)*sizeof(double));
  for(uint_t k=offset; k<n+h; ++k)
  {
    p = static_cast<double>(sol[k]);
    if(p>0.)
      err = fabs(static_cast<double>(tgt[k])/p-1.0);
    else if(tgt[k]>0)
      err = 1.0;
    else
      err = 0.0;

    if(errors)
      errors[k] = err;
    if(err>max_err)
      max_err = err;
  }
  return max_err;
}
double get_abs_error_AN(uintll_t A, uint_t n, uint128_t* tgt, int offset, double* errors)
{
  const uint128_t* sol = NULL;
  if(A!=641 && A!=61) 
    return -1.0;
  if(n==24)
    sol = A==641 ? solution_an24_A641 : solution_an24_A61;
  else if(n==16)
    sol = A==641 ? solution_an16_A641 : solution_an16_A61;
  else if(n==8)
    sol = A==641 ? solution_an8_A641 : solution_an8_A61;
  else
    return -1.0;
  uint_t h = A==641 ? 10 : 6;
  double max_err=0;
  double err = 0;
  double p;
  if(errors)
    memset(errors,0,(n+h+1)*sizeof(double));
  for(uint_t k=offset; k<n+h; ++k)
  {
    p = static_cast<double>(sol[k]>tgt[k] ? sol[k]-tgt[k] : tgt[k]-sol[k]);
    err = p/(pow(2.0,n)*binomialCoeff(static_cast<double>(n)+h,static_cast<double>(k)));

    if(errors)
      errors[k] = err;
    if(err>max_err)
      max_err = err;
  }
  return max_err;
}

long double process_result(uint128_t* counts, Statistics stats, uint_t n, uint_t h, const char* file_prefix, double* errors_abs, double* errors_rel, string app)
{
  stringstream ss;
  char sep = ',';
  // sum
  long double total = 0;
  for (uint_t i = 0; i < n+h+1; ++i) {
    total += static_cast<long double>(counts[i]);
  }
  long double prob;
  long double base1 = pow(2.0,n);
  long double base2;
  long double base;
  double mxabs=0,mxrel=0;
  for(uint_t i=0; i<n+h+1; ++i)
  {
    base2 =binomialCoeff( static_cast<long double>(n+h), static_cast<long double>(i) );
    base = base1 * base2;
    prob = counts[i]/base;
    if(n>=32 && counts[i]>(uint128_t(1)<<64))
      ss << setw(4) << i <<sep<< setw(12) << setprecision(12)<<static_cast<long double>(counts[i]);
    else
      ss << setw(4) << i <<sep<< setw(14) << static_cast<uintll_t>(counts[i]);
    ss << sep << setw(14) << prob <<sep<< setw(14)<< base;
    if(errors_abs){
      ss << sep << setw(13) << setprecision(7) << errors_abs[i];
      if(errors_abs[i]>mxabs) mxabs=errors_abs[i];
    }
    if(errors_rel){
      ss << sep << setw(13) << setprecision(7) << errors_rel[i];
      if(errors_rel[i]>mxrel) mxrel=errors_rel[i];
    }
    ss << endl;
  }
  ss << endl << endl;

  for(int i=0; i<stats.getLength(); ++i)
    ss << i <<sep<< '"' << stats.getLabel(i) << '"' 
       <<sep<< stats.getAverage(i) 
       <<sep<< stats.getUnit(i)<<endl;
  
  ss << endl << "\"Total\"" <<sep<< total << endl;
  if(errors_abs) ss<<"\"MaxAbsErr\""<<sep<<mxabs<<endl;
  if(errors_rel) ss<<"\"MaxRelErr\""<<sep<<mxrel<<endl;
  ss << endl;
  ss << app;
  cout << ss.str();

  if(file_prefix)
  {
    char fname[256];
    ofstream fp;
    sprintf(fname,"%s_n%u_h%u.csv",file_prefix, n, h);
    fp.open(fname);
    fp << ss.str();
    fp.close();
    cout << "File " << fname << " written." << endl << endl;
  }
  return total;
}

void process_result_hamming(uint128_t* counts, Statistics stats, uint_t n, uint_t h, const char* file_prefix)
{
  long double total = process_result(counts, stats, n, h, file_prefix);
  cout << endl << "Sum counts = " << setw(12)<<setprecision(12)<< total << endl;
}
void process_result_hamming_mc(uint128_t* counts, Statistics stats, uint_t n, uint_t h, int with_1bit, uint_t iterations, const char* file_prefix)
{
  char tmp_file_prefix[192];
  long double total;
  double errors_abs[64];
  double errors_rel[64];
  get_abs_error_hamming(n, counts, OFFSET, with_1bit, errors_abs);
  get_rel_error_hamming(n, counts, OFFSET, with_1bit, errors_rel);
  if(file_prefix){
    sprintf(tmp_file_prefix, "%s_m%u", file_prefix, iterations);
    total = process_result(counts,stats,n,h,tmp_file_prefix, errors_abs, errors_rel);
  }else
    total = process_result(counts,stats,n,h,nullptr, errors_abs, errors_rel);

  cout << endl << "Sum counts = " << setw(12)<<setprecision(12)<<total << " (with estimation factor 2^n*counts[d]/iterations)" << endl;

}


void process_result_ancoding(uint128_t* counts, Statistics stats, uint_t n, uint_t A, const char* file_prefix, string app)
{
  char tmp_file_prefix[192];
  long double total;
  uint_t h = static_cast<uint_t>( ceil(log(A)/log(2.0)) );
  if(file_prefix){
    sprintf(tmp_file_prefix, "%s_A%u", file_prefix, A);
    total = process_result(counts,stats,n,h,tmp_file_prefix,nullptr,nullptr,app);
  }else
    total = process_result(counts,stats,n,h,nullptr,nullptr,nullptr,app);

  cout << endl << "Sum counts = " << setw(12)<<setprecision(12)<< total << endl;
  cout << " A = ";
  printbits(A, n);
  cout << " = " << A << endl;
}
void process_result_ancoding_mc(uint128_t* counts, Statistics stats, uint_t n, uint_t A, uint_t iterations, const char* file_prefix, string app)
{
  char tmp_file_prefix[192];
  double errors_abs[64];
  double errors_rel[64];
  get_abs_error_AN(A, n, counts, OFFSET, errors_abs);
  get_rel_error_AN(A, n, counts, OFFSET, errors_rel);
  long double total;
  uint_t h = static_cast<uint_t>( ceil(log2(A)) );
  if(file_prefix){
    sprintf(tmp_file_prefix, "%s_m%u_A%u", file_prefix, iterations, A);
    total = process_result(counts,stats,n,h,tmp_file_prefix, errors_abs, errors_rel, app);
  }else
    total = process_result(counts,stats,n,h,nullptr, errors_abs, errors_rel, app);
  cout << endl << "Sum counts = " << setw(12)<<setprecision(12)<<total << " (with estimation factor 2^n*counts[d]/iterations)" << endl;
  cout << " A = ";
  printbits(A, n);
  cout << " = " << A << endl;  
}

void printbits(uintll_t v, uintll_t n) 
{
  uintll_t mask = (1ull<<n)-1;
  uintll_t p=n;
  v &= mask;
  for (; v&mask; v <<= 1, --p) 
    putchar('0' + ((v&mask)>>(n-1)));
  for (; p; --p)
    putchar('0');
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
