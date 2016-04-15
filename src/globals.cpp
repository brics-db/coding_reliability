#include "globals.h"

#include <helper.h>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

static
std::ostream&
operator<<( std::ostream& dest, uint128_t value );
void printbits(uintll_t v, uintll_t n);

static long double process_result(uint128_t* counts, Statistics stats, uint_t n, uint_t h, const char* file_prefix);

double get_rel_error_AN641(uint_t n, uint128_t* tgt, int offset)
{
  const uint128_t* sol = NULL;
  if(n==24)
    sol = solution_an24_A641;
  else if(n==16)
    sol = solution_an16_A641;
  else if(n==8)
    sol = solution_an8_A641;
  else
    return -1.0;
  double max_err=0;
  double err = 0;
  double p;
  for(uint_t k=offset; k<n+10; ++k)
  {
    p = static_cast<double>(sol[k]);// / sum;
    if(p>0.)
      err = fabs(static_cast<double>(tgt[k])/p-1.0);
    else if(tgt[k]>0)
      err = 1.0;

    if(err>max_err)
      max_err = err;
  }
  return max_err;
}
double get_abs_error_AN641(uint_t n, uint128_t* tgt, int offset)
{
  const uint128_t* sol = NULL;
  if(n==24)
    sol = solution_an24_A641;
  else if(n==16)
    sol = solution_an16_A641;
  else if(n==8)
    sol = solution_an8_A641;
  else
    return -1.0;
  double max_err=0;
  double err = 0;
  double p;
  for(uint_t k=offset; k<n+10; ++k)
  {
    p = static_cast<double>(sol[k]>tgt[k] ? sol[k]-tgt[k] : tgt[k]-sol[k]);
    err = p/(pow(2.0,10.0+n)*binomialCoeff(static_cast<double>(n)+10.0,static_cast<double>(k)));

    if(err>max_err)
      max_err = err;
  }
  return max_err;
}

long double process_result(uint128_t* counts, Statistics stats, uint_t n, uint_t h, const char* file_prefix)
{
  stringstream ss;
  char sep = ',';
  // sum
  long double total = 0;
  for (uint_t i = 0; i < n+h; ++i) {
    total += static_cast<long double>(counts[i]);
  }
  long double prob;
  long double base1 = pow(2.0,n);
  long double base2;
  long double base;
  for(uint_t i=0; i<n+h+1; ++i)
  {
    base2 =binomialCoeff( static_cast<long double>(n+h), static_cast<long double>(i) );
    base = base1 * base2;
    prob = counts[i]/base;
    if(n>=32 && counts[i]>(uint128_t(1)<<64))
      ss << setw(4) << i <<sep<< setw(12) << setprecision(12)<<static_cast<long double>(counts[i]);
    else
      ss << setw(4) << i <<sep<< setw(14) << static_cast<uintll_t>(counts[i]);
    ss << sep << setw(14) << prob <<sep<< setw(14)<< base << endl;
  }
  ss << endl << endl;
  for(int i=0; i<stats.getLength(); ++i)
    ss << i <<sep<< '"' << stats.getLabel(i) << '"' 
       <<sep<< stats.getAverage(i) 
       <<sep<< stats.getUnit(i) 
       <<sep<< total << endl;

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
  const uintll_t count_counts = n+h+1;
  const uintll_t bitcount_message = n+h;
  const uintll_t count_messages = (1ull << n);
  const uint_t count_edges_shift = (n+bitcount_message);
  const uint128_t count_edges = static_cast<uint128_t>(1)<<count_edges_shift;
  const uintll_t mul_1distance = bitcount_message;
  const uintll_t mul_2distance = bitcount_message * (bitcount_message - 1ull) / 2ull;

  long double total = process_result(counts, stats, n, h, file_prefix);

  // for 16-bit data:
  // 2^16 different messages --> 2^16 * 2^16 (legal code words) + 2^16*2^16*22 (all 1-bit-differences) - 2^16*23 (edges to self and self-1bit-diff)
/*  cout << "counted edges (undetectable bit flips, i.e. distance >= 3 to either other codewords or into the 1-distance sphere around other codewords)\n";
  cout << "    => \"all bit flips that result in either another codeword or in a word with distance of 1 to another codeword\"\n";
  cout << "    = " << total << "\n\n";

  cout << "total edges for " << n << "- bit data: " << (count_messages * count_messages * count_counts - (count_messages * count_counts)) << '\n';
  cout << "all possible edges for " << bitcount_message << "-bit messages...\n";
  cout << "    from codewords to all other words (extended hamming): 2^" << n << " * 2^" << bitcount_message << " = " << count_edges << '\n';
  cout << "    without self-edges (weight = 0) : 2^" << (n + bitcount_message) << " - 2^" << n << " = " << (count_edges - (0x1ull << n))
      << '\n';
  cout << "    without detectable bit flips (weight={0,1,2}) : 2^" << (n + bitcount_message) << " - 2^" << n << " - 2^" << n << "*"
      << mul_1distance << " - 2^" << n << "*(sum(i=" << (mul_1distance - 1) << ":1, i)\n";
  cout << "        = 2^" << count_edges_shift << " - 2^" << n << " - 2^" << n << "*" << mul_1distance << " - 2^" << n << "*"
      << mul_2distance;
  cout << "        = " << static_cast<uintll_t>(count_edges - (0x1ull << n) - (0x1ull << n) * mul_1distance - (0x1ull << n) * mul_2distance) << '\n';
*/
}


void process_result_ancoding(uint128_t* counts, Statistics stats, uint_t n, uint_t A, const char* file_prefix)
{
  char tmp_file_prefix[192];
  long double total;
  uint_t h = static_cast<uint_t>( ceil(log(A)/log(2.0)) );
  if(file_prefix){
    sprintf(tmp_file_prefix, "%s_A%u", file_prefix, A);
    total = process_result(counts,stats,n,h,tmp_file_prefix);
  }else
    total = process_result(counts,stats,n,h,nullptr);

  cout << endl << "Sum counts = " << setw(12)<<setprecision(12)<< total << endl;
  cout << " A = ";
  printbits(A, n);
  cout << " = " << A << endl;
}
void process_result_ancoding_mc(uint128_t* counts, Statistics stats, uint_t n, uint_t A, uint_t iterations, const char* file_prefix)
{
  char tmp_file_prefix[192];
  long double total;
  uint_t h = static_cast<uint_t>( ceil(log2(A)) );
  if(file_prefix){
    sprintf(tmp_file_prefix, "%s_m%u_A%u", file_prefix, iterations, A);
    total = process_result(counts,stats,n,h,tmp_file_prefix);
  }else
    total = process_result(counts,stats,n,h,nullptr);
  cout << endl << "Sum counts = " << setw(12)<<setprecision(12)<<total << "(with estimation factor 2^n*counts[d]/iterations)" << endl;
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
