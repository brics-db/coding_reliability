
// Copyright 2016 Matthias Werner, Till Kolditz
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
#include "algorithms.h"

#include <helper.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>

Flags g_flags;

void print_help(){
  printf("\nUSAGE:\n\n\t-h\tprint help\n"
         "\t-a NUM\t AN-coding algorithm with A=<NUM>\n"
         "\t-s NUM\t ... search super A beginning with NUM\n"
         "\t-S NUM\t ... search super A ending with NUM\n"
         "\t-c NUM\t ... search super A by file which contains the candidates\n"
         "\t-m NUM\t 1-D grid approximation with NUM iterations\n"
         "\t-M NUM\t 2-D grid approximation with NUM iterations in second dimension\n"
         "\t-k NUM\t number of bits as input size\n"
         "\t-f STR\t prefix of output file (parameters are added).\n"
         "\t-d NUM\t Number of GPUs to be used (0=max).\n"
         "\t-i NUM\t GPU device index.\n"
         "\t-v NUM\t Verbosity level (0,1,2).\n"
         "\n");
}

void parse_cmdline(int argc, char** argv)
{

  for(int i=0;i<argc;++i)
  {
    if(strcmp(argv[i],"-h")==0){
      print_help();
      exit(0);
    }
    if(strcmp(argv[i],"-v")==0){
      g_flags.verbose = atoi(argv[i+1]);
    }else if(strcmp(argv[i],"-a")==0){
      g_flags.A = atoi(argv[i+1]);
      if(g_flags.A==0) g_flags.A = 1;
    }else if(strcmp(argv[i],"-s")==0){
      g_flags.search_super_A=1;
      g_flags.search_start=atoi(argv[i+1]);
    }else if(strcmp(argv[i],"-S")==0){
      g_flags.search_super_A=1;
      g_flags.search_end=atoi(argv[i+1]);
    }else if(strcmp(argv[i],"-c")==0){
      g_flags.search_super_A=2;
      g_flags.search_file=argv[i+1];
    }else if(strcmp(argv[i],"-m")==0){
      g_flags.with_grid = 1;
      g_flags.mc_iterations=atoi(argv[i+1]);
      assert(g_flags.mc_iterations>0);    
    }else if(strcmp(argv[i],"-M")==0){
      g_flags.with_grid = 2;
      g_flags.mc_iterations_2=atoi(argv[i+1]);
      assert(g_flags.mc_iterations_2>0);    
    }
    if(strcmp(argv[i],"-f")==0)
      g_flags.file_prefix = argv[i+1];
    if(strcmp(argv[i],"-k")==0)
    {
      g_flags.n=atoi(argv[i+1]);
    }
    if(strcmp(argv[i],"-d")==0)
    {
      g_flags.nr_dev=atoi(argv[i+1]);
      assert(g_flags.nr_dev>=0);
    }
    if(strcmp(argv[i],"-i")==0)
    {
      g_flags.dev=atoi(argv[i+1]);
      assert(g_flags.dev>=0);
      assert(g_flags.dev<g_flags.nr_dev);
      g_flags.nr_dev = 1;
    }
  }
  if(g_flags.with_grid==2 && g_flags.mc_iterations_2==0)
    g_flags.mc_iterations_2 = g_flags.mc_iterations;
  if(g_flags.search_start>g_flags.search_end)
    g_flags.search_end = g_flags.search_start;
  if(g_flags.search_start==0 && g_flags.search_end>0)
    g_flags.search_start = g_flags.search_end;
  if(!g_flags.file_prefix)
    g_flags.file_prefix = "result";
  if(g_flags.search_super_A==2) {
    std::ifstream f(g_flags.search_file);
    if(f.good()==false)
      throw std::runtime_error("Candidates file does not exist.");
  }
}

void getCUDADeviceInformations(std::ostream& info, int dev) {
  cudaDeviceProp prop;
  int runtimeVersion = 0;
  size_t f=0, t=0;
  CHECK_ERROR( cudaRuntimeGetVersion(&runtimeVersion) );
  CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
  CHECK_ERROR( cudaMemGetInfo(&f, &t) );
  info << '"' << prop.name << '"'
       << ", \"CC\", " << prop.major << '.' << prop.minor
       << ", \"PCI Bus ID\", " << prop.pciBusID
       << ", \"PCI Device ID\", " << prop.pciDeviceID
       << ", \"Multiprocessors\", "<< prop.multiProcessorCount
       << ", \"Memory [MiB]\", "<< t/1048576
       << ", \"MemoryFree [MiB]\", " << f/1048576
       << ", \"ECC enabled\", " << prop.ECCEnabled
       << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
       << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
       << ", \"CUDA Runtime\", " << runtimeVersion
    ;
  if(g_flags.verbose==1) {
    std::cout << dev << ": " << prop.name << "\n";
  }
}

void header(std::ostream& ss)
{
  int dev = 0;
  int end = g_flags.nr_dev;
  if(g_flags.dev>=0) {
    dev = g_flags.dev;
    end = dev+1;
  }
  // skip init time
  for(; dev < end; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaDeviceSynchronize() );
    getCUDADeviceInformations(ss, dev);
    CHECK_ERROR( cudaDeviceSynchronize() );
  }
}

void get_lowest_prob(uint128_t* counts, size_t count_counts, uint128_t& mincb, uintll_t& minb)
{
  minb  = 0xFFFF;
  mincb = static_cast<uint128_t>(-1);
  for(uint_t i=1; i<(count_counts+1)/2; ++i)
  {
    if(counts[i]!=0 && counts[i]<mincb)
    {
      minb=i;
      mincb=counts[i];
      break;
    }
  }
}

template<typename TList>
void parse_candidates_file(const char* fname, TList& result) {
  std::ifstream fp;
  std::string tmp;
  std::string token;
  fp.open(fname);
  while (!fp.eof()) {
    getline(fp, tmp, '\n');
    std::stringstream ss(tmp);
    while (std::getline(ss, token, ',')) {
      if(!token.empty()) {
        uintll_t val = std::stoll(token);
        if(val==0)
          std::cerr << "Read A has value 0 [ignored].\n";
        else
          result.push_back(val);
      }
    }
  }
  fp.close();
}

double ancoding_search_super_A_by_file(std::stringstream& ss) {
  uint128_t mincb = 0, mxmincb=static_cast<uint128_t>(-1), mxmincb2=static_cast<uint128_t>(-1);
  uintll_t minb = 0, mxminb=0, mxminb2=0, superA = 0, superA2=0;
  uintll_t n = g_flags.n;
  double times[2] = {0};
  double ttimes = 0;
  uint128_t counts[COUNTS_MAX_WIDTH] = {0};
  const char sep = ',';

  std::vector<uintll_t> list_As;
  parse_candidates_file(g_flags.search_file, list_As);

  for(auto it=list_As.cbegin(); it!=list_As.cend(); ++it)
  {
    uintll_t A = *it;
    uint_t h = floor(log(A)/log(2.0))+1;

    if(g_flags.with_grid==0)
      run_ancoding(A, h, g_flags, times, counts);
    else
      run_ancoding_grid(A, h, g_flags, times, counts);

    get_lowest_prob(counts, n+h+1, mincb, minb);

    if(mxminb<minb || (mxminb==minb && mxmincb>mincb))
    {
      mxminb=minb;
      mxmincb=mincb;
      superA = A;
    }
    if(mxminb2<minb || (mxminb2==minb && mxmincb2>mincb))
    {
      mxminb2=minb;
      mxmincb2=mincb;
      superA2 = A;
    }
    ttimes += times[1];

    process_result(ss, A, h, g_flags, times, counts);
    ss << sep << mxminb << sep << mxmincb << sep << superA
       << sep << mxminb2 << sep << mxmincb2 << sep << superA2;
    ss << "\n";

    if( ((A+1) & ~A) == (A+1) ) // is A+1 power-of-two? (A+2 will be in a new ²-class)
    {
      mxmincb2=static_cast<uint128_t>(-1);
      mxminb2=0;
      superA2=0;
    }
  }
  return ttimes;
}

double ancoding_search_super_A(std::stringstream& ss)
{
  uint128_t mincb = 0, mxmincb=static_cast<uint128_t>(-1), mxmincb2=static_cast<uint128_t>(-1);
  uintll_t minb = 0, mxminb=0, mxminb2=0, superA = 0, superA2=0;
  uintll_t n = g_flags.n;
  uintll_t A = g_flags.search_start;
  uintll_t A_end = g_flags.search_end;
  double times[2] = {0};
  double ttimes = 0;
  uint128_t counts[COUNTS_MAX_WIDTH] = {0};
  const char sep = ',';

  for(;A<=A_end; A+=2)
  {
    uint_t h = floor(log(A)/log(2.0))+1;

    if(g_flags.with_grid==0)
      run_ancoding(A, h, g_flags, times, counts);
    else
      run_ancoding_grid(A, h, g_flags, times, counts);

    get_lowest_prob(counts, n+h+1, mincb, minb);

    if(mxminb<minb || (mxminb==minb && mxmincb>mincb))
    {
      mxminb=minb;
      mxmincb=mincb;
      superA = A;
    }
    if(mxminb2<minb || (mxminb2==minb && mxmincb2>mincb))
    {
      mxminb2=minb;
      mxmincb2=mincb;
      superA2 = A;
    }
    ttimes += times[1];

    process_result(ss, A, h, g_flags, times, counts);
    ss << sep << mxminb << sep << mxmincb << sep << superA
       << sep << mxminb2 << sep << mxmincb2 << sep << superA2;
    ss << "\n";

    if( ((A+1) & ~A) == (A+1) ) // is A+1 power-of-two? (A+2 will be in a new ²-class)
    {
      mxmincb2=static_cast<uint128_t>(-1);
      mxminb2=0;
      superA2=0;
    }
  }
  return ttimes;
}

void getDeviceInfos(int dev) {
  cudaDeviceProp prop;
  int runtimeVersion = 0;
  size_t f=0, t=0;
  CHECK_ERROR( cudaRuntimeGetVersion(&runtimeVersion) );
  CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
  CHECK_ERROR( cudaMemGetInfo(&f, &t) );
  std::cout << '"' << prop.name << '"'
       << ", \"CC\", " << prop.major << '.' << prop.minor
       << ", \"PCI Bus ID\", " << prop.pciBusID
       << ", \"PCI Device ID\", " << prop.pciDeviceID
       << ", \"Multiprocessors\", "<< prop.multiProcessorCount
       << ", \"Memory [MiB]\", "<< t/1048576
       << ", \"MemoryFree [MiB]\", " << f/1048576
       << ", \"ECC enabled\", " << prop.ECCEnabled
       << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
       << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
       << ", \"CUDA Runtime\", " << runtimeVersion
       << std::endl;
    ;
}

int main(int argc, char** argv)
{
  std::stringstream ss;
  char fname[256];
  double ttimes = 0;


  int ndevs=0;
  CHECK_ERROR( cudaGetDeviceCount( &ndevs ) );
  if(ndevs==0)
    throw std::runtime_error("No CUDA capable device found");

  for(int i=0; i<ndevs; ++i)
    getDeviceInfos(i);

  if(argc<2){
    print_help();
    return 0;
  }

  parse_cmdline(argc,argv);

  header(ss);
  ss<<"\n";

  if(g_flags.search_super_A==1)
  {
    sprintf(fname,"%s_k%u_a%u-%u.csv",g_flags.file_prefix, g_flags.n, g_flags.search_start, g_flags.search_end );
    ttimes = ancoding_search_super_A(ss);
  }
  else if(g_flags.search_super_A==2)
  {
    std::string tmp ( g_flags.search_file );
    std::replace(tmp.begin(), tmp.end(), '/', '_');
    sprintf(fname,"%s_k%u_c_%s.csv",g_flags.file_prefix, g_flags.n, tmp.c_str() );
    ttimes = ancoding_search_super_A_by_file(ss);
  }
  else if(g_flags.with_grid)
  {
    uint128_t counts[COUNTS_MAX_WIDTH] = {0};
    double times[2] = {0};
    uint_t h = ceil(log(g_flags.A)/log(2.0));
    sprintf(fname,"%s_k%u_h%u_a%u_g%u.csv",g_flags.file_prefix, g_flags.n, h, g_flags.A, g_flags.with_grid );
    run_ancoding_grid(g_flags.A, h, g_flags, times, counts);
    process_result(ss, g_flags.A, h, g_flags, times, counts);
    ttimes = times[1];
  }
  else
  {
    uint128_t counts[COUNTS_MAX_WIDTH] = {0};
    double times[2] = {0};
    uint_t h = ceil(log(g_flags.A)/log(2.0));
    sprintf(fname,"%s_k%u_h%u_a%u.csv",g_flags.file_prefix, g_flags.n, h, g_flags.A );
    run_ancoding(g_flags.A, h, g_flags, times, counts);
    process_result(ss, g_flags.A, h, g_flags, times, counts);
    ttimes = times[1];

  }

  std::ofstream fp;
  fp.open(fname);
  fp << "\"Total Time [s]\"," << ttimes << "\n";
  fp << ss.str();
  fp.close();

  return 0;
}

