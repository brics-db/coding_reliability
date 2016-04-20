/*
 * @copyright 2015, 2016 Till Kolditz <till.kolditz@gmail.com>
 * This file is distributed under the Apache License Version 2.0; you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 */

#include "globals.h"
#include "algorithms.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

struct Flags {
  int an_coding; // AN coded words
  uintll_t A;
  int h_coding; // hamming coded words
  int with_1bit;
  int with_mc; // with monte carlo
  int with_mc_v2; // with monte carlo
  int mc_iterations; // number of monte carlo iterations
  uintll_t n; // nr of bits as input size
  int use_cpu;
  int rand_gen;
  double mc_search_bound;
  int mc_search_super_A;
  int file_output;
  int nr_dev;
} g_flags = {0,61,0,0,0,0,100,8,0,0,-1.0,0,0,0};

void print_help(){
  printf("\nUSAGE:\n\n\t-h\tprint help\n"
	 "\t-e NUM\t Extended Hamming-coding algorithm (1=optimized, 2=naive, 3=naive+ordering, 4=as [2]+more optimizations)\n"
         "\t-b    \t ... with 1-bit spheres\n"
	 "\t-a NUM\t AN-coding algorithm with A=<NUM>\n"
         "\t-s DEC\t ... AN-coding search optimal number of monte carlo iterations for given error bound <1.0.\n"
         "\t      \t -m or -M gives start value of number of iterations.\n"
         "\t-S    \t ... search super A with same code word width as hamming\n"
         "\t-m NUM\t Monte-Carlo with number of iterations, GPU, for -e 2 and -a\n"
//         "\t-M NUM\t ... AN-coding with monte carlo v2 - and number of iterations\n"
	 "\t-n NUM\t number of bits as input size\n"
         "\t-c    \t use CPU implementation (has effect on -e 1 and -a)\n"
         "\t-t NUM\t test curand generator with number of iterations\n"
         "\t-f    \t with file output (file name is generated).\n"
         "\t-d NUM\t Number of GPUs to be used (0=max).\n"
	 "\n");
}

void parse_cmdline(int argc, char** argv)
{
  //default
  g_flags.h_coding=1;
  g_flags.file_output=0;

  for(int i=0;i<argc;++i)
  {
    if(strcmp(argv[i],"-h")==0){
      print_help();
      exit(0);
    }
    if(strcmp(argv[i],"-a")==0){
      g_flags.an_coding = 1;
      g_flags.h_coding = 0;
      g_flags.A = atoi(argv[i+1]);
      if(g_flags.A==0) g_flags.A = 1;
    }else if(strcmp(argv[i],"-e")==0){
      g_flags.h_coding = atoi(argv[i+1]);
      g_flags.an_coding = 0;
    }else if(strcmp(argv[i],"-c")==0){
      g_flags.use_cpu = 1;
    }else if(strcmp(argv[i],"-s")==0){
      g_flags.mc_search_bound=atof(argv[i+1]);
      assert(g_flags.mc_search_bound>0.);
    }else if(strcmp(argv[i],"-S")==0){
      g_flags.mc_search_super_A=1;
    }else if(strcmp(argv[i],"-t")==0){
      g_flags.rand_gen = 1;
      g_flags.mc_iterations=atoi(argv[i+1]);
      assert(g_flags.mc_iterations>0);
    }else if(strcmp(argv[i],"-m")==0){
      g_flags.with_mc = 1;
      g_flags.mc_iterations=atoi(argv[i+1]);
      assert(g_flags.mc_iterations>0);
    }/*
    else if(strcmp(argv[i],"-M")==0){
      g_flags.an_coding = 1;
      g_flags.with_mc_v2 = 1;
      g_flags.mc_iterations=atoi(argv[i+1]);
      assert(g_flags.mc_iterations>0);
    }*/
    if(strcmp(argv[i],"-f")==0)
      g_flags.file_output = 1;
    if(strcmp(argv[i],"-b")==0)
      g_flags.with_1bit = 1;
    if(strcmp(argv[i],"-n")==0)
    {
      g_flags.n=atoi(argv[i+1]);
    }
    if(strcmp(argv[i],"-d")==0)
    {
      g_flags.nr_dev=atoi(argv[i+1]);
      assert(g_flags.nr_dev>=0);
    }
  }  
  if(g_flags.h_coding && g_flags.n!=8 && g_flags.n!=16 && g_flags.n!=24 && g_flags.n!=32 && g_flags.n!=40 && g_flags.n!=48)
  {
    g_flags.n=8;
    printf("Wrong input size, so set n to 8.\n");       
  }
}

void ancoding_mc_search_super_A()
{
  uintll_t mincb = 0, mxmincb=static_cast<uintll_t>(-1), mxmincb2=static_cast<uintll_t>(-1);
  uintll_t minb = 0, mxminb=0, mxminb2=0, A2=0;
  uintll_t n = g_flags.n;
  uintll_t A = n<16 ? 17 : n<32 ? 33 : 65;
  uintll_t A_end = n<16 ? 31 : n<32 ? 63 : 127;
  uintll_t superA = 0;
  double times = 0;
  double ttimes = 0;
  for(;A<=A_end; A+=2)
  {
    if(n==8)
      run_ancoding(n, A, 0, &minb, &mincb, 0, g_flags.nr_dev);
    else
      run_ancoding_mc(n, g_flags.mc_iterations, A, 0, &times, &minb, &mincb, 0, g_flags.nr_dev);
      
    if(mxminb<minb || (mxminb==minb && mxmincb>mincb))
    {
      if(mxminb==3 && mxmincb<mxmincb2)
      {
        mxminb2=mxminb;
        mxmincb2=mxmincb;
        A2 = superA;
      }
      mxminb=minb;
      mxmincb=mincb;
      superA = A;
      printf("%zu: c[%zu] = %zu\n", A, minb, mincb);
    }
    ttimes += times;
  }
  printf("Found super A=%zu after %.3lf s.\n A=%zu\tprev.A=%zu\n c[%zu]=%zu\tc[%zu]=%zu\n", 
         superA, ttimes, superA, A2, mxminb,mxmincb, mxminb2,mxmincb2);
}

void ancoding_mc_search()
{
  int step=0;
  double cur_err=1.0;
  uintll_t nr_iterations = g_flags.mc_iterations;
  if(nr_iterations<1)
    nr_iterations=1;
  printf("AN-Coding MonteCarlo starting with %zd iterations\n", nr_iterations);
  FILE* f = fopen("ancoding_mc_search.csv","w");
  double times[2];
  while(1)
  {
    cur_err = run_ancoding_mc(g_flags.n, nr_iterations, g_flags.A, 0, times, nullptr, nullptr,0, g_flags.nr_dev);
    printf("%zd iterations - max. rel. error %lf\n",nr_iterations, cur_err);
    fprintf(f, "%d,%zd,%lf,%zd,%lf,%lf,%lf\n",
            step,g_flags.n, g_flags.mc_search_bound, nr_iterations, cur_err, times[0], times[1]);
    if(cur_err<=g_flags.mc_search_bound)
      break;
    nr_iterations*=3;
    ++step;
  }
  fclose(f);
  printf("Achieved error of %lf after %zd iterations\n", cur_err, nr_iterations);
}

int main(int argc, char** argv)
{
  if(argc<2){
    print_help();
    return 0;
  }
  parse_cmdline(argc,argv);
  if(g_flags.rand_gen)
    test_curand_raw(g_flags.n, g_flags.mc_iterations, g_flags.nr_dev);
  else if(g_flags.an_coding){  /* AN-coding */
    if(g_flags.mc_search_bound>0.0)
      ancoding_mc_search();
    else if(g_flags.mc_search_super_A)
      ancoding_mc_search_super_A();
    else if(g_flags.with_mc)
      run_ancoding_mc(g_flags.n, g_flags.mc_iterations,g_flags.A,1,nullptr,nullptr,nullptr,g_flags.file_output, g_flags.nr_dev);
//    else if(g_flags.with_mc_v2)
//      run_ancoding_mc_v2(g_flags.n, g_flags.mc_iterations,1);
    else if(g_flags.use_cpu)
      run_ancoding_cpu(g_flags.n, g_flags.A, 1, nullptr, nullptr,g_flags.file_output);
    else
      run_ancoding(g_flags.n, g_flags.A, 1, nullptr, nullptr,g_flags.file_output, g_flags.nr_dev);
  }else if(g_flags.h_coding==1){ /* Ext Hamming Code */
    if(g_flags.use_cpu)
      run_hamming_cpu(g_flags.n,g_flags.with_1bit,g_flags.file_output);
    else
      run_hamming(g_flags.n,g_flags.with_1bit,g_flags.file_output, g_flags.nr_dev);
  }else if(g_flags.h_coding==2){
    if(g_flags.with_mc){
      run_hamming_mc(g_flags.n,g_flags.with_1bit,g_flags.mc_iterations,g_flags.file_output, g_flags.nr_dev);
    }else
      run_hamming_cpu_native_short(g_flags.n,g_flags.with_1bit,g_flags.file_output);
  }else if(g_flags.h_coding==3)
    run_hamming_cpu_native_short_ordering(g_flags.n,g_flags.with_1bit,g_flags.file_output);
  else if(g_flags.h_coding==4)
    run_hamming_cpu_native(g_flags.n,g_flags.with_1bit,g_flags.file_output);
  return 0;
}

