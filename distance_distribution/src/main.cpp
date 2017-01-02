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
  int with_grid; // with monte carlo
  int mc_iterations; // number of monte carlo iterations
  int mc_iterations_2; // for 2D-grid
  uintll_t n; // nr of bits as input size
  int use_cpu;
  int rand_gen;
  double mc_search_bound;
  int search_super_A;
  int file_output;
  int nr_dev;
  int verbose;
} g_flags = {0,61,0,0,0,0,100,0,8,0,0,-1.0,0,0,0,1};

void print_help(){
  printf("\nUSAGE:\n\n\t-h\tprint help\n"
	 "\t-e NUM\t Extended Hamming-coding algorithm (1=optimized, 2=naive, 3=naive+ordering, 4=as [2]+more optimizations)\n"
         "\t-b    \t ... with 1-bit spheres\n"
	 "\t-a NUM\t AN-coding algorithm with A=<NUM>\n"
         "\t-g NUM\t 1=1D-Grid 2=2D-Grid approximation with -m NUM points per dim. (Better than MonteCarlo)\n"
         "\t-s DEC\t ... AN-coding search optimal number of monte carlo iterations for given error bound <1.0.\n"
         "\t      \t -m or -M gives start value of number of iterations.\n"
         "\t-S NUM\t ... search super A within 2^NUM<A<2^(NUM+1)\n"
         "\t-m NUM\t Monte-Carlo with number of iterations, GPU, for -e 2 and -a\n"
         "\t-M NUM\t ... number of iterations in second dimension for 2D grid\n"
	 "\t-n NUM\t number of bits as input size\n"
         "\t-c    \t use CPU implementation (has effect on -e 1 and -a)\n"
         "\t-t NUM\t test curand generator with number of iterations\n"
         "\t-f    \t with file output (file name is generated).\n"
         "\t-d NUM\t Number of GPUs to be used (0=max).\n"
         "\t-v NUM\t Verbosity level (0,1,2).\n"
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
    if(strcmp(argv[i],"-v")==0){
      g_flags.verbose = atoi(argv[i+1]);
    }else if(strcmp(argv[i],"-a")==0){
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
      g_flags.search_super_A=atoi(argv[i+1]);
      assert(g_flags.search_super_A>0);
    }else if(strcmp(argv[i],"-t")==0){
      g_flags.rand_gen = 1;
      g_flags.mc_iterations=atoi(argv[i+1]);
      assert(g_flags.mc_iterations>0);
    }else if(strcmp(argv[i],"-m")==0){
      g_flags.with_mc = 1;
      g_flags.mc_iterations=atoi(argv[i+1]);
      assert(g_flags.mc_iterations>0);    
    }else if(strcmp(argv[i],"-M")==0){
      g_flags.with_mc = 1;
      g_flags.with_grid = 2;
      g_flags.mc_iterations_2=atoi(argv[i+1]);
      assert(g_flags.mc_iterations_2>0);    
    }else if(strcmp(argv[i],"-g")==0){
      g_flags.with_grid = atoi(argv[i+1]);
      assert(g_flags.with_grid<3 && g_flags.with_grid>0);         
    }
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
  if(g_flags.h_coding && g_flags.n!=4 && g_flags.n!=8 && g_flags.n!=16 && g_flags.n!=24 && g_flags.n!=32 && g_flags.n!=40 && g_flags.n!=48)
  {
    g_flags.n=8;
    printf("Wrong input size, so set n to 8.\n");       
  }
  if(g_flags.with_grid==2 && g_flags.mc_iterations_2==0)
    g_flags.mc_iterations_2 = g_flags.mc_iterations; 
}

void ancoding_search_super_A()
{
  uint128_t mincb = 0, mxmincb=static_cast<uint128_t>(-1), mxmincb2=static_cast<uint128_t>(-1);
  uintll_t minb = 0, mxminb=0, mxminb2=0, A2=0;
  uintll_t n = g_flags.n;
  uintll_t A = (1<<(g_flags.search_super_A-1))+1;
  uintll_t A_end =(1<<(g_flags.search_super_A))-1;
  uintll_t superA = 0;
  double times[2] = {0};
  double ttimes = 0;
  for(;A<=A_end; A+=2)
  {
    if(n<=16)
      run_ancoding(n, A, 0,times, &minb, &mincb, 0, g_flags.nr_dev);
    else
      run_ancoding_grid(g_flags.with_grid,
                        g_flags.n,
                        g_flags.mc_iterations, g_flags.mc_iterations_2,
                        A,
                        0, times, &minb, &mincb,
                        0,
                        g_flags.nr_dev);
    if(mxminb<minb || (mxminb==minb && mxmincb>mincb))
    {
      if(mxminb!=minb)
      {
        mxminb2=mxminb;
        mxmincb2=mxmincb;
        A2 = superA;
      }
      mxminb=minb;
      mxmincb=mincb;
      superA = A;
      std::cout << A<<": c["<<minb<<"] = "<<mincb<<std::endl;
    }
    ttimes += times[1];
  }
  std::cout<<"n,"<<n
           <<",h,"<<g_flags.search_super_A
           <<",superA,"<<superA
           <<",prevA,"<<A2
           <<",c["<<mxminb<<"],"<<mxmincb
           <<",c["<<mxminb2<<"],"<<mxmincb2
           <<",time[s],"<<ttimes
           <<(n<=16?",exact":",grid")
           <<std::endl;
}
/// Find #iterations according to a given error bound
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
  else if(g_flags.an_coding || g_flags.search_super_A){  /* AN-coding */
    if(g_flags.mc_search_bound>0.0)
      ancoding_mc_search();
    else if(g_flags.search_super_A)
      ancoding_search_super_A();
    else if(g_flags.with_mc){
      if(g_flags.with_grid)
        run_ancoding_grid(g_flags.with_grid, 
                          g_flags.n, 
                          g_flags.mc_iterations, g_flags.mc_iterations_2,
                          g_flags.A,
                          g_flags.verbose,nullptr,nullptr,nullptr,
                          g_flags.file_output, 
                          g_flags.nr_dev);
      else
        run_ancoding_mc(g_flags.n, 
                        g_flags.mc_iterations,
                        g_flags.A,
                        g_flags.verbose,nullptr,nullptr,nullptr,
                        g_flags.file_output, 
                        g_flags.nr_dev);
    }else if(g_flags.use_cpu)
      run_ancoding_cpu(g_flags.n, 
                       g_flags.A, 
                       g_flags.verbose, nullptr, nullptr, nullptr,
                       g_flags.file_output);
    else
      run_ancoding(g_flags.n, 
                   g_flags.A, 
                   g_flags.verbose, nullptr, nullptr, nullptr,
                   g_flags.file_output, 
                   g_flags.nr_dev);
  }else if(g_flags.h_coding==1){ /* Ext Hamming Code */
    if(g_flags.use_cpu)
      run_hamming_cpu(g_flags.n,g_flags.with_1bit,g_flags.file_output);
    else if(g_flags.with_grid)
    {      
      run_hamming_grid(g_flags.n,
                       g_flags.with_1bit,
                       g_flags.mc_iterations,
                       g_flags.file_output, 
                       g_flags.nr_dev);
    }else if(g_flags.with_mc)
      run_hamming_mc(g_flags.n,
                     g_flags.with_1bit,
                     g_flags.mc_iterations,
                     g_flags.file_output, 
                     g_flags.nr_dev);    
    else
      run_hamming(g_flags.n,g_flags.with_1bit,g_flags.file_output, g_flags.nr_dev);
  }else if(g_flags.h_coding==2){
    run_hamming_cpu_native_short(g_flags.n,g_flags.with_1bit,g_flags.file_output);
  }else if(g_flags.h_coding==3)
    run_hamming_cpu_native_short_ordering(g_flags.n,g_flags.with_1bit,g_flags.file_output);
  else if(g_flags.h_coding==4)
    run_hamming_cpu_native(g_flags.n,g_flags.with_1bit,g_flags.file_output);
  return 0;
}

