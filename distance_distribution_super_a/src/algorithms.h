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

#ifndef ALGORITHMS_H_
#define ALGORITHMS_H_

#include "globals.h"

void run_ancoding(uintll_t A, uint_t h, Flags& flags, double* times, uint128_t* counts);
void run_ancoding_grid(uintll_t A, uint_t h, Flags& flags, double* times, uint128_t* counts);

#endif
