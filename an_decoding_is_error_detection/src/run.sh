#!/usr/bin/env /bin/bash

# Copyright 2016 Till Kolditz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

{ time ./an_decoding_is_error_detection 1>results.out 2>results.err ; } >results.time &

grep -e BAD $(ls an_decoding_is_error_detection_*.out) | tee results.bad

cat results.out | grep -e done -e skipping | awk '{print $2}' >results.done

#files=$(ls an_decoding_is_error_detection_*.out); for num in $(cat an_decoding_is_error_detection.out3); do count=$(grep "BAD A=$num " $files | wc -l); if [[ $count -ne 0 ]]; then echo "$num $count" >> an_decoding_is_error_detection.out4; fi; done

rm -f results.all.out results.all.sorted
for f in $(ls an_decoding_is_error_detection_*.out); do cat $f >>results.all.out; done
sort -gt "|" -k 1 results.out.all >results.out.sorted
