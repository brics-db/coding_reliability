#!/usr/bin/env /bin/bash

#{ time ./an_decoding_is_error_detection 1>an_decoding_is_error_detection.out 2>an_decoding_is_error_detection.err ; } >an_decoding_is_error_detection.time &

#grep -e BAD $(ls an_decoding_is_error_detection_*.out)

#files=$(ls an_decoding_is_error_detection_*.out); for num in $(cat an_decoding_is_error_detection.out3); do count=$(grep "BAD A=$num " $files | wc -l); if [[ $count -ne 0 ]]; then echo "$num $count" >> an_decoding_is_error_detection.out4; fi; done

#rm -f an_decoding_is_error_detection_all.out an_decoding_is_error_detection_all_sorted.out
#for f in $(ls an_decoding_is_error_detection_*.out); do cat $f >> an_decoding_is_error_detection_all.out; done
#sort -gt "|" -k 1 an_decoding_is_error_detection_all.out >an_decoding_is_error_detection_all_sorted.out

#cat an_decoding_is_error_detection.out | grep -e done -e skipping | awk '{print $2}' >an_decoding_is_error_detection.done
#cat an_decoding_is_error_detection.done | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/ /g' >an_decoding_is_error_detection.done2
DONE=$(cat an_decoding_is_error_detection.done | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/ /g')
DONE2=$(grep -e computing an_decoding_is_error_detection.out | awk '{print $2}')
for A in ${DONE2}; do echo -n "$A : "; grep -e $A an_decoding_is_error_detection.done; echo ""; done
#{ time ./an_decoding_is_error_detection - <an_decoding_is_error_detection.done 1>an_decoding_is_error_detection.out 2>an_decoding_is_error_detection.err ; } &>an_decoding_is_error_detection.time &

