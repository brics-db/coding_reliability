#!/bin/perl
#
#Copyright (C) 2015, 2016 Till Kolditz
#
#File: ancoding_count_undetectable_errors.cpp
#Authors:
#    Till Kolditz - till.kolditz@gmail.com
#
#This file is distributed under the Apache License Version 2.0; you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#

use warnings;
use strict;
use Tie::File;
use Math::Counting ':big';
use Math::BigFloat ':constant';
use Try::Tiny;

my $argc = @ARGV;
if ($argc != 2) {
	print "Usage: perl ancoding_eval.pl <file_in> <file_out>\n";
	exit;
}

my $file_in = $ARGV[0];
my $file_out = $ARGV[1];

tie my @dat, 'Tie::File', $file_in or die "Can't tie $file_in $!";
open (OUT , '>', $file_out) or die "Can't open file $file_out $!";

try {
	my $lastLine = $#dat;
	print "lastLine: $lastLine\n";

	my $minWidth = 1000000;
	my $maxWidth = 0;

	for my $i (0..$lastLine) {
		my $line = $dat[$i];
		my @parts = split('\t', $line);
		my $num = @parts;
		if ($num > 3) {
			$num -= 3;
			$minWidth = ($num) < $minWidth ? ($num) : $minWidth;
			$maxWidth = ($num) > $maxWidth ? ($num) : $maxWidth;
		}
	}

	print "minWidth: $minWidth\n";
	print "maxWidth: $maxWidth\n";

	my $lastNum = 0;
	my @combinations = ();
	my $combsTotal = 0;
	for my $i (0..$lastLine) {
		my $line = $dat[$i];
		my @parts = split('\t', $line);
		my $num = @parts;
		
		if ($num > 3) {
			$num -= 3;
			
			if ($lastNum != $num) {
				@combinations = ();
				$combsTotal = 0;
				for my $j (0..$num) {
					my $combs = $parts[2] * bcomb($num, $j);
					push(@combinations, $combs);
					$combsTotal += $combs;
				}
				#print join(", ", @combinations)."\n";
			}
			
			my $leadingZeroProbs = 0;
			for my $i (1..$num/2) {
				if ($parts[$i+2] == 0) {
					++$leadingZeroProbs;
				} else {
					last;
				}
			}
			my $trailingZeroProbs = 0;
			for my $i ($num..($num/2+1)) {
				if ($parts[$i+2] == 0) {
					++$trailingZeroProbs;
				} else {
					last;
				}
			}
			
			my $total = 0;
			for my $j (1..$num) {
				$total += $parts[$j+2];
			}
			
			print OUT $parts[0]."\t".$parts[1]."\t".$num;
			for my $i (0..$num) {
				print OUT "\t".$parts[$i+2];
			}
			for my $i = (($num+1)..$maxWidth) {
				print OUT "\t";
			}
			for my $i (1..$num) {
				#print sprintf("%s / %s = %.10f\n", $parts[$i+2], $combinations[$i], 1.0 * $parts[$i+2] / $combinations[$i]);
				print OUT sprintf("\t%.10f", 1.0 * $parts[$i+2] / $combinations[$i]);
			}
			#print OUT "\t$total\t$combsTotal\t".(1.0 * $total / $combsTotal);
			print OUT "\t$leadingZeroProbs\t$trailingZeroProbs\n";
			
			$lastNum = $num;
		} else {
			print OUT $line."\n";
		}
	}
} catch {
	warn "Got a die: $_"
} finally {
	untie @dat;
	close OUT;
}
