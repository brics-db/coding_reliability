# coding_reliability
How reliable are Hamming Code and AN-Coding?
============================================

Each generation of computer hardware gets faster and faster, currently reducing transistor feature sizes further and further. External influences like heat, cosmic rays, aging, and others already lead to bit flips in memory cells. The trend is that ICs with ever smaller transistors and ever higher densities (have you heard of Moore's Law? ;-)) become more and more susceptible to single and multi bit flips.
This project aims at quantifying the bit flip detection capabilities of some known data coding schemes by delivering absolute reliabilities, currently of Extended Hamming and AN Coding.
Absolute means that the values computed are not based on any error model and you will have to multiply all results by probabilities that reflect your error / system model.

This said, we wanted to know how many bit flips can be detected by either of the encodings for different data widths, in order to compare bit flip detection capabilities. What we acutally compute, however, is how many bit flips can *not* be detected.

The idea is the following:
We represent all code words as an undirected, weighted graph. The graph is fully connected and the edges' weights are computed by the Hamming distance between each codeword. Now we group-count up all the weights -- i.e. we count for each distance how many edges there are. It's as simple as that. Since the total number of K bit flips possible for a codeword of size N bits is N over K (i.e. possible combinations to flip bits where it is not important which bits flip first), you can compute the "absolute reliability" by dividing the computed distances' counts by the according number of combinations.

Extended Hamming Code
=====================
For Extended Hamming Code we extend the graph by the 1-bit-distance spheres of all codewords. That is because Extended Hamming corrects single bit errors and thus will decode into a wrong one after a bit flip which leads to another codeword or its 1-bit-distance sphere.

License
=======
This code is published under the Apache License Version 2.0. See file LICENSE.