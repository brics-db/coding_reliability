/*
 * Compile e.g. with
 *   g++ -std=c++17 -O3 -march=native -o ANcandidates_d2-24_A2-24 ANcandidates.cpp -fopenmp -lpthread
 * Run e.g. with
 *   /usr/bin/time /bin/bash -c "OMP_THREAD_LIMIT=\$(nproc) ./ANcandidates_d2-24_A2-24 1>ANcandidates_d2-24_A2-24.out 2>ANcandidates_d2-24_A2-24.err"
 *   /usr/bin/time /bin/bash -c "OMP_THREAD_LIMIT=\$(echo \"\$(nproc)-2\"|bc) ./ANcandidates_d2-24_A2-24 1>ANcandidates_d2-24_A2-24.out 2>ANcandidates_d2-24_A2-24.err"
 */

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <map>
#include <vector>
#include <cerrno>

typedef typename std::string::size_type str_size_t;

struct data {
	size_t n;
	size_t k;
	size_t a;
	size_t sdr;
	size_t numCandidates;

	data() : n(-1), k(-1), a(-1), sdr(-1), numCandidates(-1) {
	}

	data(int n, int k, int a, int sdr, int numCandidates) : n(n), k(k), a(a), sdr(sdr), numCandidates(numCandidates) {
	}

	bool operator==(const data & other) {
		return (n == other.n) && (k == other.k) && (a == other.a) && (sdr == other.sdr) && (numCandidates == other.numCandidates);
	}

	bool operator<(const data & other) {
		return (k == other.k && a < other.a) || k < other.k;
	}

	bool compare_to(const data & other, std::ostream & out) const {
		bool isOK = true;
		if (n != other.n) {
			out << "n(" << n << ") != other.n(" << other.n << ')';
			isOK = false;
		}
		if (k != other.k) {
			out << (isOK ? "" : " ") << "k(" << k << ") != other.k(" << other.k << ')';
			isOK = false;
		}
		if (a != other.a) {
			out << (isOK ? "" : " ") << "a(" << a << ") != other.a(" << other.a << ')';
			isOK = false;
		}
		if (sdr != other.sdr) {
			out << (isOK ? "" : " ") << "sdr (" << sdr << ") != other.sdr(" << other.sdr << ')';
			isOK = false;
		}
		if (numCandidates != other.numCandidates) {
			out << (isOK ? "" : " ") << "numCandidates (" << numCandidates << ") != other.numCandidates(" << other.numCandidates << ')';
			isOK = false;
		}
		if (isOK) {
			out << "same data";
		}
		return isOK;
	}

	bool compare_to(const data & other) const {
		return compare_to(other, std::clog);
	}

		std::ostream & operator<<(std::ostream & out) {
			 return out << '[' << n << '/' << k << '/' << a << '/' << sdr << '/' << numCandidates << ']';
		}
};

std::ostream & operator<<(std::ostream & out, data & d) {
	 return out << '[' << d.n << '/' << d.k << '/' << d.a << '/' << d.sdr << '/' << d.numCandidates << ']';
}

struct data_file_out {
	const data & d;
	data_file_out(const data & d) : d(d) {
	}
};

std::ostream & operator<<(std::ostream & out, data_file_out && d) {
	 return out << d.d.n << '\t' << d.d.k << '\t' << d.d.a << '\t' << d.d.sdr << '\t' << d.d.numCandidates;
}

namespace std
{
    template<> struct hash<::data>
    {
        typedef ::data argument_type;
        typedef std::size_t result_type;
		// static constexpr const result_type multiplier = std::numeric_limits<ssize_t>::max() >> (sizeof(ssize_t) * 4); // OK, let's assume we never get to such large data lengths and widths of A
		static constexpr const result_type multiplier = 128; // OK, let's assume we never get to such large data lengths and widths of A
        result_type operator()(const argument_type & d) const noexcept
        {
            // result_type const h1 ( std::hash<std::string>{}(d.n) );
            // result_type const h2 ( std::hash<std::string>{}(d.k) );
            // return h1 ^ (h2 << 1); // or use boost::hash_combine (see Discussion)
			return d.k * multiplier + d.a;
        }
        result_type operator()(argument_type & d) const noexcept
        {
            // result_type const h1 ( std::hash<std::string>{}(d.n) );
            // result_type const h2 ( std::hash<std::string>{}(d.k) );
            // return h1 ^ (h2 << 1); // or use boost::hash_combine (see Discussion)
			return d.k * multiplier + d.a;
        }
    };
	bool operator==(const ::data & d1, const ::data & d2) {
		return hash<::data>{}(d1) == hash<::data>{}(d2);
	}
	bool operator<(const ::data & d1, const ::data & d2) {
		return hash<::data>{}(d1) < hash<::data>{}(d2);
	}
}

typedef std::map<::data, std::vector<size_t>> MyMap;

// std::cout << "# bit width data = [" << minBitWidthData << ',' << maxBitWidthData << "]\n";
// std::cout << "# bit width A = [" << minBitWidthA << ',' << maxBitWidthA << "]\n";
// std::cout << "# n=|C|\tk=|D|\t|A|\t|SDR|\t|candidates|\tcandidates\n";

void check_istringstream(std::istringstream & iss, const char * const filename, bool is_eof_OK = false) {
	if (iss.good() || (is_eof_OK && iss.eof())) {
		return;
	} else if (iss.eof()) {
		std::stringstream ss;
		ss << "EOF came too early in \"" << filename << '"';
		throw std::runtime_error(ss.str());
	} else if (iss.fail()) {
		std::stringstream ss;
		ss << "fail in \"" << filename << "\": " << std::strerror(errno);
		throw std::runtime_error(ss.str());
	} else {
		std::stringstream ss;
		ss << "bad in \"" << filename << "\": " << std::strerror(errno);
		throw std::runtime_error(ss.str());
	}
}

void parse_candidates(std::istringstream & issLine, std::string & strCandidate, data & d, std::vector<size_t> & vec, const char* const filename) {
	size_t numCandidate = 0;
	while (std::getline(issLine, strCandidate, ',')) {
		++numCandidate;
		const char * str_beg = strCandidate.c_str();
		char * str_end = nullptr;
		ssize_t candidate = strtoll(str_beg, &str_end, 0); // auto-detect base
		if (str_end == str_beg) {
			std::clog << "Error at candidate " << numCandidate << ". ";
		} else if (errno == ERANGE) {
			std::clog << "Parsed number is too large at candidate " << numCandidate << ". ";
		} else {
			vec.push_back(candidate);
		}
	}
	if (numCandidate != d.numCandidates) {
		std::clog << "numCandidate(" << numCandidate << ") != d.numCandidates(" << d.numCandidates << ") ";
	}
	check_istringstream(issLine, filename, true);
}

bool compare_candidate_vectors(const std::vector<size_t> & vec1, const std::vector<size_t> & vec2, std::ostream & out) {
	bool is_same_size = vec1.size() == vec2.size();
	if (!is_same_size) {
		out << " the previous one is " << (vec1.size() < vec2.size() ? "smaller" : "larger") << " than the new one!";
	}
	bool are_same_candidates = true;
	const auto vec1end = vec1.end();
	const auto vec2end = vec2.end();
	for (auto iter1 = vec1.begin(), iter2 = vec2.begin(); iter1 != vec1end && iter2 != vec2end; ++iter1, ++iter2) {
		if (*iter1 != *iter2) {
			are_same_candidates = false;
			out << " *iter1 (" << *iter1 << ") != *iter2 (" << *iter2 << ") at position " << (iter1 - vec1.begin()) << '!';
		}
	}
	if (are_same_candidates) {
		out << " same candidates ";
		if (is_same_size) {
			out << ":-)";
		} else {
			out << ":-|";
		}
	} else {
		out << " :-(";
	}
	return is_same_size && are_same_candidates;
}

bool compare_candidate_vectors(const std::vector<size_t> & vec1, const std::vector<size_t> & vec2) {
	return compare_candidate_vectors(vec1, vec2, std::clog);
}

void parse_file(MyMap & mymap, const char * const filename, std::ifstream & file) {
	std::string line;
	while (std::getline(file, line)) {
		bool do_skip_line = false;
		for (str_size_t i = 0; i < line.size(); ++i) {
			if (line[i] == ' ' || line[i] == '\t') {
			} else if (line[i] == '#') {
				do_skip_line = true;
			}
		}
		if (do_skip_line) {
			continue;
		}
		std::istringstream issLine (line);
		issLine >> std::skipws;
		data d;
		issLine >> d.n;
		check_istringstream(issLine, filename);
		issLine >> d.k;
		check_istringstream(issLine, filename);
		issLine >> d.a;
		check_istringstream(issLine, filename);
		issLine >> d.sdr;
		check_istringstream(issLine, filename);
		issLine >> d.numCandidates;
		check_istringstream(issLine, filename);
		std::clog << "parsing " << d << std::flush;

		std::string strCandidate;
		auto & vec = mymap[d];
		if (vec.size() != 0) {
			// comparison mode
			std::clog << " This combination is already present";
			std::vector<size_t> vec2;
			vec2.reserve(d.numCandidates);
			parse_candidates(issLine, strCandidate, d, vec2, filename);
			std::sort(vec2.begin(), vec2.end());
			compare_candidate_vectors(vec, vec2);
		} else {
			// insertion mode
			vec.reserve(d.numCandidates);
			parse_candidates(issLine, strCandidate, d, vec, filename);
			if (vec.size() != d.numCandidates) {
				std::clog << " vec.size(" << vec.size() << ") != d.numCandidates(" << d.numCandidates << ")! Clearing vector!";
				vec.clear();
				vec.shrink_to_fit();
			} else {
				std::sort(vec.begin(), vec.end());
			}
		}
		std::clog << " done" << std::endl;
	}
	if (file.bad()) {
		throw std::runtime_error("something bad happened. did you provide a directory as file?");
	}
}

int main(int argc, char ** argv) {
	if (argc < 3) {
		std::stringstream ss;
		ss << "usage: " << argv[0] << " <file1> <file2> [<files...>]";
		std::cerr << ss.str() << std::endl;
		return 1;
	}

	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "-?") == 0 || std::strcmp(argv[i], "--help") == 0) {
			std::cout << "usage: " << argv[0] << " <file1> <file2> [<files...>]" << std::endl;
			return 0; // only print usage
		}
	}

	const size_t num_files = argc - 1;
	std::vector<std::pair<char*, std::ifstream>> file_streams;
	file_streams.reserve(num_files);
	for (int i = 1; i < argc; ++i) {
		auto & ref = file_streams.emplace_back(argv[i], argv[i]);
		if (!ref.second) {
			std::stringstream ss;
			ss << "error: <file " << i << "> \"" << argv[i] << "\" does not exist!";
			std::cerr << ss.str() << std::endl;
			return 1;
		}
	}

	MyMap fullmap;
	std::vector<MyMap> mymaps;
	mymaps.reserve(num_files);
	for (auto & name_stream : file_streams) {
		parse_file(mymaps.emplace_back(), name_stream.first, name_stream.second);
		name_stream.second.close(); // release file descriptor and allow to release memory
	}
	for (size_t i = 0; i < num_files; ++i) {
		auto & map1 = mymaps[i];
		for (size_t j = i + 1; j < file_streams.size(); ++j) {
			auto & map2 = mymaps[j];
			for (auto & entry1 : map1) {
				auto iter = map2.find(entry1.first);
				if (iter != map2.end()) {
					std::clog << "Checking [" << entry1.first.n << '/' << entry1.first.k << '/' << entry1.first.a << "] ";
					bool is_same_data = entry1.first.compare_to(iter->first); // we do not use all members of type data for the hash!
					bool are_same_candidates = compare_candidate_vectors(entry1.second, iter->second);
					if (is_same_data && are_same_candidates) {
						auto iterfull = fullmap.find(entry1.first);
						if (iterfull == fullmap.end()) {
							fullmap.emplace(entry1);
						}
					} else {
						// check whether it was present in fullmap and if so, delete
						auto iterfull = fullmap.find(entry1.first);
						if (iterfull != fullmap.end()) {
							fullmap.erase(iterfull);
						}
					}
					std::clog << std::endl;
				} else {
					auto iterfull = fullmap.find(entry1.first);
					if (iterfull != fullmap.end()) {
						bool is_same_data = entry1.first.compare_to(iterfull->first);
						bool are_same_candidates = compare_candidate_vectors(entry1.second, iterfull->second);
						if (is_same_data && are_same_candidates) {
						} else {
							fullmap.erase(iterfull);
						}
					} else {
						// only emplace if not yet seen and no dissimilarities found
						fullmap.emplace(entry1);
					}
				}
				std::clog << std::flush;
			}
		}
	}
	std::clog << "fullmap has " << fullmap.size() << " entries." << std::endl;
	for (auto & entry : fullmap) {
		std::cout << data_file_out(entry.first) << '\t';
		auto iter = entry.second.begin();
		std::cout << *iter++;
		while (iter != entry.second.end()) {
			std::cout << ',' << *iter++;
		}
		std::cout << '\n';
	}
	std::cout << std::flush;

	return 0;
}
