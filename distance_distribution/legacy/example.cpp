#include <iostream>
#include <iomanip>
#include <cstdint>
#include <bitset>
#include <limits>
#include <vector>
#include <utility>
#include <algorithm>

template<typename T, T A, T n, T k>
int print_matrix() {
	constexpr const T d_max = static_cast<T>((1ull << k) - 1ull);
	for (T d = 0; d <= d_max; ++d) {
		std::cout << d << '\t';
	}
	std::cout << '\n';
	for (T d = 0; d <= d_max; ++d) {
		std::cout << (d * A) << '\t';
	}
	std::cout << '\n';
	for (T d = 0; d <= d_max; ++d) {
		std::bitset<n> b(d * A);
		std::cout << b << '\t';
	}
	std::cout << std::endl;

	std::cout << "---";
	for (T d = 0; d <= d_max; ++d) {
		std::cout << '\t' << (d * A);
	}
	std::cout << '\n';
	T dHmin = std::numeric_limits<T>::max();
	std::vector<std::pair<T, T>> pairs;
	uint16_t histogram[n + 1]{};
	for (T d1 = 0; d1 <= d_max; ++d1) {
		std::cout << (d1 * A);
		for (T d2 = 0; d2 <= d_max; ++d2) {
			const T dH = __builtin_popcountll((d1 * A) ^ (d2 * A));
			histogram[dH]++;
			if (dH > 0) {
				if (dH < dHmin) {
					pairs.clear();
					dHmin = dH;
				}
				if (dH == dHmin) {
					auto iter = std::find(pairs.begin(), pairs.end(), std::make_pair<T, T>(d2*A, d1*A));
					if (iter == pairs.end()) {
						pairs.emplace_back(std::make_pair<T, T>(d1*A, d2*A));
					}
				}
			}
			std::cout << '\t' << dH;
		}
		std::cout << '\n';
	}
	std::cout << "dHmin = " << dHmin << '\n';
	std::cout << "limiting pairs are: (" << pairs.size() << ')';
	std::for_each(pairs.begin(), pairs.end(), [] (auto x) {std::cout << " [" << x.first << ", " << x.second << ']';});
	std::cout << "\nhistogram:\n";
	T total = 0;
	for (int i = 0; i <= n; ++i) {
		std::cout << std::setw(3) << i << ": " << histogram[i] << '\n';
		total += histogram[i];
	}
	std::cout << "total edges: " << total << std::endl;
}

int main() {
	// concrete example with A=19, |A|=5, |D|=k=5, |C|=10
	print_matrix<uint16_t, 19, 10, 5>();
	std::cout << "\n\n";
	// concrete example with A=53, |A|=6, |D|=k=5, |C|=10
	print_matrix<uint16_t, 53, 11, 5>();
}

