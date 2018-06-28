#include <iostream>
#include <iomanip>
#include <cstdint>
#include <bitset>
#include <limits>
#include <vector>
#include <utility>
#include <algorithm>
#include <sstream>

#define THROW_ERROR(MESSAGE, LINE) {                                                \
    std::string filename(__FILE__);                                                 \
    auto filesub = filename.substr(filename.rfind('/') + 1);                        \
    std::stringstream ss;                                                           \
    ss << "[ERROR @ " << filesub << ':' << LINE << "] " << MESSAGE << std::endl;    \
    throw std::runtime_error(ss.str());                                             \
}

template<typename T>
void print_matrix_static(
        T A,
        T n,
        T k);

typedef size_t bitwidth_t;

void print_matrix_dynamic(
        bitwidth_t dataWidth,
        size_t A,
        size_t n,
        size_t k);

int main(
        int argc,
        char** argv) {
    if (argc == 1) {
        // concrete example with A=19, |A|=5, |D|=k=5, |C|=10
        print_matrix_static<uint16_t>(19, 10, 5);
        std::cout << "\n\n";
        // concrete example with A=53, |A|=6, |D|=k=5, |C|=10
        print_matrix_static<uint16_t>(53, 11, 5);
    } else {
        if (argc == 3) {
            size_t T, A, n, k;
            std::stringstream ss;
            ss << argv[1] << ' ' << argv[2];
            ss >> A >> k;
            size_t bwA = (8 * sizeof(A) - __builtin_clzll(A));
            n = bwA + k;
            T = n <= 8 ? 8 : (n <= 16 ? 16 : (n <= 32 ? 32 : 64));
            print_matrix_dynamic(T, A, n, k);
        } else {
            THROW_ERROR("Usage: [<A> <k>].\n\tA: the code parameter.\n\tk: raw data width in bits.\n\tImplicit (computed) parameters:\n\tn: total code width in bits = |A| + k.", __LINE__)
        }
    }
    return 0;
}

void print_codeword_binary(
        size_t n,
        size_t d,
        size_t A) {
    if (n <= 64) {
        size_t cw = d * A;
        for (size_t i = 0, mask = 1; i < n; ++i) {
            std::cout << ((cw & mask) != 0);
            mask <<= 1;
        }
        std::cout << '\t';
    } else {
        THROW_ERROR("Unsupported data width " << n, __LINE__)
    }
}

template<typename T>
void print_matrix_static(
        T A,
        T n,
        T k) {
    std::cout << "Settings: A=" << A << ". n=" << n << ". k=|A|=" << k << std::endl;
    const T d_max = static_cast<T>((1ull << k) - 1ull);
    for (T d = 0; d <= d_max; ++d) {
        std::cout << d << '\t';
    }
    std::cout << '\n';
    for (T d = 0; d <= d_max; ++d) {
        std::cout << (d * A) << '\t';
    }
    std::cout << '\n';
    for (T d = 0; d <= d_max; ++d) {
        print_codeword_binary(n, d, A);
    }
    std::cout << std::endl;

    std::cout << "---";
    for (T d = 0; d <= d_max; ++d) {
        std::cout << '\t' << (d * A);
    }
    std::cout << '\n';
    T dHmin = std::numeric_limits<T>::max();
    std::vector<std::pair<T, T>> pairs;
    std::unique_ptr<uint16_t[]> histogram(new uint16_t[n + 1]);
    // uint16_t histogram[n + 1] {};
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
                    auto iter = std::find(pairs.begin(), pairs.end(), std::make_pair<T, T>(d2 * A, d1 * A));
                    if (iter == pairs.end()) {
                        pairs.emplace_back(std::make_pair<T, T>(d1 * A, d2 * A));
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
    for (size_t i = 0; i <= n; ++i) {
        std::cout << std::setw(3) << i << ": " << histogram[i] << '\n';
        total += histogram[i];
    }
    std::cout << "total edges: " << total << std::endl;
}

void print_matrix_dynamic(
        bitwidth_t dataWidth,
        size_t A,
        size_t n,
        size_t k) {
//    if (__builtin_popcountll(A) > dataWidth) {
//        THROW_ERROR("|A| should be < " << dataWidth << ", but is " << __builtin_popcountll(A), __LINE__)
//    } else if (n > dataWidth) {
//        THROW_ERROR("n should be < " << dataWidth << ", but is " << n, __LINE__)
//    } else if (k > dataWidth) {
//        THROW_ERROR("n should be < " << dataWidth << ", but is " << n, __LINE__)
//    } else if ((__builtin_popcountll(A) + k) > dataWidth) {
//        THROW_ERROR("(|A| + k) should be <= " << dataWidth << ", but is " << (__builtin_popcountll(A) + k), __LINE__)
//    }
    if (dataWidth <= 16) {
        print_matrix_static<uint16_t>(static_cast<uint16_t>(A), static_cast<uint16_t>(n), static_cast<uint16_t>(k));
    } else if (dataWidth <= 32) {
        print_matrix_static<uint32_t>(static_cast<uint32_t>(A), static_cast<uint32_t>(n), static_cast<uint32_t>(k));
    } else if (dataWidth <= 64) {
        print_matrix_static<uint64_t>(static_cast<uint64_t>(A), static_cast<uint64_t>(n), static_cast<uint64_t>(k));
    }
}
