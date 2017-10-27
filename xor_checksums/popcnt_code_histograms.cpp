#include <iostream>
#include <sstream>
#include <cstdlib>
#include <omp.h>

uint64_t factorial(
        uint64_t n) {
    uint64_t result = 1;
    for (; n; --n)
        result *= n;
    return result;
}

uint64_t binomial_coefficient(
        uint64_t n,
        uint64_t k) {
    uint64_t result = 1;
    if (n < k)
        throw std::runtime_error("n < k");
    if (k > 1) {
        for (uint64_t i = 1; i <= k; ++i) {
            result = result * (n + 1 - i) / i;
        }
    }
    return result;
}

void compute_xor_weight_distribution(
        const uint64_t num_databits,
        const uint64_t num_datawords) {
    const uint64_t num_databits_total = num_databits * num_datawords;
    const uint64_t mask = (0x1ull << num_databits) - 1ull;

    const uint64_t num_histogram_entries = num_databits_total + num_databits + 1;
    uint64_t * histogram = new uint64_t[num_histogram_entries];
    for (uint64_t i = 0ull; i < num_histogram_entries; ++i) {
        histogram[i] = 0ull; // just to be sure
    }

    int num_threads = 0;
    uint64_t ** local_histograms = nullptr;
#pragma omp parallel
    {
#pragma omp master
        {
            num_threads = omp_get_num_threads();
            local_histograms = new uint64_t*[num_threads];
            const uint64_t size_num_histogram_entries_bytes = num_histogram_entries * sizeof(uint64_t);
            const uint64_t aligned_num = (size_num_histogram_entries_bytes + 63 /*cachelinewidth*/) & (~63);
            for (int i = 0; i < num_threads; ++i) {
                local_histograms[i] = new uint64_t[aligned_num];
                for (uint64_t k = 0; k < num_histogram_entries; ++k) {
                    local_histograms[i][k] = 0;
                }
            }
        }
    }

    uint64_t num_codewords = 0x1ull << num_databits_total; // 2 ^ (num_datawords * num_databits)
    const uint64_t max_dataword = num_codewords - 1; // 1-bit case only !!!
    const bool do_use_openmp = num_databits_total > 24ull;
    const uint64_t datawords_per_thread = do_use_openmp ? (num_codewords / static_cast<uint64_t>(num_threads)) : num_codewords;
    // std::cout << "#datawords=" << num_datawords << '\n' << "#codewords=" << num_codewords << '\n' << "#threads=" << num_threads << '\n' << "#datawords per thread=" << datawords_per_thread << std::endl;

#pragma omp parallel
    {
        const int thread_num = omp_get_thread_num();
        uint64_t * local_histogram = local_histograms[thread_num];
        const uint64_t min = datawords_per_thread * thread_num;
        const uint64_t max = thread_num == (num_threads - 1) ? max_dataword : (min + (datawords_per_thread - 1));
        if (do_use_openmp || thread_num == 0) {
            // std::stringstream ss;
            // ss << thread_num << ": min=" << min << " max=" << max << std::endl;
            // std::cout << ss.str();
            switch (num_databits) {
                case 1:
                    for (uint64_t dataword = min; dataword <= max; ++dataword) {
                        const uint64_t popcnt_data = __builtin_popcountll(dataword);
                        const uint64_t popcnt_code = popcnt_data + (popcnt_data & mask);
                        ++local_histogram[popcnt_code];
                    }
                    break;

                default:
                    for (uint64_t dataword = min; dataword <= max; ++dataword) {
                        const uint64_t popcnt_data = __builtin_popcountll(dataword);
                        uint64_t checksum = dataword & mask;
                        uint64_t temp_dataword = dataword >> num_databits;
                        for (uint64_t x = 1; x < num_datawords; ++x, temp_dataword >>= num_databits) {
                            checksum ^= temp_dataword & mask;
                        }
                        const uint64_t popcnt_code = popcnt_data + __builtin_popcountll(checksum);
                        ++local_histogram[popcnt_code];
                    }
            }
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        for (uint64_t k = 0; k < num_histogram_entries; ++k) {
            histogram[k] += local_histograms[i][k];
        }
        delete[] local_histograms[i];
    }
    delete[] local_histograms;

    std::cout << num_datawords;
    for (uint64_t i = 0; i < num_histogram_entries; ++i) {
        std::cout << ';' << histogram[i];
    }
    std::cout << std::endl;
    delete[] histogram;
}

int main() {
    const uint64_t max_num_datawords = 8;
    const uint64_t max_num_databits = 8;
    for (uint64_t num_databits = 1; num_databits <= max_num_databits; ++num_databits) {
        const uint64_t num_checksumbits = num_databits;
        const uint64_t max_num_codebits = max_num_datawords * num_databits + num_checksumbits;
        std::cout << "|D|=" << num_databits << "\n#D\\|C|";
        for (uint64_t i = 0; i <= max_num_codebits; ++i) {
            std::cout << ';' << i;
        }
        std::cout << std::endl;
        for (uint64_t num_datawords = 1; num_datawords <= max_num_datawords; ++num_datawords) {
            if (num_databits * num_datawords <= 64) {
                compute_xor_weight_distribution(num_databits, num_datawords);
            } else {
                std::cout << "Case " << num_databits << '/' << num_datawords << " not implemented yet!" << std::endl;
            }
        }
        if (num_databits < max_num_databits) {
            std::cout << std::endl;
        }
    }
    return 0;
}

