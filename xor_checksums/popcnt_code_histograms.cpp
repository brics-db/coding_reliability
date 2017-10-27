#include <iostream>
#include <sstream>
#include <omp.h>

size_t factorial(
        size_t n) {
    size_t result = 1;
    for (; n; --n)
        result *= n;
    return result;
}

size_t binomial_coefficient(
        size_t n,
        size_t k) {
    size_t result = 1;
    if (n < k)
        throw std::runtime_error("n < k");
    if (k > 1) {
        for (size_t i = 1; i <= k; ++i) {
            result = result * (n + 1 - i) / i;
        }
    }
    return result;
}

void compute_xor_weight_distribution(
        const size_t num_databits,
        const size_t num_datawords) {
    const size_t num_databits_total = num_databits * num_datawords;
    const size_t mask = (0x1 << num_databits) - 1;

    const size_t num_histogram_entries = num_databits * num_datawords + num_databits + 1;
    size_t * histogram = new size_t[num_histogram_entries];
    for (size_t i = 0; i < num_histogram_entries; ++i) {
        histogram[i] = 0; // just to be sure
    }

    size_t num_threads = 0;
    size_t ** local_histograms = nullptr;
#pragma omp parallel
    {
#pragma omp master
        {
            num_threads = omp_get_num_threads();
            local_histograms = new size_t*[num_threads];
            const size_t size_num_histogram_entries_bytes = num_histogram_entries * sizeof(size_t);
            const size_t aligned_num = (size_num_histogram_entries_bytes + 63 /*cachelinewidth*/) & (~63);
            for (size_t i = 0; i < num_threads; ++i) {
                local_histograms[i] = new size_t[aligned_num];
                for (size_t k = 0; k < num_histogram_entries; ++k) {
                    local_histograms[i][k] = 0;
                }
            }
        }
    }

    size_t num_codewords = 0x1 << (num_datawords * num_databits); // 2 ^ (num_datawords * num_databits)
    const size_t max_dataword = num_codewords - 1; // 1-bit case only !!!
    const bool do_use_openmp = (num_datawords * num_databits) > 24;
    const size_t datawords_per_thread = do_use_openmp ? (num_codewords / num_threads) : num_codewords;
    // std::cout << "#datawords=" << num_datawords << '\n' << "#codewords=" << num_codewords << '\n' << "#threads=" << num_threads << '\n' << "#datawords per thread=" << datawords_per_thread << std::endl;

#pragma omp parallel
    {
        const int thread_num = omp_get_thread_num();
        size_t * local_histogram = local_histograms[thread_num];
        const size_t min = datawords_per_thread * thread_num;
        const size_t max = thread_num == (num_threads - 1) ? max_dataword : (min + (datawords_per_thread - 1));
        if (do_use_openmp || thread_num == 0) {
            // std::stringstream ss;
            // ss << thread_num << ": min=" << min << " max=" << max << std::endl;
            // std::cout << ss.str();
            switch (num_databits) {
                case 1:
                    for (size_t dataword = min; dataword <= max; ++dataword) {
                        const size_t popcnt_data = __builtin_popcountll(dataword);
                        const size_t popcnt_code = popcnt_data + (popcnt_data & mask);
                        ++local_histogram[popcnt_code];
                    }
                    break;

                default:
                    for (size_t dataword = min; dataword <= max; ++dataword) {
                        const size_t popcnt_data = __builtin_popcountll(dataword);
                        size_t checksum = dataword & mask;
                        size_t temp_dataword = dataword >> num_databits;
                        for (size_t x = 1; x < num_datawords; ++x, temp_dataword >>= num_databits) {
                            checksum ^= temp_dataword & mask;
                        }
                        const size_t popcnt_code = popcnt_data + __builtin_popcountll(checksum);
                        ++local_histogram[popcnt_code];
                    }
            }
        }
    }

    for (size_t i = 0; i < num_threads; ++i) {
        for (size_t k = 0; k < num_histogram_entries; ++k) {
            histogram[k] += local_histograms[i][k];
        }
        delete[] local_histograms[i];
    }
    delete[] local_histograms;

    std::cout << num_datawords;
    for (size_t i = 0; i < num_histogram_entries; ++i) {
        std::cout << ';' << histogram[i];
    }
    std::cout << std::endl;
    delete[] histogram;
}

int main() {
    const size_t max_num_datawords = 8;
    const size_t max_num_databits = 8;
    for (size_t num_databits = 1; num_databits <= max_num_databits; ++num_databits) {
        const size_t num_checksumbits = num_databits;
        const size_t max_num_codebits = max_num_datawords * num_databits + num_checksumbits;
        std::cout << "|D|=" << num_databits << "\n#D\\|C|";
        for (size_t i = 0; i <= max_num_codebits; ++i) {
            std::cout << ';' << i;
        }
        std::cout << std::endl;
        for (size_t num_datawords = 1; num_datawords <= max_num_datawords; ++num_datawords) {
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

