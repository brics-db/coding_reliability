// Copyright 2017 Till Kolditz
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

/*
 * print_latex_code.cpp
 *
 *  Created on: 23.11.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#include <iostream>
#include <vector>

/*
 * The following nested vectors describe the counted weight distributions for XOR checksums
 * over increasing checksum widths (from 1 to 6 bits) and over increasing block sizes in
 * numbers of data words (from 1 to 8).
 */

std::vector<std::vector<size_t>> PascalTriangle = { {1}, {1, 1}, {1, 2, 1}, {1, 3, 3, 1}, {1, 4, 6, 4, 1}, {1, 5, 10, 10, 5, 1}, {1, 6, 15, 20, 15, 6, 1}, {1, 7, 21, 35, 35, 21, 7, 1}, {1, 8, 28, 56,
        70, 56, 28, 8, 1}};

std::vector<std::vector<std::vector<size_t>>> triangles = { { {1, 0, 1}, {1, 0, 3, 0}, {1, 0, 6, 0, 1}, {1, 0, 10, 0, 5, 0}, {1, 0, 15, 0, 15, 0, 1}, {1, 0, 21, 0, 35, 0, 7, 0}, {1, 0, 28, 0, 70, 0,
        28, 0, 1, }, {1, 0, 36, 0, 126, 0, 84, 0, 9, 0}}, { {1, 0, 2, 0, 1}, {1, 0, 6, 0, 9, 0, 0}, {1, 0, 12, 0, 38, 0, 12, 0, 1}, {1, 0, 20, 0, 110, 0, 100, 0, 25, 0, 0}, {1, 0, 30, 0, 255, 0, 452,
        0, 255, 0, 30, 0, 1}, {1, 0, 42, 0, 511, 0, 1484, 0, 1519, 0, 490, 0, 49, 0, 0}, {1, 0, 56, 0, 924, 0, 3976, 0, 6470, 0, 3976, 0, 924, 0, 56, 0, 1}, {1, 0, 72, 0, 1548, 0, 9240, 0, 21942, 0,
        21816, 0, 9324, 0, 1512, 0, 81, 0, 0}}, { {1, 0, 3, 0, 3, 0, 1}, {1, 0, 9, 0, 27, 0, 27, 0, 0, 0}, {1, 0, 18, 0, 111, 0, 252, 0, 111, 0, 18, 0, 1}, {1, 0, 30, 0, 315, 0, 1300, 0, 1575, 0, 750,
        0, 125, 0, 0, 0}, {1, 0, 45, 0, 720, 0, 4728, 0, 10890, 0, 10890, 0, 4728, 0, 720, 0, 45, 0, 1},
        {1, 0, 63, 0, 1428, 0, 13692, 0, 50862, 0, 87906, 0, 73892, 0, 28812, 0, 5145, 0, 343, 0, 0, 0}, {1, 0, 84, 0, 2562, 0, 33796, 0, 184047, 0, 489384, 0, 677404, 0, 489384, 0, 184047, 0, 33796,
                0, 2562, 0, 84, 0, 1}, {1, 0, 108, 0, 4266, 0, 74124, 0, 555687, 0, 2106648, 0, 4349484, 0, 5012280, 0, 3259359, 0, 1172988, 0, 221130, 0, 20412, 0, 729, 0, 0, 0}}, { {1, 0, 4, 0, 6,
        0, 4, 0, 1}, {1, 0, 12, 0, 54, 0, 108, 0, 81, 0, 0, 0, 0}, {1, 0, 24, 0, 220, 0, 936, 0, 1734, 0, 936, 0, 220, 0, 24, 0, 1}, {1, 0, 40, 0, 620, 0, 4600, 0, 16150, 0, 23000, 0, 15500, 0, 5000,
        0, 625, 0, 0, 0, 0}, {1, 0, 60, 0, 1410, 0, 16204, 0, 92655, 0, 245880, 0, 336156, 0, 245880, 0, 92655, 0, 16204, 0, 1410, 0, 60, 0, 1}, {1, 0, 84, 0, 2786, 0, 45892, 0, 388815, 0, 1645224, 0,
        3795932, 0, 5013288, 0, 3811759, 0, 1634052, 0, 388962, 0, 48020, 0, 2401, 0, 0, 0, 0}, {1, 0, 112, 0, 4984, 0, 111440, 0, 1312028, 0, 8080240, 0, 28212296, 0, 58900688, 0, 75191878, 0,
        58900688, 0, 28212296, 0, 8080240, 0, 1312028, 0, 111440, 0, 4984, 0, 112, 0, 1}, {1, 0, 144, 0, 8280, 0, 241392, 0, 3770748, 0, 31810320, 0, 156470184, 0, 474376176, 0, 913696038, 0,
        1134373680, 0, 913304808, 0, 474674256, 0, 156463164, 0, 31729968, 0, 3796632, 0, 244944, 0, 6561, 0, 0, 0, 0}}, { {1, 0, 5, 0, 10, 0, 10, 0, 5, 0, 1}, {1, 0, 15, 0, 90, 0, 270, 0, 405, 0,
        243, 0, 0, 0, 0, 0}, {1, 0, 30, 0, 365, 0, 2280, 0, 7570, 0, 12276, 0, 7570, 0, 2280, 0, 365, 0, 30, 0, 1}, {1, 0, 50, 0, 1025, 0, 11000, 0, 65250, 0, 207500, 0, 326250, 0, 275000, 0, 128125,
        0, 31250, 0, 3125, 0, 0, 0, 0, 0}, {1, 0, 75, 0, 2325, 0, 38255, 0, 356925, 0, 1880175, 0, 5430385, 0, 9069075, 0, 9069075, 0, 5430385, 0, 1880175, 0, 356925, 0, 38255, 0, 2325, 0, 75, 0, 1},
        {1, 0, 105, 0, 4585, 0, 107345, 0, 1450645, 0, 11436061, 0, 52275405, 0, 145032405, 0, 253464995, 0, 283717595, 0, 203208635, 0, 92090355, 0, 26062855, 0, 4453855, 0, 420175, 0, 16807, 0, 0,
                0, 0, 0}, {1, 0, 140, 0, 8190, 0, 258860, 0, 4784365, 0, 52757488, 0, 349426280, 0, 1451310000, 0, 3926830610, 0, 7085345960, 0, 8618294580, 0, 7085345960, 0, 3926830610, 0,
                1451310000, 0, 349426280, 0, 52757488, 0, 4784365, 0, 258860, 0, 8190, 0, 140, 0, 1}, {1, 0, 180, 0, 13590, 0, 557700, 0, 13516245, 0, 198669456, 0, 1797107400, 0, 10434318480, 0,
                40412485170, 0, 107228617560, 0, 198132288516, 0, 257304583800, 0, 235831795650, 0, 152400327120, 0, 68952814920, 0, 21549873744, 0, 4560685965, 0, 635585940, 0, 55571670, 0, 2755620,
                0, 59049, 0, 0, 0, 0, 0}}, { {1, 0, 6, 0, 15, 0, 20, 0, 15, 0, 6, 0, 1}, {1, 0, 18, 0, 135, 0, 540, 0, 1215, 0, 1458, 0, 729, 0, 0, 0, 0, 0, 0}, {1, 0, 36, 0, 546, 0, 4500, 0, 21615,
        0, 59976, 0, 88796, 0, 59976, 0, 21615, 0, 4500, 0, 546, 0, 36, 0, 1}, {1, 0, 60, 0, 1530, 0, 21500, 0, 180375, 0, 915000, 0, 2727500, 0, 4575000, 0, 4509375, 0, 2687500, 0, 956250, 0, 187500,
        0, 15625, 0, 0, 0, 0, 0, 0}, {1, 0, 90, 0, 3465, 0, 74256, 0, 965700, 0, 7810200, 0, 39025140, 0, 119084400, 0, 228441150, 0, 282933020, 0, 228441150, 0, 119084400, 0, 39025140, 0, 7810200, 0,
        965700, 0, 74256, 0, 3465, 0, 90, 0, 1}, {1, 0, 126, 0, 6825, 0, 207312, 0, 3866100, 0, 45688776, 0, 343956676, 0, 1653232560, 0, 5208837102, 0, 11048544500, 0, 16047779790, 0, 16063842480, 0,
        11058285700, 0, 5197396680, 0, 1650783540, 0, 347165392, 0, 46236057, 0, 3529470, 0, 117649, 0, 0, 0, 0, 0, 0}, {1, 0, 168, 0, 12180, 0, 498008, 0, 12609666, 0, 205069368, 0, 2168797764, 0,
        15062491080, 0, 70505344239, 0, 228464996368, 0, 522872230440, 0, 855774378480, 0, 1007913655580, 0, 855774378480, 0, 522872230440, 0, 228464996368, 0, 70505344239, 0, 15062491080, 0,
        2168797764, 0, 205069368, 0, 12609666, 0, 498008, 0, 12180, 0, 168, 0, 1}, {1, 0, 216, 0, 20196, 0, 1069704, 0, 35320914, 0, 756667656, 0, 10699223796, 0, 101302920216, 0, 659293363359, 0,
        3029547258864, 0, 10042992371016, 0, 24389430403536, 0, 43834381407036, 0, 58670892322704, 0, 58666946478984, 0, 43831904516784, 0, 24392524459599, 0, 10043743769784, 0, 3028347825876, 0,
        659133648936, 0, 101536684434, 0, 10737627624, 0, 739057284, 0, 29760696, 0, 531441, 0, 0, 0, 0, 0, 0}}};

void print_triangle_odd_increase(
        const std::vector<std::vector<size_t>> & triangle,
        const size_t num_layers,
        const size_t first_layer_num,
        const size_t increase) {
    // for odd increases we need staggered alignment
    // 1) print all column specifiers
    const size_t max = 2 * triangle[num_layers - 1].size() + 2;
    for (size_t i = 0; i < max; ++i) {
        std::cout << " c";
    }
    std::cout << "}\n";

    // 2) force column widths
    std::cout << "        ~";
    for (size_t i = 0; i < max; ++i) {
        std::cout << " & ~";
    }
    std::cout << " \\\\\n";

    // 3) print header lines
    std::cout << "        \\toprule\n";
    const size_t columns_spanned_for_diagonal_headings = (num_layers - 1) * increase + 1;
    // 3.A) first header line with the top most diagonal headings
    std::cout << "        & \\multicolumn{" << (1 + columns_spanned_for_diagonal_headings) << "}{l}{\\(\\llbracket\\mathbb{C}\\rrbracket\\)}";
    for (size_t i = 0; i < triangle[0].size(); ++i) {
        std::cout << " & \\multicolumn{2}{c}{" << i << "}";
    }
    std::cout << " \\\\\n";
    // 3.B) second header line with the arrows and a diagonal heading at the end
    std::cout << "        \\(\\#\\mathbb{D}\\) & \\multicolumn{" << columns_spanned_for_diagonal_headings << "}{l}{}";
    for (size_t i = 0; i < triangle[0].size(); ++i) {
        std::cout << " & \\multicolumn{2}{c}{\\(\\swarrow\\)}";
    }
    std::cout << " & \\multicolumn{2}{c}{\\(" << triangle[0].size() << "\\)} \\\\\n";
    // 3.C) a rule up to the last arrow
    const size_t last_column_for_long_rule = columns_spanned_for_diagonal_headings + triangle[0].size() * 2;
    std::cout << "        \\cmidrule{1-" << last_column_for_long_rule << "}\n";
    // 4) the content lines
    size_t num_layer = 0;
    size_t diagonal_counter = triangle[0].size();
    for (const auto & layer : triangle) {
        ++num_layer;
        std::cout << "        " << (num_layer - 1 + first_layer_num);
        const size_t num_columns_skipped = (num_layers - num_layer) * increase;
        if (num_columns_skipped > 5) {
            std::cout << " & \\multicolumn{" << num_columns_skipped << "}{l}{}";
        } else {
            for (size_t i = 0; i < num_columns_skipped; ++i) {
                std::cout << " &";
            }
        }
        for (const auto weight : layer) {
            std::cout << " & \\multicolumn{2}{c}{" << weight << "}";
        }
        if (num_layer < num_layers) {
            std::cout << " & \\multicolumn{2}{|c}{\\(\\swarrow\\)}";
            if (num_layer < (num_layers - 1)) {
                ++diagonal_counter;
                if (increase - 1) {
                    std::cout << " & \\multicolumn{" << (increase - 1) << "}{c}{}";
                }
                std::cout << " & \\multicolumn{2}{c}{\\(" << diagonal_counter << "\\)}";
            }
        }
        const size_t column_rule = last_column_for_long_rule + (num_layer - 1) * increase + 1;
        std::cout << " \\\\\n";
        if (num_layer < num_layers) {
            std::cout << "        \\cmidrule{" << column_rule << "-" << (column_rule + increase - 1) << "}\n";
        }
    }
}

void print_triangle_even_increase(
        const std::vector<std::vector<size_t>> & triangle,
        const size_t num_layers,
        const size_t first_layer_num,
        const size_t increase) {
    // for even increase, the numbers will be below each other
    // 1) print all column specifiers
    const size_t max = triangle[num_layers - 1].size() + 2;
    for (size_t i = 0; i < max; ++i) {
        std::cout << " c";
    }
    std::cout << "}\n";

    // 2) force column widths
    std::cout << "        ~";
    for (size_t i = 0; i < max; ++i) {
        std::cout << " & ~";
    }
    std::cout << " \\\\\n";

    // 3) print header lines
    std::cout << "        \\toprule\n";
    const size_t columns_spanned_for_diagonal_headings = (num_layers - 1) * increase + 1;
    // 3.A) first header line with the top most diagonal headings
    std::cout << "        & \\multicolumn{" << (1 + columns_spanned_for_diagonal_headings) << "}{l}{\\(\\llbracket\\mathbb{C}\\rrbracket\\)}";
    for (size_t i = 0; i < triangle[0].size(); ++i) {
        std::cout << " & \\multicolumn{2}{c}{" << i << "}";
    }
    std::cout << " \\\\\n";
    // 3.B) second header line with the arrows and a diagonal heading at the end
    std::cout << "        \\(\\#\\mathbb{D}\\) & \\multicolumn{" << columns_spanned_for_diagonal_headings << "}{l}{}";
    for (size_t i = 0; i < triangle[0].size(); ++i) {
        std::cout << " & \\multicolumn{2}{c}{\\(\\swarrow\\)}";
    }
    std::cout << " & \\multicolumn{2}{c}{\\(" << triangle[0].size() << "\\)} \\\\\n";
    // 3.C) a rule up to the last arrow
    const size_t last_column_for_long_rule = columns_spanned_for_diagonal_headings + triangle[0].size() * 2;
    std::cout << "        \\cmidrule{1-" << last_column_for_long_rule << "}\n";
    // 4) the content lines
    size_t num_layer = 0;
    size_t diagonal_counter = triangle[0].size();
    for (const auto & layer : triangle) {
        ++num_layer;
        std::cout << "        " << (num_layer - 1 + first_layer_num);
        const size_t num_columns_skipped = (num_layers - num_layer) * increase;
        if (num_columns_skipped > 5) {
            std::cout << " & \\multicolumn{" << num_columns_skipped << "}{l}{}";
        } else {
            for (size_t i = 0; i < num_columns_skipped; ++i) {
                std::cout << " &";
            }
        }
        for (const auto weight : layer) {
            std::cout << " & \\multicolumn{2}{c}{" << weight << "}";
        }
        if (num_layer < num_layers) {
            std::cout << " & \\multicolumn{2}{|c}{\\(\\swarrow\\)}";
            if (num_layer < (num_layers - 1)) {
                ++diagonal_counter;
                if (increase - 1) {
                    std::cout << " & \\multicolumn{" << (increase - 1) << "}{c}{}";
                }
                std::cout << " & \\multicolumn{2}{c}{\\(" << diagonal_counter << "\\)}";
            }
        }
        const size_t column_rule = last_column_for_long_rule + (num_layer - 1) * increase + 1;
        std::cout << " \\\\\n";
        if (num_layer < num_layers) {
            std::cout << "        \\cmidrule{" << column_rule << "-" << (column_rule + increase - 1) << "}\n";
        }
    }
}

void print_header() {
    std::cout << "\\begin{table}%\n"
            "    \\centering\n"
            "    \\footnotesize\n"
            "    \\begin{tabular}{c";
}

void print_footer(
        const size_t size_checksum) {
    std::cout << "        \\bottomrule\n"
            "    \\end{tabular}\n"
            "    \\caption{Weight distribution triangle for " << size_checksum << "-bit checksums.}\n"
            "    \\label{tab:XOR:WeightDistribution:triangle:" << size_checksum << "-bit}\n"
            "\\end{table}\n\n\n\n\n\n";
    // "        ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~\n"
}

void print_footer(
        const char * const str_caption,
        const char * const str_label) {
    std::cout << "        \\bottomrule\n"
            "    \\end{tabular}\n"
            "    \\caption{" << str_caption << "}\n"
            "    \\label{" << str_label << "}\n"
            "\\end{table}\n\n\n\n\n\n";
    // "        ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~ & ~\n"
}

int main() {
    print_header();
    print_triangle_odd_increase(PascalTriangle, PascalTriangle.size(), 0, 1);
    print_footer("The first 9 rows of the Pascal Triangle.", "tab:PascalTriangle:triangle");

    size_t size_checksum = 0;
    for (const auto & triangle : triangles) {
        ++size_checksum;

        print_header();

        const size_t num_layers = triangle.size();
        const size_t increase = triangle[1].size() - triangle[0].size();
        if (increase & 0x1) {
            print_triangle_odd_increase(triangle, num_layers, 1, increase);
        } else {
            print_triangle_even_increase(triangle, num_layers, 1, increase);
        }

        print_footer(size_checksum);
    }
    return 0;
}
