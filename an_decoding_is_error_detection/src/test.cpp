#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>

int main()
{
    const size_t W = 7;
    const size_t Cwidth = 8;
    const size_t A = 3;
    const size_t mask = (1ll << Cwidth) - 1;
    ssize_t c = ~((1ll << (Cwidth - 1)) - 1);
    ssize_t firstCW = c + (c % A);
    std::cout << std::setw(W) << "c" << " = " << c << "\n";
    std::cout << std::setw(W) << "c" << " = 0x" << std::hex << (mask & c) << "\n";
    std::cout << std::setw(W) << "A" << " = " << std::dec << A << "\n";
    std::cout << std::setw(W) << "c%A" << " = " << (c % A) << "\n";
    std::cout << std::setw(W) << "c%A" << " = 0x" << std::hex << (mask & (c % A)) << "\n";
    std::cout << std::setw(W) << "c+(c%A)" << " = " << std::dec << firstCW << "\n";
    std::cout << std::setw(W) << "c+(c%A)" << " = 0x" << std::hex << firstCW << "\n";
    std::cout << std::setw(W) << "fCW%A" << " = " << std::dec << (firstCW % A) << "\n";

    std::cout << std::setw(W) << "tmp" << " = " << std::dec;
    for (ssize_t i = 0, tmp = (c + 31); i < 32; ++i, --tmp) {
        std::cout << (((tmp % A) == 1) ? '0' : '1');
    }
    std::cout << std::endl;

    uint32_t mmIsNotACodeword = 0;
    for (ssize_t i = 0, tmp = c; i < 32; ++i, ++tmp) {
        if ((tmp % A) != 1) {
            mmIsNotACodeword |= (1ll << i);
        }
    }
    std::cout << std::setw(W) << "mmINAC" << " = 0x" << std::hex << mmIsNotACodeword << "\n";
}

