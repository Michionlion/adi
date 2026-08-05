#include "adi/adi.hpp"

#include <iostream>
#include <string_view>

int main(int argc, char ** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "adi " << adi::version() << '\n';
        return 0;
    }

    std::cerr << "usage: adi --version\n";
    return 2;
}
