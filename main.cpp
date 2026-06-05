#include <exception>
#include <iostream>

void run_benchmark_cli(int argc, char** argv);

int main(int argc, char** argv) {
    try {
        run_benchmark_cli(argc, argv);

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
