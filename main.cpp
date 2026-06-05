#include <exception>
#include <iostream>

void run_benchmark_demo();

int main() {
    try {
        run_benchmark_demo();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
