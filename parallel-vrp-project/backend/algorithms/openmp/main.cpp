#include "vrp_openmp.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filepath> [--threads N] [--json|--vrplib]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    int threads = 0;
    bool is_json = (filepath.find(".json") != std::string::npos);

    // Read from env var if present
    const char* env_threads = std::getenv("OMP_NUM_THREADS");
    if (env_threads) {
        try { threads = std::stoi(env_threads); } catch (...) {}
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            is_json = true;
        } else if (arg == "--vrplib") {
            is_json = false;
        }
    }

    if (threads <= 0) {
        threads = omp_get_max_threads();
    }

    try {
        CVRPInstance inst;
        if (is_json) {
            inst = parse_json_file(filepath);
        } else {
            inst = parse_vrplib_file(filepath);
        }

        OpenMPSolution sol = solve_cvrp_openmp(inst, threads);
        print_openmp_solution_json(sol, std::cout);

        return sol.valid ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
