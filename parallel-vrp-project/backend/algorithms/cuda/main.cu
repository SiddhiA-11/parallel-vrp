#include "vrp_cuda.cuh"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filepath> [--block-size 64|128|256|512] [--json|--vrplib]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    int block_size = 256;
    bool is_json = (filepath.find(".json") != std::string::npos);

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--block-size" && i + 1 < argc) {
            block_size = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            is_json = true;
        } else if (arg == "--vrplib") {
            is_json = false;
        }
    }

    try {
        CVRPInstance inst;
        if (is_json) {
            inst = parse_json_file(filepath);
        } else {
            inst = parse_vrplib_file(filepath);
        }

        CUDASolution sol = solve_cvrp_cuda(inst, block_size);
        print_cuda_solution_json(sol, std::cout);

        return sol.valid ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    }
}
