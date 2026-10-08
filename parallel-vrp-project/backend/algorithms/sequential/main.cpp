#include "vrp.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filepath> [--vrplib|--json]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    bool is_json = (filepath.find(".json") != std::string::npos);
    if (argc >= 3) {
        std::string flag = argv[2];
        if (flag == "--json") is_json = true;
        if (flag == "--vrplib") is_json = false;
    }

    try {
        CVRPInstance inst;
        if (is_json) {
            inst = parse_json_file(filepath);
        } else {
            inst = parse_vrplib_file(filepath);
        }

        Solution sol = solve_cvrp_sequential(inst);
        print_solution_json(sol, std::cout);

        return sol.valid ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
