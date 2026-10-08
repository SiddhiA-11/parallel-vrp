#include "vrp_hybrid.h"
#include <iostream>

int main(int argc, char* argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np <processes> " << argv[0] << " <filepath> [--threads N] [--json|--vrplib]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string filepath = argv[1];
    int threads_per_proc = 2;
    bool is_json = (filepath.find(".json") != std::string::npos);

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads_per_proc = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            is_json = true;
        } else if (arg == "--vrplib") {
            is_json = false;
        }
    }

    try {
        CVRPInstance inst;
        if (rank == 0) {
            if (is_json) {
                inst = parse_json_file(filepath);
            } else {
                inst = parse_vrplib_file(filepath);
            }
        }

        HybridSolution sol = solve_cvrp_hybrid(inst, rank, size, threads_per_proc);

        if (rank == 0) {
            print_hybrid_solution_json(sol, std::cout);
        }

        MPI_Finalize();
        return (rank == 0 && !sol.valid) ? 2 : 0;
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Hybrid Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
}
