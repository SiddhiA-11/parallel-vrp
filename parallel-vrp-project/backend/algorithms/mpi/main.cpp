#include "vrp_mpi.h"
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np <processes> " << argv[0] << " <filepath> [--json|--vrplib]" << std::endl;
        }
        MPI_Finalize();
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
        if (rank == 0) {
            if (is_json) {
                inst = parse_json_file(filepath);
            } else {
                inst = parse_vrplib_file(filepath);
            }
        }

        MPISolution sol = solve_cvrp_mpi(inst, rank, size);

        if (rank == 0) {
            print_mpi_solution_json(sol, std::cout);
        }

        MPI_Finalize();
        return (rank == 0 && !sol.valid) ? 2 : 0;
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "MPI Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
}
