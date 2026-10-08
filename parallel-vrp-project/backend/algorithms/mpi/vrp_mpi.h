#ifndef VRP_MPI_H
#define VRP_MPI_H

#include "../sequential/vrp.h"
#include <mpi.h>

struct MPISolution : public Solution {
    int mpi_processes = 1;
    double computation_time_ms = 0.0;
    double communication_time_ms = 0.0;
};

void run_mpi_cvrp(int argc, char* argv[]);
MPISolution solve_cvrp_mpi(CVRPInstance& inst, int rank, int size);
void print_mpi_solution_json(const MPISolution& sol, std::ostream& out = std::cout);

#endif // VRP_MPI_H
