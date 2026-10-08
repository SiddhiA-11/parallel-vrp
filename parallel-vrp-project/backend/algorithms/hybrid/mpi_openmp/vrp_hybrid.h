#ifndef VRP_HYBRID_H
#define VRP_HYBRID_H

#include "../../sequential/vrp.h"
#include "../../openmp/vrp_openmp.h"
#include <mpi.h>
#include <omp.h>

struct HybridSolution : public Solution {
    int mpi_processes = 1;
    int threads_per_process = 1;
    double computation_time_ms = 0.0;
    double communication_time_ms = 0.0;
};

HybridSolution solve_cvrp_hybrid(CVRPInstance& inst, int rank, int size, int threads_per_proc);
void print_hybrid_solution_json(const HybridSolution& sol, std::ostream& out = std::cout);

#endif // VRP_HYBRID_H
