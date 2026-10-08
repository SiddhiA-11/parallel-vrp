#ifndef VRP_OPENMP_H
#define VRP_OPENMP_H

#include "../sequential/vrp.h"
#include <omp.h>

struct OpenMPSolution : public Solution {
    int threads_used = 1;
};

void compute_distance_matrix_omp(CVRPInstance& inst, int num_threads);
bool two_opt_route_omp_inter(std::vector<Route>& routes, const CVRPInstance& inst, int num_threads);
bool two_opt_single_route_omp(Route& route, const CVRPInstance& inst, int num_threads);
OpenMPSolution solve_cvrp_openmp(CVRPInstance& inst, int num_threads);
void print_openmp_solution_json(const OpenMPSolution& sol, std::ostream& out = std::cout);

#endif // VRP_OPENMP_H
