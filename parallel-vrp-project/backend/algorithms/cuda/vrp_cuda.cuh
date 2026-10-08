#ifndef VRP_CUDA_CUH
#define VRP_CUDA_CUH

#include "../sequential/vrp.h"

struct CUDASolution : public Solution {
    int block_size = 256;
    double h2d_transfer_time_ms = 0.0;
    double d2h_transfer_time_ms = 0.0;
    double gpu_kernel_time_ms = 0.0;
    std::string device_name = "NVIDIA GPU";
};

// Prototypes for CUDA functions
CUDASolution solve_cvrp_cuda(CVRPInstance& inst, int block_size = 256);
void print_cuda_solution_json(const CUDASolution& sol, std::ostream& out = std::cout);

#endif // VRP_CUDA_CUH
