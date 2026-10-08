#include "vrp_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// CUDA Kernel: Distance Matrix Computation
// Uses 2D grid/block layout for coalesced memory access
__global__ void compute_distance_matrix_kernel(
    const double* __restrict__ d_coords_x,
    const double* __restrict__ d_coords_y,
    double* __restrict__ d_dist_matrix,
    int total_nodes)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row

    if (i < total_nodes && j < total_nodes) {
        if (i == j) {
            d_dist_matrix[i * total_nodes + j] = 0.0;
        } else {
            double dx = d_coords_x[i] - d_coords_x[j];
            double dy = d_coords_y[i] - d_coords_y[j];
            d_dist_matrix[i * total_nodes + j] = sqrt(dx * dx + dy * dy);
        }
    }
}

// Structure for 2-opt candidate reduction
struct MoveCandidate {
    double delta;
    int i;
    int j;
};

// CUDA Kernel: 2-Opt Candidate Move Evaluation
// Parallelizes evaluation over candidate edge swaps (i, j)
__global__ void evaluate_2opt_kernel(
    const int* __restrict__ d_route,
    int route_len,
    const double* __restrict__ d_dist_matrix,
    int total_nodes,
    MoveCandidate* __restrict__ d_block_best)
{
    extern __shared__ MoveCandidate s_moves[];
    int tid = threadIdx.x;

    MoveCandidate best_local;
    best_local.delta = -1e-6;
    best_local.i = -1;
    best_local.j = -1;

    // Total pairs is approximately (route_len - 2) * (route_len - 3) / 2
    int total_pairs = 0;
    for (int i = 0; i < route_len - 2; ++i) {
        total_pairs += (route_len - 1 - (i + 2));
    }

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Map linear global_idx to (i, j) pair
    int curr_idx = 0;
    for (int i = 0; i < route_len - 2; ++i) {
        int u1 = d_route[i];
        int u2 = d_route[i + 1];
        double d_u1_u2 = d_dist_matrix[u1 * total_nodes + u2];

        for (int j = i + 2; j < route_len - 1; ++j) {
            if (curr_idx == global_idx) {
                int v1 = d_route[j];
                int v2 = d_route[j + 1];

                double current_edges = d_u1_u2 + d_dist_matrix[v1 * total_nodes + v2];
                double new_edges = d_dist_matrix[u1 * total_nodes + v1] + d_dist_matrix[u2 * total_nodes + v2];
                double delta = new_edges - current_edges;

                if (delta < best_local.delta) {
                    best_local.delta = delta;
                    best_local.i = i;
                    best_local.j = j;
                }
            }
            curr_idx++;
        }
    }

    s_moves[tid] = best_local;
    __syncthreads();

    // Reduction within shared memory of the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_moves[tid + s].delta < s_moves[tid].delta) {
                s_moves[tid] = s_moves[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_best[blockIdx.x] = s_moves[0];
    }
}

CUDASolution solve_cvrp_cuda(CVRPInstance& inst, int block_size) {
    CUDASolution sol;
    sol.algorithm = "CUDA";
    sol.dataset_name = inst.name;
    sol.customer_count = (int)inst.customers.size();
    sol.max_vehicles = inst.vehicle_count;
    sol.capacity = inst.capacity;
    sol.block_size = block_size;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        sol.valid = false;
        sol.error_message = "No CUDA-capable GPU detected on this system.";
        return sol;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sol.device_name = prop.name;

    auto total_wall_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    int total_nodes = sol.customer_count + 1;
    size_t coords_size = total_nodes * sizeof(double);
    size_t matrix_size = total_nodes * total_nodes * sizeof(double);

    std::vector<double> h_coords_x(total_nodes);
    std::vector<double> h_coords_y(total_nodes);
    h_coords_x[0] = inst.depot.x;
    h_coords_y[0] = inst.depot.y;
    for (int i = 0; i < sol.customer_count; ++i) {
        h_coords_x[i + 1] = inst.customers[i].coord.x;
        h_coords_y[i + 1] = inst.customers[i].coord.y;
    }

    double* d_coords_x = nullptr;
    double* d_coords_y = nullptr;
    double* d_dist_matrix = nullptr;

    cudaMalloc((void**)&d_coords_x, coords_size);
    cudaMalloc((void**)&d_coords_y, coords_size);
    cudaMalloc((void**)&d_dist_matrix, matrix_size);

    // Host-to-Device Transfer Timing
    cudaEventRecord(start_event, 0);
    cudaMemcpy(d_coords_x, h_coords_x.data(), coords_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_y, h_coords_y.data(), coords_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, start_event, stop_event);
    sol.h2d_transfer_time_ms += h2d_ms;

    // Launch Distance Matrix Kernel
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((total_nodes + 15) / 16, (total_nodes + 15) / 16);

    cudaEventRecord(start_event, 0);
    compute_distance_matrix_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_coords_x, d_coords_y, d_dist_matrix, total_nodes);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start_event, stop_event);
    sol.gpu_kernel_time_ms += kernel_ms;
    sol.dist_calc_time_ms = kernel_ms;

    // Device-to-Host Transfer Timing
    std::vector<double> h_flat_matrix(total_nodes * total_nodes);
    cudaEventRecord(start_event, 0);
    cudaMemcpy(h_flat_matrix.data(), d_dist_matrix, matrix_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float d2h_ms = 0.0f;
    cudaEventElapsedTime(&d2h_ms, start_event, stop_event);
    sol.d2h_transfer_time_ms += d2h_ms;

    // Populate inst.dist_matrix
    inst.dist_matrix.assign(total_nodes, std::vector<double>(total_nodes, 0.0));
    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            inst.dist_matrix[i][j] = h_flat_matrix[i * total_nodes + j];
        }
    }

    // CPU Stage: Capacity-aware Route Construction
    int n = sol.customer_count;
    std::vector<bool> visited(n + 1, false);
    int unvisited_count = n;

    while (unvisited_count > 0) {
        Route route;
        int remaining_cap = inst.capacity;
        int current_loc = 0;

        while (true) {
            int nearest_cust = -1;
            double min_dist = 1e18;

            for (int cid = 1; cid <= n; ++cid) {
                if (!visited[cid] && inst.customers[cid - 1].demand <= remaining_cap) {
                    double d = inst.dist_matrix[current_loc][cid];
                    if (d < min_dist) {
                        min_dist = d;
                        nearest_cust = cid;
                    }
                }
            }

            if (nearest_cust == -1) break;

            visited[nearest_cust] = true;
            unvisited_count--;
            route.customer_ids.push_back(nearest_cust);
            remaining_cap -= inst.customers[nearest_cust - 1].demand;
            route.load += inst.customers[nearest_cust - 1].demand;
            current_loc = nearest_cust;
        }

        if (route.customer_ids.empty()) break;

        double r_dist = 0.0;
        int prev = 0;
        for (int cid : route.customer_ids) {
            r_dist += inst.dist_matrix[prev][cid];
            prev = cid;
        }
        r_dist += inst.dist_matrix[prev][0];
        route.distance = r_dist;

        sol.routes.push_back(route);
    }

    // GPU Stage: 2-Opt Local Search optimization
    auto ls_wall_start = std::chrono::high_resolution_clock::now();
    for (auto& route : sol.routes) {
        if (route.customer_ids.size() < 4) continue;

        std::vector<int> full_route;
        full_route.push_back(0);
        for (int cid : route.customer_ids) full_route.push_back(cid);
        full_route.push_back(0);

        int r_len = (int)full_route.size();
        int total_pairs = 0;
        for (int i = 0; i < r_len - 2; ++i) {
            total_pairs += (r_len - 1 - (i + 2));
        }

        if (total_pairs <= 0) continue;

        int* d_route = nullptr;
        cudaMalloc((void**)&d_route, r_len * sizeof(int));
        cudaMemcpy(d_route, full_route.data(), r_len * sizeof(int), cudaMemcpyHostToDevice);

        int num_blocks = (total_pairs + block_size - 1) / block_size;
        MoveCandidate* d_block_best = nullptr;
        cudaMalloc((void**)&d_block_best, num_blocks * sizeof(MoveCandidate));

        bool improved = false;
        bool change = true;
        int iter = 0;

        while (change && iter++ < 100) {
            change = false;

            cudaMemcpy(d_route, full_route.data(), r_len * sizeof(int), cudaMemcpyHostToDevice);

            evaluate_2opt_kernel<<<num_blocks, block_size, block_size * sizeof(MoveCandidate)>>>(
                d_route, r_len, d_dist_matrix, total_nodes, d_block_best);
            cudaDeviceSynchronize();

            std::vector<MoveCandidate> h_block_best(num_blocks);
            cudaMemcpy(h_block_best.data(), d_block_best, num_blocks * sizeof(MoveCandidate), cudaMemcpyDeviceToHost);

            MoveCandidate best_global;
            best_global.delta = -1e-6;
            best_global.i = -1;
            best_global.j = -1;

            for (const auto& mc : h_block_best) {
                if (mc.delta < best_global.delta) {
                    best_global = mc;
                }
            }

            if (best_global.i != -1) {
                std::reverse(full_route.begin() + best_global.i + 1, full_route.begin() + best_global.j + 1);
                change = true;
                improved = true;
            }
        }

        cudaFree(d_route);
        cudaFree(d_block_best);

        if (improved) {
            route.customer_ids.clear();
            for (size_t i = 1; i < full_route.size() - 1; ++i) {
                route.customer_ids.push_back(full_route[i]);
            }
            double d = 0.0;
            int prev = 0;
            for (int cid : route.customer_ids) {
                d += inst.dist_matrix[prev][cid];
                prev = cid;
            }
            d += inst.dist_matrix[prev][0];
            route.distance = d;
        }
    }
    auto ls_wall_end = std::chrono::high_resolution_clock::now();
    sol.local_search_time_ms = std::chrono::duration<double, std::milli>(ls_wall_end - ls_wall_start).count();

    // Clean up GPU allocations
    cudaFree(d_coords_x);
    cudaFree(d_coords_y);
    cudaFree(d_dist_matrix);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    sol.vehicles_used = (int)sol.routes.size();
    sol.total_distance = 0.0;
    for (const auto& r : sol.routes) sol.total_distance += r.distance;

    auto total_wall_end = std::chrono::high_resolution_clock::now();
    sol.execution_time_ms = std::chrono::duration<double, std::milli>(total_wall_end - total_wall_start).count();

    std::string err;
    sol.valid = validate_solution(inst, sol, err);
    sol.error_message = err;

    return sol;
}

void print_cuda_solution_json(const CUDASolution& sol, std::ostream& out) {
    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"algorithm\": \"" << sol.algorithm << "\",\n";
    out << "  \"device_name\": \"" << sol.device_name << "\",\n";
    out << "  \"block_size\": " << sol.block_size << ",\n";
    out << "  \"dataset\": \"" << sol.dataset_name << "\",\n";
    out << "  \"customers\": " << sol.customer_count << ",\n";
    out << "  \"max_vehicles\": " << sol.max_vehicles << ",\n";
    out << "  \"vehicles_used\": " << sol.vehicles_used << ",\n";
    out << "  \"capacity\": " << sol.capacity << ",\n";
    out << "  \"total_distance\": " << sol.total_distance << ",\n";
    out << "  \"execution_time_ms\": " << sol.execution_time_ms << ",\n";
    out << "  \"gpu_kernel_time_ms\": " << sol.gpu_kernel_time_ms << ",\n";
    out << "  \"h2d_transfer_time_ms\": " << sol.h2d_transfer_time_ms << ",\n";
    out << "  \"d2h_transfer_time_ms\": " << sol.d2h_transfer_time_ms << ",\n";
    out << "  \"dist_calc_time_ms\": " << sol.dist_calc_time_ms << ",\n";
    out << "  \"local_search_time_ms\": " << sol.local_search_time_ms << ",\n";
    out << "  \"valid\": " << (sol.valid ? "true" : "false") << ",\n";
    out << "  \"error_message\": \"" << sol.error_message << "\",\n";
    out << "  \"routes\": [\n";
    for (size_t i = 0; i < sol.routes.size(); ++i) {
        const auto& r = sol.routes[i];
        out << "    {\n";
        out << "      \"route_id\": " << i << ",\n";
        out << "      \"load\": " << r.load << ",\n";
        out << "      \"distance\": " << r.distance << ",\n";
        out << "      \"customers\": [";
        for (size_t j = 0; j < r.customer_ids.size(); ++j) {
            out << r.customer_ids[j] << (j + 1 < r.customer_ids.size() ? ", " : "");
        }
        out << "]\n";
        out << "    }" << (i + 1 < sol.routes.size() ? ",\n" : "\n");
    }
    out << "  ]\n";
    out << "}\n";
}
