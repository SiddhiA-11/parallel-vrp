#include "vrp_openmp.h"

void compute_distance_matrix_omp(CVRPInstance& inst, int num_threads) {
    int total_nodes = (int)inst.customers.size() + 1;
    inst.dist_matrix.assign(total_nodes, std::vector<double>(total_nodes, 0.0));

    std::vector<Point> all_coords(total_nodes);
    all_coords[0] = inst.depot;
    for (size_t i = 0; i < inst.customers.size(); ++i) {
        all_coords[i + 1] = inst.customers[i].coord;
    }

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            if (i == j) {
                inst.dist_matrix[i][j] = 0.0;
            } else {
                double dx = all_coords[i].x - all_coords[j].x;
                double dy = all_coords[i].y - all_coords[j].y;
                inst.dist_matrix[i][j] = std::sqrt(dx * dx + dy * dy);
            }
        }
    }
}

bool two_opt_single_route_omp(Route& route, const CVRPInstance& inst, int num_threads) {
    if (route.customer_ids.size() < 4) return false;

    std::vector<int> full_route;
    full_route.reserve(route.customer_ids.size() + 2);
    full_route.push_back(0);
    for (int cid : route.customer_ids) full_route.push_back(cid);
    full_route.push_back(0);

    bool improved = false;
    bool change = true;
    int max_iterations = 200;
    int iter = 0;
    int n = (int)full_route.size();

    while (change && iter++ < max_iterations) {
        change = false;
        double global_best_delta = -1e-6;
        int global_best_i = -1, global_best_j = -1;

        // Parallelize candidate pair search over outer loop i
        #pragma omp parallel num_threads(num_threads)
        {
            double local_best_delta = -1e-6;
            int local_best_i = -1, local_best_j = -1;

            #pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < n - 2; ++i) {
                int u1 = full_route[i];
                int u2 = full_route[i + 1];
                double d_u1_u2 = inst.dist_matrix[u1][u2];

                for (int j = i + 2; j < n - 1; ++j) {
                    int v1 = full_route[j];
                    int v2 = full_route[j + 1];

                    double current_edges = d_u1_u2 + inst.dist_matrix[v1][v2];
                    double new_edges = inst.dist_matrix[u1][v1] + inst.dist_matrix[u2][v2];
                    double delta = new_edges - current_edges;

                    if (delta < local_best_delta) {
                        local_best_delta = delta;
                        local_best_i = i;
                        local_best_j = j;
                    }
                }
            }

            // Reduction of best move across threads in critical section
            #pragma omp critical
            {
                if (local_best_delta < global_best_delta) {
                    global_best_delta = local_best_delta;
                    global_best_i = local_best_i;
                    global_best_j = local_best_j;
                }
            }
        }

        if (global_best_i != -1) {
            std::reverse(full_route.begin() + global_best_i + 1, full_route.begin() + global_best_j + 1);
            change = true;
            improved = true;
        }
    }

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

    return improved;
}

OpenMPSolution solve_cvrp_openmp(CVRPInstance& inst, int num_threads) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    auto start_time = std::chrono::high_resolution_clock::now();

    OpenMPSolution sol;
    sol.algorithm = "OpenMP";
    sol.dataset_name = inst.name;
    sol.customer_count = (int)inst.customers.size();
    sol.max_vehicles = inst.vehicle_count;
    sol.capacity = inst.capacity;
    sol.threads_used = num_threads;

    // Phase 1: Parallel Distance Matrix Computation with OpenMP
    auto dist_start = std::chrono::high_resolution_clock::now();
    compute_distance_matrix_omp(inst, num_threads);
    auto dist_end = std::chrono::high_resolution_clock::now();
    sol.dist_calc_time_ms = std::chrono::duration<double, std::milli>(dist_end - dist_start).count();

    // Phase 2: Capacity-aware Route Construction
    int n = (int)inst.customers.size();
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

        if (route.customer_ids.empty()) {
            sol.valid = false;
            sol.error_message = "Infeasible: Customer demand exceeds vehicle capacity";
            break;
        }

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

    // Phase 3: Parallel Route Improvement using OpenMP
    auto ls_start = std::chrono::high_resolution_clock::now();

    // Independent routes can be safely parallelized without race conditions!
    int total_routes = (int)sol.routes.size();
    if (total_routes >= num_threads) {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int r = 0; r < total_routes; ++r) {
            two_opt_single_route_omp(sol.routes[r], inst, 1);
        }
    } else {
        // If few routes, use intra-route parallelism
        for (int r = 0; r < total_routes; ++r) {
            two_opt_single_route_omp(sol.routes[r], inst, num_threads);
        }
    }

    auto ls_end = std::chrono::high_resolution_clock::now();
    sol.local_search_time_ms = std::chrono::duration<double, std::milli>(ls_end - ls_start).count();

    sol.vehicles_used = (int)sol.routes.size();
    sol.total_distance = 0.0;
    for (const auto& r : sol.routes) {
        sol.total_distance += r.distance;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    sol.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::string err;
    sol.valid = validate_solution(inst, sol, err);
    sol.error_message = err;

    return sol;
}

void print_openmp_solution_json(const OpenMPSolution& sol, std::ostream& out) {
    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"algorithm\": \"" << sol.algorithm << "\",\n";
    out << "  \"threads\": " << sol.threads_used << ",\n";
    out << "  \"dataset\": \"" << sol.dataset_name << "\",\n";
    out << "  \"customers\": " << sol.customer_count << ",\n";
    out << "  \"max_vehicles\": " << sol.max_vehicles << ",\n";
    out << "  \"vehicles_used\": " << sol.vehicles_used << ",\n";
    out << "  \"capacity\": " << sol.capacity << ",\n";
    out << "  \"total_distance\": " << sol.total_distance << ",\n";
    out << "  \"execution_time_ms\": " << sol.execution_time_ms << ",\n";
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
