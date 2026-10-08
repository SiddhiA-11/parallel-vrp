#include "vrp_hybrid.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

HybridSolution solve_cvrp_hybrid(CVRPInstance& inst, int rank, int size, int threads_per_proc) {
    if (threads_per_proc <= 0) {
        threads_per_proc = omp_get_max_threads();
    }
    omp_set_num_threads(threads_per_proc);

    HybridSolution sol;
    sol.algorithm = "Hybrid (MPI+OpenMP)";
    sol.mpi_processes = size;
    sol.threads_per_process = threads_per_proc;
    sol.dataset_name = inst.name;
    sol.customer_count = (int)inst.customers.size();
    sol.max_vehicles = inst.vehicle_count;
    sol.capacity = inst.capacity;

    double t_start = MPI_Wtime();
    double comm_time_total = 0.0;
    double comp_time_total = 0.0;

    int total_nodes = sol.customer_count + 1;
    std::vector<double> coords_x(total_nodes);
    std::vector<double> coords_y(total_nodes);
    std::vector<int> demands(total_nodes, 0);

    if (rank == 0) {
        coords_x[0] = inst.depot.x;
        coords_y[0] = inst.depot.y;
        for (int i = 0; i < sol.customer_count; ++i) {
            coords_x[i + 1] = inst.customers[i].coord.x;
            coords_y[i + 1] = inst.customers[i].coord.y;
            demands[i + 1] = inst.customers[i].demand;
        }
    }

    // Step 1: MPI Broadcast of problem data
    double t_comm0 = MPI_Wtime();
    int meta[4] = {sol.customer_count, inst.capacity, inst.vehicle_count, threads_per_proc};
    MPI_Bcast(meta, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        sol.customer_count = meta[0];
        inst.capacity = meta[1];
        inst.vehicle_count = meta[2];
        threads_per_proc = meta[3];
        total_nodes = sol.customer_count + 1;
        coords_x.resize(total_nodes);
        coords_y.resize(total_nodes);
        demands.resize(total_nodes);
        omp_set_num_threads(threads_per_proc);
    }
    MPI_Bcast(coords_x.data(), total_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(coords_y.data(), total_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(demands.data(), total_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    comm_time_total += (MPI_Wtime() - t_comm0);

    if (rank != 0) {
        inst.depot = {coords_x[0], coords_y[0]};
        inst.customers.clear();
        for (int i = 1; i < total_nodes; ++i) {
            Customer c;
            c.id = i;
            c.coord = {coords_x[i], coords_y[i]};
            c.demand = demands[i];
            inst.customers.push_back(c);
        }
    }

    // Step 2: Hybrid Distance Matrix computation
    // MPI assigns rows to ranks, then OpenMP threads in each rank compute row elements
    double t_comp0 = MPI_Wtime();
    int rows_per_proc = total_nodes / size;
    int rem = total_nodes % size;
    int my_start = rank * rows_per_proc + std::min(rank, rem);
    int my_rows = rows_per_proc + (rank < rem ? 1 : 0);

    std::vector<double> my_dist_rows(my_rows * total_nodes);
    #pragma omp parallel for num_threads(threads_per_proc) schedule(static)
    for (int i = 0; i < my_rows; ++i) {
        int global_i = my_start + i;
        for (int j = 0; j < total_nodes; ++j) {
            if (global_i == j) {
                my_dist_rows[i * total_nodes + j] = 0.0;
            } else {
                double dx = coords_x[global_i] - coords_x[j];
                double dy = coords_y[global_i] - coords_y[j];
                my_dist_rows[i * total_nodes + j] = std::sqrt(dx * dx + dy * dy);
            }
        }
    }
    comp_time_total += (MPI_Wtime() - t_comp0);

    // Step 3: Allgather distance matrix
    double t_comm1 = MPI_Wtime();
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    int curr_disp = 0;
    for (int p = 0; p < size; ++p) {
        int p_rows = rows_per_proc + (p < rem ? 1 : 0);
        recv_counts[p] = p_rows * total_nodes;
        displs[p] = curr_disp;
        curr_disp += recv_counts[p];
    }

    std::vector<double> flat_dist_matrix(total_nodes * total_nodes);
    MPI_Allgatherv(my_dist_rows.data(), my_rows * total_nodes, MPI_DOUBLE,
                  flat_dist_matrix.data(), recv_counts.data(), displs.data(),
                  MPI_DOUBLE, MPI_COMM_WORLD);
    comm_time_total += (MPI_Wtime() - t_comm1);

    inst.dist_matrix.assign(total_nodes, std::vector<double>(total_nodes, 0.0));
    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            inst.dist_matrix[i][j] = flat_dist_matrix[i * total_nodes + j];
        }
    }

    // Step 4: Route Construction (Capacity-aware Nearest Neighbor)
    double t_comp1 = MPI_Wtime();
    int n = sol.customer_count;
    std::vector<bool> visited(n + 1, false);
    int unvisited_count = n;
    std::vector<Route> initial_routes;

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

        initial_routes.push_back(route);
    }
    comp_time_total += (MPI_Wtime() - t_comp1);

    // Step 5: Distributed + Multithreaded Local Search
    // MPI assigns routes to rank, OpenMP inside each rank speeds up 2-opt search
    double t_comp2 = MPI_Wtime();
    int total_routes = (int)initial_routes.size();
    std::vector<Route> my_optimized_routes;

    for (int r = 0; r < total_routes; ++r) {
        if (r % size == rank) {
            Route opt_r = initial_routes[r];
            two_opt_single_route_omp(opt_r, inst, threads_per_proc);
            my_optimized_routes.push_back(opt_r);
        }
    }
    comp_time_total += (MPI_Wtime() - t_comp2);

    // Step 6: Gather at Root
    double t_comm2 = MPI_Wtime();
    if (rank == 0) {
        sol.routes.resize(total_routes);
        int my_idx = 0;
        for (int r = 0; r < total_routes; ++r) {
            if (r % size == 0) {
                sol.routes[r] = my_optimized_routes[my_idx++];
            }
        }

        for (int p = 1; p < size; ++p) {
            int p_assigned = 0;
            for (int r = 0; r < total_routes; ++r) {
                if (r % size == p) p_assigned++;
            }

            for (int k = 0; k < p_assigned; ++k) {
                int r_idx = 0, cust_count = 0, load = 0;
                double dist = 0.0;
                MPI_Recv(&r_idx, 1, MPI_INT, p, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&cust_count, 1, MPI_INT, p, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&load, 1, MPI_INT, p, 202, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&dist, 1, MPI_DOUBLE, p, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<int> custs(cust_count);
                MPI_Recv(custs.data(), cust_count, MPI_INT, p, 204, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Route r;
                r.customer_ids = custs;
                r.load = load;
                r.distance = dist;
                sol.routes[r_idx] = r;
            }
        }
    } else {
        int my_idx = 0;
        for (int r = 0; r < total_routes; ++r) {
            if (r % size == rank) {
                const Route& r_obj = my_optimized_routes[my_idx++];
                int cust_count = (int)r_obj.customer_ids.size();
                int load = r_obj.load;
                double dist = r_obj.distance;

                MPI_Send(&r, 1, MPI_INT, 0, 200, MPI_COMM_WORLD);
                MPI_Send(&cust_count, 1, MPI_INT, 0, 201, MPI_COMM_WORLD);
                MPI_Send(&load, 1, MPI_INT, 0, 202, MPI_COMM_WORLD);
                MPI_Send(&dist, 1, MPI_DOUBLE, 0, 203, MPI_COMM_WORLD);
                MPI_Send(r_obj.customer_ids.data(), cust_count, MPI_INT, 0, 204, MPI_COMM_WORLD);
            }
        }
    }
    comm_time_total += (MPI_Wtime() - t_comm2);

    double t_total = MPI_Wtime() - t_start;
    sol.execution_time_ms = t_total * 1000.0;
    sol.communication_time_ms = comm_time_total * 1000.0;
    sol.computation_time_ms = comp_time_total * 1000.0;

    if (rank == 0) {
        sol.vehicles_used = (int)sol.routes.size();
        sol.total_distance = 0.0;
        for (const auto& r : sol.routes) sol.total_distance += r.distance;

        std::string err;
        sol.valid = validate_solution(inst, sol, err);
        sol.error_message = err;
    }

    return sol;
}

void print_hybrid_solution_json(const HybridSolution& sol, std::ostream& out) {
    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"algorithm\": \"" << sol.algorithm << "\",\n";
    out << "  \"mpi_processes\": " << sol.mpi_processes << ",\n";
    out << "  \"threads_per_process\": " << sol.threads_per_process << ",\n";
    out << "  \"total_execution_units\": " << (sol.mpi_processes * sol.threads_per_process) << ",\n";
    out << "  \"dataset\": \"" << sol.dataset_name << "\",\n";
    out << "  \"customers\": " << sol.customer_count << ",\n";
    out << "  \"max_vehicles\": " << sol.max_vehicles << ",\n";
    out << "  \"vehicles_used\": " << sol.vehicles_used << ",\n";
    out << "  \"capacity\": " << sol.capacity << ",\n";
    out << "  \"total_distance\": " << sol.total_distance << ",\n";
    out << "  \"execution_time_ms\": " << sol.execution_time_ms << ",\n";
    out << "  \"computation_time_ms\": " << sol.computation_time_ms << ",\n";
    out << "  \"communication_time_ms\": " << sol.communication_time_ms << ",\n";
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
