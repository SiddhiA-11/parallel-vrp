#include "vrp.h"

double calculate_euclidean_distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void compute_distance_matrix_seq(CVRPInstance& inst) {
    int total_nodes = inst.customers.size() + 1; // 0 is depot, 1..N are customers
    inst.dist_matrix.assign(total_nodes, std::vector<double>(total_nodes, 0.0));

    std::vector<Point> all_coords(total_nodes);
    all_coords[0] = inst.depot;
    for (size_t i = 0; i < inst.customers.size(); ++i) {
        all_coords[i + 1] = inst.customers[i].coord;
    }

    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < total_nodes; ++j) {
            if (i == j) {
                inst.dist_matrix[i][j] = 0.0;
            } else {
                inst.dist_matrix[i][j] = calculate_euclidean_distance(all_coords[i], all_coords[j]);
            }
        }
    }
}

CVRPInstance parse_vrplib_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    CVRPInstance inst;
    inst.name = "Unknown";
    inst.vehicle_count = 0;

    std::string line;
    enum Section { NONE, NODE_COORD, DEMAND, DEPOT } current_section = NONE;

    std::vector<Point> coords;
    std::vector<int> demands;
    std::vector<int> node_ids;

    while (std::getline(file, line)) {
        // Strip carriage returns and leading spaces
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) continue;

        if (line.find("NAME") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                inst.name = line.substr(colon + 1);
                inst.name.erase(0, inst.name.find_first_not_of(" \t"));
            }
            // Parse vehicle count from name if formatted like A-n32-k5
            size_t k_pos = inst.name.find("-k");
            if (k_pos != std::string::npos) {
                try {
                    inst.vehicle_count = std::stoi(inst.name.substr(k_pos + 2));
                } catch (...) {}
            }
        } else if (line.find("DIMENSION") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                inst.dimension = std::stoi(line.substr(colon + 1));
            }
        } else if (line.find("CAPACITY") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                inst.capacity = std::stoi(line.substr(colon + 1));
            }
        } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            current_section = NODE_COORD;
            continue;
        } else if (line.find("DEMAND_SECTION") != std::string::npos) {
            current_section = DEMAND;
            continue;
        } else if (line.find("DEPOT_SECTION") != std::string::npos) {
            current_section = DEPOT;
            continue;
        } else if (line.find("EOF") != std::string::npos) {
            break;
        }

        std::istringstream iss(line);
        if (current_section == NODE_COORD) {
            int id;
            double x, y;
            if (iss >> id >> x >> y) {
                node_ids.push_back(id);
                coords.push_back({x, y});
            }
        } else if (current_section == DEMAND) {
            int id;
            int dem;
            if (iss >> id >> dem) {
                demands.push_back(dem);
            }
        } else if (current_section == DEPOT) {
            int depot_id;
            if (iss >> depot_id && depot_id == -1) {
                current_section = NONE;
            }
        }
    }

    if (coords.empty()) {
        throw std::runtime_error("No node coordinates found in " + filepath);
    }

    // Node 1 is depot in VRPLIB convention
    inst.depot = coords[0];
    inst.customers.clear();

    for (size_t i = 1; i < coords.size(); ++i) {
        Customer c;
        c.id = (int)i; // Internal 1-based index (1 .. N)
        c.coord = coords[i];
        c.demand = (i < demands.size()) ? demands[i] : 1;
        inst.customers.push_back(c);
    }

    if (inst.vehicle_count <= 0) {
        // Calculate theoretical lower bound on vehicles
        int total_dem = 0;
        for (const auto& c : inst.customers) total_dem += c.demand;
        inst.vehicle_count = (inst.capacity > 0) ? std::max(1, (total_dem + inst.capacity - 1) / inst.capacity) : 5;
    }

    return inst;
}

CVRPInstance parse_json_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + filepath);
    }

    // Lightweight robust JSON parser for CVRP instances
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    CVRPInstance inst;
    inst.name = "Synthetic";
    inst.capacity = 100;
    inst.vehicle_count = 10;
    inst.depot = {50.0, 50.0};

    // Extract capacity
    size_t cap_pos = content.find("\"capacity\"");
    if (cap_pos != std::string::npos) {
        size_t colon = content.find(':', cap_pos);
        if (colon != std::string::npos) {
            inst.capacity = std::stoi(content.substr(colon + 1));
        }
    }

    // Extract vehicle_count
    size_t veh_pos = content.find("\"vehicle_count\"");
    if (veh_pos != std::string::npos) {
        size_t colon = content.find(':', veh_pos);
        if (colon != std::string::npos) {
            inst.vehicle_count = std::stoi(content.substr(colon + 1));
        }
    }

    // Extract depot x, y
    size_t depot_pos = content.find("\"depot\"");
    if (depot_pos != std::string::npos) {
        size_t x_pos = content.find("\"x\"", depot_pos);
        size_t y_pos = content.find("\"y\"", depot_pos);
        if (x_pos != std::string::npos && y_pos != std::string::npos) {
            inst.depot.x = std::stod(content.substr(content.find(':', x_pos) + 1));
            inst.depot.y = std::stod(content.substr(content.find(':', y_pos) + 1));
        }
    }

    // Extract customers array
    size_t cust_start = content.find("\"customers\"");
    if (cust_start != std::string::npos) {
        size_t arr_start = content.find('[', cust_start);
        size_t arr_end = content.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string cust_substr = content.substr(arr_start, arr_end - arr_start + 1);
            size_t pos = 0;
            int cid = 1;
            while ((pos = cust_substr.find('{', pos)) != std::string::npos) {
                size_t obj_end = cust_substr.find('}', pos);
                if (obj_end == std::string::npos) break;
                std::string obj_str = cust_substr.substr(pos, obj_end - pos + 1);

                Customer c;
                c.id = cid++;
                c.coord = {0.0, 0.0};
                c.demand = 10;

                size_t px = obj_str.find("\"x\"");
                if (px != std::string::npos) c.coord.x = std::stod(obj_str.substr(obj_str.find(':', px) + 1));
                size_t py = obj_str.find("\"y\"");
                if (py != std::string::npos) c.coord.y = std::stod(obj_str.substr(obj_str.find(':', py) + 1));
                size_t pdem = obj_str.find("\"demand\"");
                if (pdem != std::string::npos) c.demand = std::stoi(obj_str.substr(obj_str.find(':', pdem) + 1));

                inst.customers.push_back(c);
                pos = obj_end + 1;
            }
        }
    }

    inst.dimension = (int)inst.customers.size() + 1;
    return inst;
}

bool two_opt_route_seq(Route& route, const CVRPInstance& inst) {
    if (route.customer_ids.size() < 4) return false;

    // Full route includes depot at both ends: 0 -> c1 -> c2 -> ... -> cn -> 0
    std::vector<int> full_route;
    full_route.reserve(route.customer_ids.size() + 2);
    full_route.push_back(0);
    for (int cid : route.customer_ids) full_route.push_back(cid);
    full_route.push_back(0);

    bool improved = false;
    bool change = true;
    int max_iterations = 200;
    int iter = 0;

    while (change && iter++ < max_iterations) {
        change = false;
        double best_delta = -1e-6;
        int best_i = -1, best_j = -1;

        int n = (int)full_route.size();
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

                if (delta < best_delta) {
                    best_delta = delta;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i != -1) {
            std::reverse(full_route.begin() + best_i + 1, full_route.begin() + best_j + 1);
            change = true;
            improved = true;
        }
    }

    if (improved) {
        route.customer_ids.clear();
        for (size_t i = 1; i < full_route.size() - 1; ++i) {
            route.customer_ids.push_back(full_route[i]);
        }
        // Recalculate route distance
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

Solution solve_cvrp_sequential(CVRPInstance& inst) {
    auto start_time = std::chrono::high_resolution_clock::now();

    Solution sol;
    sol.algorithm = "Sequential";
    sol.dataset_name = inst.name;
    sol.customer_count = (int)inst.customers.size();
    sol.max_vehicles = inst.vehicle_count;
    sol.capacity = inst.capacity;

    // Phase 1: Distance matrix computation
    auto dist_start = std::chrono::high_resolution_clock::now();
    compute_distance_matrix_seq(inst);
    auto dist_end = std::chrono::high_resolution_clock::now();
    sol.dist_calc_time_ms = std::chrono::duration<double, std::milli>(dist_end - dist_start).count();

    // Phase 2: Capacity-aware Nearest Neighbor Route Construction
    int n = (int)inst.customers.size();
    std::vector<bool> visited(n + 1, false); // 1-based indexing for customers
    int unvisited_count = n;

    while (unvisited_count > 0) {
        Route route;
        int remaining_cap = inst.capacity;
        int current_loc = 0; // Starts at depot

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

            if (nearest_cust == -1) {
                // No customer fits in current vehicle capacity; return to depot
                break;
            }

            visited[nearest_cust] = true;
            unvisited_count--;
            route.customer_ids.push_back(nearest_cust);
            remaining_cap -= inst.customers[nearest_cust - 1].demand;
            route.load += inst.customers[nearest_cust - 1].demand;
            current_loc = nearest_cust;
        }

        if (route.customer_ids.empty()) {
            // Infeasible: A single customer's demand exceeds capacity
            sol.valid = false;
            sol.error_message = "Infeasible: Customer demand exceeds vehicle capacity";
            break;
        }

        // Calculate initial route distance: depot -> customers -> depot
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

    // Phase 3: Route Improvement via 2-opt Local Search
    auto ls_start = std::chrono::high_resolution_clock::now();
    for (auto& route : sol.routes) {
        two_opt_route_seq(route, inst);
    }
    auto ls_end = std::chrono::high_resolution_clock::now();
    sol.local_search_time_ms = std::chrono::duration<double, std::milli>(ls_end - ls_start).count();

    // Compute total distance & vehicles used
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

bool validate_solution(const CVRPInstance& inst, const Solution& sol, std::string& error_msg) {
    int n = (int)inst.customers.size();
    std::vector<int> visit_count(n + 1, 0);
    int total_demand_served = 0;
    int expected_total_demand = 0;
    for (const auto& c : inst.customers) expected_total_demand += c.demand;

    double calculated_total_dist = 0.0;

    for (size_t r = 0; r < sol.routes.size(); ++r) {
        const auto& route = sol.routes[r];
        int route_load = 0;
        double route_dist = 0.0;
        int prev = 0;

        for (int cid : route.customer_ids) {
            if (cid < 1 || cid > n) {
                error_msg = "Invalid customer ID " + std::to_string(cid) + " in route " + std::to_string(r);
                return false;
            }
            visit_count[cid]++;
            route_load += inst.customers[cid - 1].demand;
            route_dist += inst.dist_matrix[prev][cid];
            prev = cid;
        }
        route_dist += inst.dist_matrix[prev][0];

        if (route_load > inst.capacity) {
            error_msg = "Vehicle capacity exceeded on route " + std::to_string(r) + ": " +
                        std::to_string(route_load) + " > " + std::to_string(inst.capacity);
            return false;
        }

        total_demand_served += route_load;
        calculated_total_dist += route_dist;
    }

    for (int cid = 1; cid <= n; ++cid) {
        if (visit_count[cid] == 0) {
            error_msg = "Customer " + std::to_string(cid) + " was not visited";
            return false;
        }
        if (visit_count[cid] > 1) {
            error_msg = "Customer " + std::to_string(cid) + " was visited multiple times (" +
                        std::to_string(visit_count[cid]) + ")";
            return false;
        }
    }

    if (total_demand_served != expected_total_demand) {
        error_msg = "Total demand mismatch: served " + std::to_string(total_demand_served) +
                    ", expected " + std::to_string(expected_total_demand);
        return false;
    }

    if (inst.vehicle_count > 0 && (int)sol.routes.size() > inst.vehicle_count * 2) {
        // Warning or constraint limit
    }

    error_msg = "Solution is valid";
    return true;
}

void print_solution_json(const Solution& sol, std::ostream& out) {
    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"algorithm\": \"" << sol.algorithm << "\",\n";
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
