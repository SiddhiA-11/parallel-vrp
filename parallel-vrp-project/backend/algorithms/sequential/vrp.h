#ifndef VRP_SEQUENTIAL_H
#define VRP_SEQUENTIAL_H

#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

struct Point {
    double x;
    double y;
};

struct Customer {
    int id; // 1-based customer id (depot is id 1 in VRPLIB or index 0)
    Point coord;
    int demand;
};

struct Route {
    std::vector<int> customer_ids; // Customer IDs in visit order
    double distance = 0.0;
    int load = 0;
};

struct CVRPInstance {
    std::string name;
    int dimension = 0;        // Total nodes including depot
    int capacity = 0;         // Vehicle maximum capacity
    int vehicle_count = 0;    // Maximum available vehicles (0 = unconstrained)
    Point depot;
    std::vector<Customer> customers; // Customers (excluding depot)
    std::vector<std::vector<double>> dist_matrix; // Dimension x Dimension (index 0 = depot)
};

struct Solution {
    std::string algorithm;
    std::string dataset_name;
    int customer_count = 0;
    int max_vehicles = 0;
    int vehicles_used = 0;
    int capacity = 0;
    double total_distance = 0.0;
    double execution_time_ms = 0.0;
    double dist_calc_time_ms = 0.0;
    double local_search_time_ms = 0.0;
    bool valid = false;
    std::string error_message;
    std::vector<Route> routes;
};

// Function prototypes
double calculate_euclidean_distance(const Point& p1, const Point& p2);
void compute_distance_matrix_seq(CVRPInstance& inst);
CVRPInstance parse_vrplib_file(const std::string& filepath);
CVRPInstance parse_json_file(const std::string& filepath);

Solution solve_cvrp_sequential(CVRPInstance& inst);
bool two_opt_route_seq(Route& route, const CVRPInstance& inst);
bool validate_solution(const CVRPInstance& inst, const Solution& sol, std::string& error_msg);
void print_solution_json(const Solution& sol, std::ostream& out = std::cout);

#endif // VRP_SEQUENTIAL_H
