/**
 * @file algo4_cpp_parallel.hpp
 * @brief Pure C++ OpenMP parallelized implementation of Algorithm 4
 *
 * This implementation completely avoids Python's GIL by:
 * 1. Using only C++ data structures
 * 2. Running entirely in nogil sections
 * 3. Using OpenMP atomic operations for thread safety
 *
 * Expected performance: 4-16× speedup on multi-core systems
 */

#ifndef ALGO4_CPP_PARALLEL_HPP
#define ALGO4_CPP_PARALLEL_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <string>
#include <omp.h>

namespace algo4 {

/**
 * @brief Node in the computational graph
 */
struct Node {
    int id;
    std::string op_tag;
    std::vector<int> parent_ids;
    std::vector<double> parent_derivs;
    std::vector<double> parent_vals;
    double out_val;

    // Default constructor
    Node() : id(0), op_tag(""), out_val(0.0) {}
};

/**
 * @brief Computational tape
 */
struct Tape {
    std::vector<Node> nodes;
    std::unordered_map<int, int> var_to_idx;  // var_id -> index in nodes

    // Default constructor
    Tape() {}
};

/**
 * @brief Thread-safe sparse symmetric matrix using unordered_map
 */
class SparseMatrix {
private:
    int n_;
    std::unordered_map<int64_t, double> data_;
    std::unordered_map<int, std::unordered_set<int>> adj_;
    omp_lock_t lock_;  // OpenMP lock for thread safety

    inline int64_t key(int i, int j) const {
        int min_idx = (i <= j) ? i : j;
        int max_idx = (i <= j) ? j : i;
        return (static_cast<int64_t>(min_idx) << 32) | static_cast<int64_t>(max_idx);
    }

public:
    SparseMatrix(int n);
    ~SparseMatrix();

    // Thread-safe operations
    void add(int i, int j, double val);
    double get(int i, int j) const;
    std::vector<std::pair<int, double>> get_neighbors(int i) const;
    void clear_row_col(int i);
};

/**
 * @brief Compute second derivatives for a node
 * Uses int64 encoding for pairs: (i << 32) | j
 */
void compute_second_derivatives(
    const Node& node,
    std::unordered_map<int64_t, double>& d2
);

/**
 * @brief Pure C++ Algorithm 4 with OpenMP parallelization
 *
 * @param tape Computational tape
 * @param input_indices Indices of input variables
 * @param output_idx Index of output variable
 * @param n_threads Number of OpenMP threads
 * @return Hessian matrix as 2D vector
 */
std::vector<std::vector<double>> algo4_cpp_parallel(
    const Tape& tape,
    const std::vector<int>& input_indices,
    int output_idx,
    int n_threads = 16
);

/**
 * @brief Helper: Extract Hessian submatrix for inputs
 */
std::vector<std::vector<double>> extract_input_hessian(
    const SparseMatrix& W,
    const std::vector<int>& input_indices
);

} // namespace algo4

#endif // ALGO4_CPP_PARALLEL_HPP
