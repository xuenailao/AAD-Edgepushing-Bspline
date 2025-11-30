/**
 * @file algo4_cpp_parallel.cpp
 * @brief Implementation of pure C++ OpenMP parallelized Algorithm 4
 */

#include "algo4_cpp_parallel.hpp"
#include <cmath>
#include <algorithm>
#include <functional>
#include <iostream>

namespace algo4 {

// ============================================================================
// SparseMatrix Implementation
// ============================================================================

SparseMatrix::SparseMatrix(int n) : n_(n) {
    omp_init_lock(&lock_);
}

SparseMatrix::~SparseMatrix() {
    omp_destroy_lock(&lock_);
}

void SparseMatrix::add(int i, int j, double val) {
    if (val == 0.0) return;

    int64_t k = key(i, j);

    // Thread-safe update using OpenMP lock
    omp_set_lock(&lock_);

    auto it = data_.find(k);
    if (it != data_.end()) {
        it->second += val;
        if (std::abs(it->second) < 1e-15) {
            data_.erase(it);
            // Update adjacency
            auto& adj_i = adj_[i];
            adj_i.erase(j);
            if (adj_i.empty()) adj_.erase(i);
            if (i != j) {
                auto& adj_j = adj_[j];
                adj_j.erase(i);
                if (adj_j.empty()) adj_.erase(j);
            }
        }
    } else {
        data_[k] = val;
        // Update adjacency
        adj_[i].insert(j);
        if (i != j) {
            adj_[j].insert(i);
        }
    }

    omp_unset_lock(&lock_);
}

double SparseMatrix::get(int i, int j) const {
    int64_t k = key(i, j);
    auto it = data_.find(k);
    return (it != data_.end()) ? it->second : 0.0;
}

std::vector<std::pair<int, double>> SparseMatrix::get_neighbors(int i) const {
    std::vector<std::pair<int, double>> result;
    auto adj_it = adj_.find(i);
    if (adj_it == adj_.end()) {
        return result;
    }

    for (int j : adj_it->second) {
        double val = get(i, j);
        if (val != 0.0) {
            result.push_back({j, val});
        }
    }
    return result;
}

void SparseMatrix::clear_row_col(int idx) {
    auto adj_it = adj_.find(idx);
    if (adj_it == adj_.end()) return;

    // Get neighbors to clear
    std::vector<int> neighbors(adj_it->second.begin(), adj_it->second.end());

    // Clear all entries
    for (int j : neighbors) {
        int64_t k = key(idx, j);
        data_.erase(k);

        // Update adjacency of j
        auto& adj_j = adj_[j];
        adj_j.erase(idx);
        if (adj_j.empty()) {
            adj_.erase(j);
        }
    }

    // Remove idx from adjacency
    adj_.erase(idx);
}

// ============================================================================
// Second Derivatives Computation
// ============================================================================

// Helper to encode pair as int64
static inline int64_t encode_pair(int i, int j) {
    int min_idx = (i <= j) ? i : j;
    int max_idx = (i <= j) ? j : i;
    return (static_cast<int64_t>(min_idx) << 32) | static_cast<int64_t>(max_idx);
}

void compute_second_derivatives(
    const Node& node,
    std::unordered_map<int64_t, double>& d2
) {
    d2.clear();

    if (node.op_tag == "mul" && node.parent_ids.size() == 2) {
        int idx0 = node.parent_ids[0];
        int idx1 = node.parent_ids[1];

        if (idx0 == idx1) {
            d2[encode_pair(idx0, idx0)] = 2.0;
        } else {
            d2[encode_pair(idx0, idx1)] = 1.0;
        }
    }
    else if (node.op_tag == "div" && node.parent_ids.size() == 2) {
        int idx0 = node.parent_ids[0];
        int idx1 = node.parent_ids[1];
        double y_val = node.parent_vals[1];

        if (y_val != 0) {
            if (idx0 != idx1) {
                d2[encode_pair(idx0, idx1)] = -1.0 / (y_val * y_val);
            }
            double x_val = node.parent_vals[0];
            d2[encode_pair(idx1, idx1)] = 2.0 * x_val / (y_val * y_val * y_val);
        }
    }
    else if (node.op_tag == "pow" && node.parent_ids.size() == 2) {
        int idx0 = node.parent_ids[0];
        double x_val = node.parent_vals[0];
        double n_val = node.parent_vals[1];

        if (x_val != 0) {
            d2[encode_pair(idx0, idx0)] = n_val * (n_val - 1) * std::pow(x_val, n_val - 2);
        }
    }
    else if (node.op_tag == "exp" && node.parent_ids.size() == 1) {
        int idx0 = node.parent_ids[0];
        double x_val = node.parent_vals[0];
        d2[encode_pair(idx0, idx0)] = std::exp(x_val);
    }
    else if (node.op_tag == "log" && node.parent_ids.size() == 1) {
        int idx0 = node.parent_ids[0];
        double x_val = node.parent_vals[0];
        if (x_val > 0) {
            d2[encode_pair(idx0, idx0)] = -1.0 / (x_val * x_val);
        }
    }
    else if (node.op_tag == "erf" && node.parent_ids.size() == 1) {
        int idx0 = node.parent_ids[0];
        double x_val = node.parent_vals[0];
        const double sqrt_pi = 1.7724538509055159;  // sqrt(pi)
        d2[encode_pair(idx0, idx0)] = -(4.0 * x_val / sqrt_pi) * std::exp(-x_val * x_val);
    }
}

// ============================================================================
// Algorithm 4 Implementation
// ============================================================================

std::vector<std::vector<double>> algo4_cpp_parallel(
    const Tape& tape,
    const std::vector<int>& input_indices,
    int output_idx,
    int n_threads
) {
    // Set number of threads
    omp_set_num_threads(n_threads);

    int n_nodes = tape.nodes.size();

    // Find the maximum variable index to properly size arrays
    int n_vars = output_idx + 1;
    for (const auto& node : tape.nodes) {
        n_vars = std::max(n_vars, node.id + 1);
        for (int pid : node.parent_ids) {
            n_vars = std::max(n_vars, pid + 1);
        }
    }

    // Initialize W matrix and adjoint vector
    SparseMatrix W(n_vars);
    std::vector<double> vbar(n_vars, 0.0);
    vbar[output_idx] = 1.0;

    // Reverse sweep (must be sequential due to dependencies)
    for (int node_idx = n_nodes - 1; node_idx >= 0; --node_idx) {
        const Node& node = tape.nodes[node_idx];
        int i = node.id;

        // Get predecessors
        std::vector<int> preds = node.parent_ids;
        std::sort(preds.begin(), preds.end());
        preds.erase(std::unique(preds.begin(), preds.end()), preds.end());

        // Compute first derivatives
        std::unordered_map<int, double> d1;
        for (size_t k = 0; k < node.parent_ids.size(); ++k) {
            int j = node.parent_ids[k];
            d1[j] += node.parent_derivs[k];
        }

        // Compute second derivatives
        std::unordered_map<int64_t, double> d2;
        compute_second_derivatives(node, d2);

        // PUSHING STAGE - PARALLELIZED!
        auto neighbors = W.get_neighbors(i);
        int n_neighbors = neighbors.size();

        // TEMPORARY: Disable parallelization for debugging
        //#pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int idx = 0; idx < n_neighbors; ++idx) {
            int p = neighbors[idx].first;
            double w_pi = neighbors[idx].second;

            if (p == i) {
                // Case 1: Diagonal W(i,i) contribution
                for (size_t a = 0; a < preds.size(); ++a) {
                    int j = preds[a];
                    double dj = (d1.count(j) > 0) ? d1[j] : 0.0;
                    if (dj == 0.0) continue;

                    for (size_t b = a; b < preds.size(); ++b) {
                        int k = preds[b];
                        double dk = (d1.count(k) > 0) ? d1[k] : 0.0;
                        if (dk != 0.0) {
                            double update_val = dj * dk * w_pi;
                            W.add(j, k, update_val);
                        }
                    }
                }
            } else {
                // Case 2: Off-diagonal W(p,i) contribution
                for (int j : preds) {
                    double dj = (d1.count(j) > 0) ? d1[j] : 0.0;
                    if (dj == 0.0) continue;

                    if (j == p) {
                        W.add(p, p, 2.0 * dj * w_pi);
                    } else {
                        W.add(p, j, dj * w_pi);
                    }
                }
            }
        }

        // Clear row/column i
        W.clear_row_col(i);

        // CREATING STAGE - PARALLELIZED!
        double vbar_i = vbar[i];
        if (vbar_i != 0.0 && !d2.empty()) {
            // Convert to vector for parallel iteration
            std::vector<std::pair<int64_t, double>> d2_vec(d2.begin(), d2.end());
            int n_d2 = d2_vec.size();

            // TEMPORARY: Disable parallelization for debugging
            //#pragma omp parallel for schedule(static) num_threads(n_threads)
            for (int idx = 0; idx < n_d2; ++idx) {
                int64_t key = d2_vec[idx].first;
                double val = d2_vec[idx].second;

                // Decode pair
                int j = static_cast<int>(key >> 32);
                int k = static_cast<int>(key & 0xFFFFFFFF);

                if (k >= j && val != 0.0) {
                    W.add(j, k, vbar_i * val);
                }
            }
        }

        // ADJOINT STAGE (sequential, very fast)
        if (vbar_i != 0.0) {
            for (int j : preds) {
                double dj = (d1.count(j) > 0) ? d1[j] : 0.0;
                if (dj != 0.0) {
                    #pragma omp atomic
                    vbar[j] += vbar_i * dj;
                }
            }
        }
    }

    // Extract Hessian for inputs
    return extract_input_hessian(W, input_indices);
}

std::vector<std::vector<double>> extract_input_hessian(
    const SparseMatrix& W,
    const std::vector<int>& input_indices
) {
    int n = input_indices.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double val = W.get(input_indices[i], input_indices[j]);
            H[i][j] = val;
            if (i != j) {
                H[j][i] = val;
            }
        }
    }

    return H;
}

} // namespace algo4
