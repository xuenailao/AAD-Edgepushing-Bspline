#!/bin/bash
# Compile Cython-optimized modules for Edge-Pushing AAD
#
# Usage:
#   bash compile_cython.sh
#
# This compiles:
# 1. symm_sparse_adjlist_cpp (C++ sparse matrix)
# 2. algo4_sparse_openmp (Cython-optimized sparse algo4)

set -e  # Exit on error

echo "========================================================================"
echo "Compiling Cython modules for Edge-Pushing AAD"
echo "========================================================================"

cd aad_edge_pushing/edge_pushing

# Step 1: Compile C++ sparse matrix
echo ""
echo "[1/2] Compiling symm_sparse_adjlist_cpp..."
python setup_cython_cpp.py build_ext --inplace
if [ $? -eq 0 ]; then
    echo "✓ symm_sparse_adjlist_cpp compiled successfully"
else
    echo "✗ Failed to compile symm_sparse_adjlist_cpp"
    exit 1
fi

# Step 2: Compile Cython sparse algo4
echo ""
echo "[2/2] Compiling algo4_sparse_openmp..."
python setup_sparse_openmp.py build_ext --inplace
if [ $? -eq 0 ]; then
    echo "✓ algo4_sparse_openmp compiled successfully"
else
    echo "✗ Failed to compile algo4_sparse_openmp"
    exit 1
fi

cd ../..

echo ""
echo "========================================================================"
echo "Compilation complete!"
echo ""
echo "You can now use --cython flag:"
echo "  python test_ep_vs_bumping_sparse.py --configs \"6x6\" --cython"
echo "========================================================================"
