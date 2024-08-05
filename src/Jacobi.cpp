// 2D stationary Heat equation solver using Jacobi iteration

// \detla u = f(x,y,t)
// Discretization in space gives A*u = b
// LDU decomposition gives
// (L+D+U)*u = b -> Du = b - (L+U)*u
// Jacobi iteration: u_new = D^(-1)*(b - (L+U)*u_old)

#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

// Include parts of Kokkos kernels
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_nrm2.hpp>

#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

using Scalar  = default_scalar;
using Ordinal = default_lno_t;
using Offset  = default_size_type;

#pragma region Kokkos_properties
// Choose Execution Space.       Default is Threads
// typedef Kokkos::Serial   ExecSpace;          // Serial execution space.
typedef Kokkos::Threads ExecSpace; // Parallel execution space.
// typedef Kokkos::OpenMP   EexecSpace;         // OpenMP execution space.
// typedef Kokkos::Cuda     ExecSpace;          // Cuda execution space.

// Choose device memory space.       Default is HostSpace
typedef Kokkos::HostSpace MemSpace; // Host memory space.
// typedef Kokkos::CudaSpace     MemSpace;      // Cuda memory space.
// typedef Kokkos::CudaUVMSpace  MemSpace;      // Cuda UVM memory space.

// Choose memory Layout.         Default is LayoutRight
// typedef Kokkos::LayoutLeft   Layout;            // Column major / column wise
typedef Kokkos::LayoutRight Layout; // Row major / row wise (preferred)

// Use a RangePolicy.
typedef Kokkos::RangePolicy<ExecSpace> range_policy; // Specifies how to parallelize a loop. In this case, it will use the parallel_for function.

// Allocate y, x vectors and Matrix A on device.
// EXERCISE: Use MemSpace and Layout.
typedef Kokkos::View<double *, MemSpace> ViewVectorType;
typedef Kokkos::View<double **, Layout, MemSpace> ViewMatrixType;
#pragma endregion

using device_type = Kokkos::Device<ExecSpace, MemSpace>;
using crs_matrix_type = KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void, Offset>;

crs_matrix_type Create_A(int n, int N, double h){

    using graph_type = typename crs_matrix_type::staticcrsgraph_type;
    using row_map_type = typename graph_type::row_map_type;
    using entries_type = typename graph_type::entries_type;
    using values_type = typename crs_matrix_type::values_type;

    // const Offset numNNZ = (n-2)*(n-2)*4 + (n-2)*2*3 + 2*(n-2)*3 + 2*2*2; // Number of non-zero entries for LU   
    const Offset numNNZ = (n-2)*(n-2)*5 + (n-2)*2*4 + 2*(n-2)*4 + 2*2*3; // Number of non-zero entries for A   
    typename row_map_type::non_const_type row_map("row pointers", N + 1);
    typename entries_type::non_const_type entries("column indices", numNNZ);
    typename values_type::non_const_type values("values", numNNZ);

    // Build the row pointers and store numNNZ
    typename row_map_type::HostMirror row_map_h = Kokkos::create_mirror_view(row_map);

    Ordinal numRows = N;

    row_map_h(0) = 0;
    for (Ordinal rowIdx = 1; rowIdx < N + 1; rowIdx++) {
        int blocknum = (rowIdx-1) / n;
        bool isFirstBlock = (blocknum == 0);
        bool isLastBlock = (blocknum == n-1); 
        bool isFirstRow = ((rowIdx-1) % n == 0);
        bool isLastRow = ((rowIdx-1) % n == n-1);

        if (isFirstBlock || isLastBlock) {
            if (isFirstRow || isLastRow) {
                // row_map_h(rowIdx) = row_map_h(rowIdx-1) + 2; // for LU
                row_map_h(rowIdx) = row_map_h(rowIdx-1) + 3; // for A
            } else {
                // row_map_h(rowIdx) = row_map_h(rowIdx-1) + 3; // for LU
                row_map_h(rowIdx) = row_map_h(rowIdx-1) + 4; // for A
            } 
        } else {
            if (isFirstRow || isLastRow) {
                // row_map_h(rowIdx) = row_map_h(rowIdx-1) + 3; // for LU
                row_map_h(rowIdx) = row_map_h(rowIdx-1) + 4; // for A
            } else {
                // row_map_h(rowIdx) = row_map_h(rowIdx-1) + 4; // for LU
                row_map_h(rowIdx) = row_map_h(rowIdx-1) + 5; // for A
            } 
        }
    }

    Kokkos::deep_copy(row_map, row_map_h);

    // Build the matrix
    typename entries_type::HostMirror entries_h = Kokkos::create_mirror_view(entries);
    typename values_type::HostMirror values_h = Kokkos::create_mirror_view(values);

    for (Ordinal rowIdx = 0; rowIdx < N; rowIdx++) {
        int blocknum = rowIdx / n;
        bool isFirstBlock = (blocknum == 0);
        bool isLastBlock = (blocknum == n-1); 
        bool isFirstRow = (rowIdx % n == 0);
        bool isLastRow = (rowIdx % n == n-1);

        std::vector<Ordinal> entryIndices;
        std::vector<Scalar> entryValues;

        // Off diagonal n to the left
        if (!isFirstBlock) {
            entryIndices.push_back(rowIdx-n);
            entryValues.push_back(1/(h*h));
        }

        // Off diagonal 1 to the left
        if (!isFirstRow) {
            entryIndices.push_back(rowIdx-1);
            entryValues.push_back(1/(h*h));
        }

        // Diagonal for A
        entryIndices.push_back(rowIdx);
        entryValues.push_back(-4/(h*h)); 

        // Off diagonal 1 to the right
        if (!isLastRow) {
            entryIndices.push_back(rowIdx+1);
            entryValues.push_back(1/(h*h));
        }

        // Off diagonal n to the right
        if (!isLastBlock) {
            entryIndices.push_back(rowIdx+n);
            entryValues.push_back(1/(h*h));
        }

        // Assign to entries and values
        for (size_t i = 0; i < entryIndices.size(); ++i) {
            entries_h(row_map_h(rowIdx)+i) = entryIndices[i];
            values_h(row_map_h(rowIdx)+i) = entryValues[i];
        }
    }

    Kokkos::deep_copy(entries, entries_h);
    Kokkos::deep_copy(values, values_h);

    graph_type graph(entries, row_map);
    crs_matrix_type A("A", numRows, values, graph);

    return A;
}

std::vector<double> Jacobi(int n, 
                           double h, 
                           ViewVectorType u_old, 
                           ViewVectorType b, 
                           crs_matrix_type A, 
                           double D){
    Kokkos::Timer jacobi_timer;

    // Iteration parameters
    double tolerance = 1e-4;
    double residual = 1;

    int iter = 0;

    // Intialize u_new
    ViewVectorType u_new("u_new", n * n);

    // Initialize residual vector
    ViewVectorType r_k("evec", n * n);

    // Initialize error vector
    std::vector<double> evec;

    Kokkos::parallel_for(range_policy(0, n * n), KOKKOS_LAMBDA(const int i) { r_k(i) = 0; });

    double initial_time = jacobi_timer.seconds();

    // Print iteration time
    printf("%f\n", initial_time);

    // Calculate the initial residual
    // Calculate residual vector
    // r_k = - A * u
    KokkosSparse::spmv("N", -1.0, A, u_old, 0.0, r_k);

    // r_k = r_k + b
    KokkosBlas::axpby(1.0, b, 1.0, r_k);

    // Calculate the error
    residual = KokkosBlas::nrm2(r_k)/(n*n);

    // Update error vector
    evec.push_back(residual);

    // Print the initial residual
    printf("%f\n", residual);

    double start_time = jacobi_timer.seconds();

    // Jacobi iteration: u_k+1 = u_k + D^(-1) * r_k, r_k = b - A*u_k
    while (residual > tolerance)
    {
        residual = 0;

        // Update u_new
        Kokkos::parallel_for(range_policy(0, n * n), KOKKOS_LAMBDA(const int i) {
            // Calculate residual vector
            double sum = 0;
            int row_start = A.graph.row_map(i);
            int row_end = A.graph.row_map(i + 1);

            for (int j = row_start; j < row_end; j++)
            {
                int col = A.graph.entries(j);
                sum += A.values(j) * u_old(col);
            }

            // update error vector
            r_k(i) = b(i) - sum;

            // Update u_new
            u_new(i) = u_old(i) + (1 / D) * r_k(i);
        });

        // Calculate error
        residual = KokkosBlas::nrm2(r_k)/(n*n);

        iter++;

        // // Update u_old
        KokkosBlas::axpby(1.0, u_new, 0.0, u_old); // Update u_old (u_old = u_new + 0*u_old)

        // Update error vector
        evec.push_back(residual);
    }

    // Print the number of iterations
    printf("%d\n", iter);

    printf("[");
    // Print the error vector
    for (int i = 0; i < evec.size(); i++){
	    printf("%f,", evec[i]);
    }
    printf("]");

    double time = jacobi_timer.seconds() - start_time;

    std::vector<double> return_values(2);

    return_values[0] = time;
    return_values[1] = iter;

    return return_values;

}

ViewVectorType rhs(int n, 
                   double h){
    ViewVectorType b("b", n * n);
    Kokkos::parallel_for(range_policy(0, n * n), KOKKOS_LAMBDA(const int i) {
        int x = i % n;
        int y = i / n;

        b(i) = - 2 * M_PI * M_PI * sin(M_PI * (x+h) * h) * sin(M_PI * (y+h) * h);
    });
    return b;
}

ViewVectorType initial_guess(int n){
    // Initialize u_old and u_new
    ViewVectorType u_old("u_old", n * n);
    ViewVectorType u_new("u_new", n * n);

    Kokkos::parallel_for(range_policy(0, n * n), KOKKOS_LAMBDA(const int i) {
        //u_old(i) = (std::chrono::system_clock::now().time_since_epoch().count() % 100)/100.0;
	u_old(i) = 1;
        u_new(i) = 1;
    });

    return u_old;
}


int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {       
	 // Grid parameters
        double L = 1;       // length of the domain in both directions
        int n = 2;              // number of inner grid points in one direction
	
        for (int i = 0; i < argc; i++){
            if ( ( strcmp( argv[ i ], "-n" ) == 0 ) ) {
                n = atoi( argv[ ++i ] ) ;
            }
        }

	int numThreads = ExecSpace().concurrency();
        printf("%d\n", numThreads);

        // Print n
        printf("%d\n", n);

        // Print bs (NA)
        int bs = 0;
        printf("%d\n", bs);

        // Print nb (NA)
        int nb = 0;
        printf("%d\n", nb);

        // Print o (NA)
        int o = 0;
        printf("%d\n", o);

        int N = n * n;      // number of grid points

        double h = L / (n+1); // grid spacing

        Kokkos::Timer timer_global;

        // Create 2D discretization matrix A
        crs_matrix_type A = Create_A(n, N, h);

        // Set variable D for Jacobi iteration
        double D = -4/(h*h);        

        // Jacobi iteration
        std::vector<double> times;
        std::vector<double> iterations;
        std::vector<double> initial_errors;

        ViewVectorType init_residual("init_residual", n * n);

        int init_error = 1;

        // Initialize rhs vector
        ViewVectorType b = rhs(n, h);

        // Initialize intitial guess
        ViewVectorType u_old = initial_guess(n);

        // Calculate the initial error
        // Calculate residual vector
        // r_k = - A * u
        KokkosSparse::spmv("N", -1.0, A, u_old, 0.0, init_residual); 

        // r_k = r_k + b
        KokkosBlas::axpby(1.0, b, 1.0, init_residual);
    
        // Calculate the error
        init_error = KokkosBlas::nrm2(init_residual)/(n*n);

        // Update error vector
        initial_errors.push_back(init_error);

        std::vector<double> res = Jacobi(n, h, u_old, b, A, D);

        times.push_back(res[0]);
        iterations.push_back(res[1]);

        double time_global = timer_global.seconds();

        // Print the total time
        printf("%f\n", time_global);

    }

    Kokkos::finalize();
}
