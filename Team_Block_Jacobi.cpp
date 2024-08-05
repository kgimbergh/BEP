// 2D stationary Heat equation solver using Jacobi iteration

// \detla u = f(x,y,t)
// Discretization in space gives A*u = b
// LDU decomposition gives
// (A-D+D)*u = b -> Du = b - (A-D)*u
// Jacobi iteration: D u_new = b - (A-D)*u_old
//                   D u_new = D*u_old + b - A*u_old = D*u_old + r_k 

#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using adj_matrix_type = std::vector<std::vector<std::pair<int, int>>>;

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

// Include parts of Kokkos kernels
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosLapack_gesv.hpp>

#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_BsrMatrix.hpp>

#include <KokkosBatched_Vector.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_SolveLU_Decl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>


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
    typedef Kokkos::View<double *, Layout, MemSpace> ViewVectorType;
    typedef Kokkos::View<double **, Layout, MemSpace> ViewMatrixType;
#pragma endregion

using device_type = Kokkos::Device<ExecSpace, MemSpace>;
using crs_matrix_type = KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void, Offset>;
using bsr_matrix_type = KokkosSparse::Experimental::BsrMatrix<Scalar, Ordinal, device_type, void, Offset>;

// -1 = self, 0 = right, 1 = left, 2 = up, 3 = down
// Create A independently of the size n, but just of size (block_size * num_blocks)x(block_size * num_blocks)
crs_matrix_type Create_A(int n, 
                         double h){

    using graph_type = typename crs_matrix_type::staticcrsgraph_type;
    using row_map_type = typename graph_type::row_map_type;
    using entries_type = typename graph_type::entries_type;
    using values_type = typename crs_matrix_type::values_type;

    // const Offset numNNZ = (n-2)*(n-2)*4 + (n-2)*2*3 + 2*(n-2)*3 + 2*2*2; // Number of non-zero entries for LU   
    const Offset numNNZ = (n-2)*(n-2)*5 + (n-2)*2*4 + 2*(n-2)*4 + 2*2*3; // Number of non-zero entries for A   
    typename row_map_type::non_const_type row_map("row pointers", n*n + 1);
    typename entries_type::non_const_type entries("column indices", numNNZ);
    typename values_type::non_const_type values("values", numNNZ);

    // Build the row pointers and store numNNZ
    typename row_map_type::HostMirror row_map_h = Kokkos::create_mirror_view(row_map);

    Ordinal numRows = n*n;

    row_map_h(0) = 0;
    for (Ordinal rowIdx = 1; rowIdx < (n*n) + 1; rowIdx++) {
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

    for (Ordinal rowIdx = 0; rowIdx < (n*n); rowIdx++) {
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

crs_matrix_type Create_R(int n, 
                         int num_blocks, 
                         int block_size, 
                         int overlap){
        using graph_type = typename crs_matrix_type::staticcrsgraph_type;
        using row_map_type = typename graph_type::row_map_type;
        using entries_type = typename graph_type::entries_type;
        using values_type = typename crs_matrix_type::values_type;
    
        const Offset numNNZ = num_blocks*num_blocks * block_size*block_size; // Number of non-zero entries for R
        typename row_map_type::non_const_type row_map("row pointers", num_blocks*num_blocks * block_size*block_size + 1);
        typename entries_type::non_const_type entries("column indices", numNNZ);
        typename values_type::non_const_type values("values", numNNZ);
    
        // Build the row pointers and store numNNZ
        typename row_map_type::HostMirror row_map_h = Kokkos::create_mirror_view(row_map);
    
        Ordinal numRows = row_map.extent(0) - 1;
    
        row_map_h(0) = 0;
        for (Ordinal rowIdx = 1; rowIdx < numRows + 1; rowIdx++) {
            row_map_h(rowIdx) = row_map_h(rowIdx-1) + 1;
        }
    
        Kokkos::deep_copy(row_map, row_map_h);
    
        // Build the matrix
        typename entries_type::HostMirror entries_h = Kokkos::create_mirror_view(entries);
        typename values_type::HostMirror values_h = Kokkos::create_mirror_view(values);
    
        for (Ordinal blockIdx = 0; blockIdx < num_blocks*num_blocks; blockIdx++) {
            int block_x = blockIdx % num_blocks;
            int block_y = blockIdx / num_blocks;

            for (Ordinal rowIdx = 0; rowIdx < block_size*block_size; rowIdx++) {
                int local_x = rowIdx % block_size;
                int local_y = rowIdx / block_size;

                int global_index = (block_x*block_size - block_x*overlap + local_x) + (block_y*block_size - block_y*overlap + local_y) * n;
                entries(blockIdx*block_size*block_size + rowIdx) = global_index;
                values(blockIdx*block_size*block_size + rowIdx) = 1.0;
            }
        }
    
        Kokkos::deep_copy(entries, entries_h);
        Kokkos::deep_copy(values, values_h);
    
        graph_type graph(entries, row_map);
        crs_matrix_type R("R", n*n, values, graph);
    
        return R;
}

crs_matrix_type Create_R_tilde(crs_matrix_type R,
                               int n){
    // Get data from R
    auto R_values = R.values;
    auto R_graph = R.graph;
    auto R_row_map = R_graph.row_map;
    auto R_entries = R_graph.entries;

    // Create new values, row map and entries for R_tilde
    using graph_type = typename crs_matrix_type::staticcrsgraph_type;
    using row_map_type = typename graph_type::row_map_type;
    using entries_type = typename graph_type::entries_type;
    using values_type = typename crs_matrix_type::values_type;

    typename row_map_type::non_const_type R_tilde_row_map("row pointers", R_row_map.extent(0));
    typename entries_type::non_const_type R_tilde_entries("column indices", R_entries.extent(0));
    typename values_type::non_const_type R_tilde_values("values", R_values.extent(0));

    // Copy the row_map and entries of R to R_tilde
    Kokkos::deep_copy(R_tilde_row_map, R_row_map);
    Kokkos::deep_copy(R_tilde_entries, R_entries);

    // For each value in entries, get the number of occurrences in entries, and save at which row this happens.
    std::vector<std::vector<int>> occurences(n*n);

    int row_num = 0;
    for (int i = 0; i < R_entries.extent(0); i++)
    {
        // Get current row index
        if (i >= R_row_map(row_num + 1))
        {
            row_num++;
        }

        occurences[R_entries(i)].push_back(row_num);
    }

    int max_occurence = 0;

    // For each value in values, divide by the number of occurrences
    for (int i = 0; i < R_values.extent(0); i++)
    {
        R_tilde_values(i) = R_values(i) / occurences[R_entries(i)].size();

	if (occurences[R_entries(i)].size() > max_occurence) {
		max_occurence = occurences[R_entries(i)].size();
	}
    }

    printf("%d\n", max_occurence);

    // Now use this information to construct the transpose of R_tilde
    typename row_map_type::non_const_type R_T_tilde_row_map("row pointers", n*n + 1);
    typename entries_type::non_const_type R_T_tilde_entries("column indices", R_values.extent(0));
    typename values_type::non_const_type R_T_tilde_values("values", R_values.extent(0));

    // The row map is based on the number of occurrences of each value in R_entries
    R_T_tilde_row_map(0) = 0;
    for (int i = 1; i < R_T_tilde_row_map.extent(0) + 1; i++)
    {
        R_T_tilde_row_map(i) = R_T_tilde_row_map(i-1) + occurences[i-1].size();
    }

    // The entries map is contructed from the occurences
    for (int i = 0; i < n*n; i++)
    {
        for (int j = 0; j < occurences[i].size(); j++)
        {
            R_T_tilde_entries(R_T_tilde_row_map(i) + j) = occurences[i][j];
            R_T_tilde_values(R_T_tilde_row_map(i) + j) = 1.0/occurences[i].size();
        }
    }
    
    // Create a new graph
    graph_type R_tilde_graph(R_T_tilde_entries, R_T_tilde_row_map);

    // Create a new crs_matrix_type with the new values
    crs_matrix_type R_tilde("R_tilde", R_row_map.extent(0) - 1, R_T_tilde_values, R_tilde_graph);

    return R_tilde;
}



bsr_matrix_type Create_A_hat(int num_blocks, 
                             int block_size,
                             double h){
    // A contains 2D laplacian matrices of size block_size*block_size x block_size*block_size on the diagonal
    // We create this matrix directly in block format

    // Create an A matrix with n = block_size
    crs_matrix_type A_block = Create_A(block_size, h);

    bsr_matrix_type A_reallyblock(A_block, block_size*block_size);

    // Get block matrix data
    auto A_reallyblock_values = A_reallyblock.values;

    using graph_type = typename crs_matrix_type::staticcrsgraph_type;
    using row_map_type = typename graph_type::row_map_type;
    using entries_type = typename graph_type::entries_type;
    using values_type = typename crs_matrix_type::values_type;

    // There are num_blocks*num_blocks blocks in the matrix
    // All of them are the same block_size*block_size x block_size*block_size matrix

    const Offset numNNZ = num_blocks*num_blocks * block_size*block_size * block_size*block_size;
    typename row_map_type::non_const_type row_map("row pointers", num_blocks*num_blocks + 1);
    typename entries_type::non_const_type entries("column indices", num_blocks*num_blocks);
    typename values_type::non_const_type values("values", numNNZ);

    // Build the row pointers and store numNNZ
    typename row_map_type::HostMirror row_map_h = Kokkos::create_mirror_view(row_map);

    Ordinal numRows = num_blocks*num_blocks;

    row_map_h(0) = 0;
    for (Ordinal rowIdx = 1; rowIdx < numRows + 1; rowIdx++) {
        row_map_h(rowIdx) = row_map_h(rowIdx-1) + 1;
    }

    Kokkos::deep_copy(row_map, row_map_h);

    // Build the matrix
    typename entries_type::HostMirror entries_h = Kokkos::create_mirror_view(entries);
    typename values_type::HostMirror values_h = Kokkos::create_mirror_view(values);

    for (Ordinal blockIdx = 0; blockIdx < num_blocks*num_blocks; blockIdx++) {
        for (Ordinal rowIdx = 0; rowIdx < block_size*block_size * block_size*block_size; rowIdx++) {
            values_h(blockIdx * (block_size*block_size * block_size*block_size) + rowIdx) = A_reallyblock_values(rowIdx);
        }
        entries_h(blockIdx) = blockIdx;
    }

    Kokkos::deep_copy(entries, entries_h);
    Kokkos::deep_copy(values, values_h);

    graph_type graph(entries, row_map);

    bsr_matrix_type A_hat("A_hat", numRows, values, graph, block_size*block_size);

    return A_hat;
}


template <typename AlgoTagType>
struct Functor_BatchedTeamLU{
    bsr_matrix_type block_matrix;

    Functor_BatchedTeamLU(bsr_matrix_type block_matrix_) 
    : block_matrix(block_matrix_) {}

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
        const int i = member.league_rank();
        ViewMatrixType block = block_matrix.unmanaged_block(i);

        KokkosBatched::TeamLU<MemberType, AlgoTagType>::invoke(member, block);
    }
};


template <typename TransType, typename AlgoTagType>
struct Functor_BatchedTeamSolveLU{
    bsr_matrix_type D;
    ViewVectorType U;

    Functor_BatchedTeamSolveLU(bsr_matrix_type D_, ViewVectorType U_)
    : D(D_), U(U_) {}

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
        const int i = member.league_rank();
        auto block_D = D.unmanaged_block(i);
        auto block_U = Kokkos::subview(U, std::pair<int, int>(i*D.blockDim(), (i+1)*D.blockDim()));

        KokkosBatched::TeamSolveLU<MemberType, TransType, AlgoTagType>::invoke(member, block_D, block_U);

    }
};


void Jacobi(int n, 
            double h, 
            ViewVectorType u, 
            ViewVectorType b, 
            crs_matrix_type A,
            bsr_matrix_type A_hat,
            crs_matrix_type R,
            crs_matrix_type R_tilde,
            int block_size, 
            int num_blocks,
	    int team_size,
            std::string problem)
            
            {
    
    Kokkos::Timer jacobi_timer;

    // Perform LU decomposition on the diagonal blocks
    using AlgoTagType = KokkosBatched::Algo::LU::Blocked;
    using TransType = KokkosBatched::Trans::NoTranspose;
    
    using policy_type = Kokkos::TeamPolicy<ExecSpace>;
    const int league_size = num_blocks*num_blocks;
    policy_type policy(league_size, Kokkos::AUTO);
    if (team_size != 0){
	    policy = policy_type(league_size, team_size);
    }

    Kokkos::parallel_for("LU decomposition", policy, 
                         Functor_BatchedTeamLU<AlgoTagType>(A_hat));


    // Iteration parameters
    double tolerance = 1e-4;
    double residual = 1;
    int iter = 0;

    // Vector to store the error
    std::vector<double> evec;

    // Initialize residual vector r_k
    ViewVectorType r_k("residual", n*n);

    // Initialize extended residual vector r_k
    ViewVectorType r_k_extended("residual_extended", num_blocks*num_blocks*block_size*block_size);

    // Initialize residual matrix r of size (num_blocks*numblocks, block_size*block_size)
    ViewMatrixType r("residual_matrix", num_blocks*num_blocks, block_size*block_size);

    double initial_time = jacobi_timer.seconds();
    printf("%f\n", initial_time);

    // Jacobi iteration: D u_k+1 = b - (A-D) u_k
    //                           = D u_k + (b-A u_k)
    //                     u_k+1 = u_k + D^-1 (b - A u_k)
    //                           = u_k + sum(R_T_tilde * A_hat^-1 * R * r_k)  
    Kokkos::Timer iteration_timer;
    while (residual > tolerance)
    {
        // Calculate residual vector
        // r_k = b - A * u
	
	Kokkos::parallel_for("Calculate residual", range_policy(0, n*n), KOKKOS_LAMBDA(const int i) {
            double sum = 0;
            int row_start = A.graph.row_map(i);
            int row_end = A.graph.row_map(i + 1);

            for (int j = row_start; j < row_end; j++)
            {
                int col = A.graph.entries(j);
                sum += A.values(j) * u(col);
            }

            // update error vector
            r_k(i) = b(i) - sum;
        });

        // Calculate the residual
        residual = KokkosBlas::nrm2(r_k)/(n*n);

	if (iter == 0){
		printf("%f\n", residual);
	}

        // Update error vector
        evec.push_back(residual);

        // The following is the iteration:
        // u_new = u_old + R_T_tilde * A_hat^-1 * R * r_k

        // Calculate r_k_extended = R * r_k

        Kokkos::parallel_for("Extend residual", range_policy(0, num_blocks*num_blocks), KOKKOS_LAMBDA(const int i) {
            // Every row of R contains exactly one non-zero element
            for (int j = 0; j < block_size*block_size; j++)
            {
                r_k_extended(i*block_size*block_size + j) = r_k(R.graph.entries(i*block_size*block_size + j));
            }
        });
            
        // Solve the linear system A_hat x = r_k using forward backward substitution
        Kokkos::parallel_for("Solve LU systems", policy, 
                             Functor_BatchedTeamSolveLU<TransType, AlgoTagType>(A_hat, r_k_extended));


        // Calculate r_k = R_T_tilde * r_k_extended
        Kokkos::parallel_for("Reduce residual", range_policy(0, n*n), KOKKOS_LAMBDA(const int i){
            double sum = 0;
            int row_start = R_tilde.graph.row_map(i);
            int row_end = R_tilde.graph.row_map(i + 1);

            for (int j = row_start; j < row_end; j++)
            {
                int col = R_tilde.graph.entries(j);
                sum += R_tilde.values(j) * r_k_extended(col);
            }

            r_k(i) = sum;
        });
        
        
        
        // Update u = u + x
        KokkosBlas::axpby(1.0, r_k, 1.0, u); 

        iter++;

	double iteration_time = iteration_timer.seconds();

	// if (iter % 1000 == 0){
	// 	printf("%d %f\n", iter, residual);
	// }

	// if (iter % 30000 == 0){
	// 	break;
	//}

    }

    double iteration_time = iteration_timer.seconds();

    // Print the number of iterations
    printf("%d\n", iter);

    // Print the residual to check
    printf("%f\n", residual);

    Kokkos::fence();
    
}


ViewVectorType create_b(int n, 
                   double h,
                   std::string problem){
    ViewVectorType b("b", n*n);
    Kokkos::parallel_for(range_policy(0, n), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < n; j++){
            int x = (i * n + j) % n;
            int y = (i * n + j) / n;

            if (problem == "sines") {
                b(i*n + j) = - 2 * M_PI * M_PI * sin(M_PI * (x+1) * h) * sin(M_PI * (y+1) * h);
            }
            else {
                printf("Problem not recognized\n");
            }
        }
    });

    return b;
}


ViewVectorType initial_guess(int n,
                             std::string problem){
    // Initialize u_old
    ViewVectorType u_start("u_start", n*n);

    Kokkos::Random_XorShift64_Pool<> rand_pool(std::time(NULL));

    Kokkos::parallel_for(range_policy(0, n), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < n; j++){
            // provide seed value for random number generator
            auto rand_gen = rand_pool.get_state();
            
            if (problem == "sines"){
                // Initialize u_old with exact solution
                // u_start(i * n + j) = rand_gen.drand() * 10;
                u_start(i * n + j) = 1;
            } 
            
            else {
                printf("Problem not recognized\n");
            }
        }
    });                       
    return u_start;
}


int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        // Grid parameters
        double L = 1;       // length of the domain in both directions
	int n = 3;
	int block_size = 2;
	int num_blocks = 2;
	int team_size = 0;

	int numThreads = ExecSpace().concurrency();
	printf("%d\n", numThreads);


	for ( int i = 0; i < argc; i++ ) {
		if ( ( strcmp( argv[ i ], "-n" ) == 0 ) ) {
      		n = atof(argv[ ++i ]);
		}
		else if ( ( strcmp( argv[ i ], "-block_size" ) == 0 ) ) {
                block_size = atof(argv[ ++i ]);
                }
		else if ( ( strcmp( argv[ i ], "-num_blocks" ) == 0 ) ) {
                num_blocks = atof(argv[ ++i ]);
                }
		else if ( ( strcmp( argv[ i ], "-team_size" ) == 0 ) ) {
                team_size = atof(argv[ ++i ]);
                }

	}

	printf("%d\n", n);
	printf("%d\n", block_size);
	printf("%d\n", num_blocks);

	if ((block_size * num_blocks - n)%(num_blocks - 1) != 0){
		printf("Configuration of n, block_size, and num_blocks not possible\n");
		return 0;
	}

        int overlap = (block_size * num_blocks - n) / (num_blocks - 1);
        printf("%d\n", overlap);

        // Start timer
        Kokkos::Timer timer_global;

        double h = L / (n+1); // grid spacing

        // Create 2D discretization matrix A
        crs_matrix_type A = Create_A(n, h);

        // Choose problem
        std::string problem = "sines";

        // Initialize rhs vector
        ViewVectorType b = create_b(n, h, problem);

        // Initialize intitial guess
        ViewVectorType u_start = initial_guess(n, problem);

        // Create R matrix
        crs_matrix_type R = Create_R(n, num_blocks, block_size, overlap);

        // Create R_tilde matrix
        crs_matrix_type R_tilde = Create_R_tilde(R, n);

        // Create A_hat matrix
        bsr_matrix_type A_hat = Create_A_hat(num_blocks, block_size, h);

        // Jacobi iteration
        Jacobi(n, h, u_start, b, A, A_hat, R, R_tilde, block_size, num_blocks, team_size, problem);

        double time_global = timer_global.seconds();

        printf("%f\n", time_global);

    }

    Kokkos::finalize();

}
