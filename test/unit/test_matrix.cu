#include <numeric>
#include <random>
#include <vector>

//#include <cuda.h>
#include <hip/hip_runtime_api.h>

#include <arbor/math.hpp>

#include "algorithms.hpp"
#include "matrix.hpp"
#include "memory/memory.hpp"
#include "util/span.hpp"

#include "backends/gpu/cuda_common.hpp"
#include "backends/gpu/matrix_state_flat.hpp"
#include "backends/gpu/matrix_state_interleaved.hpp"
#include "backends/gpu/matrix_interleave.hpp"
#include "backends/gpu/matrix_state_fine.hpp"

#include "../gtest.h"
#include "common.hpp"


using namespace arb;

using gpu::impl::npos;
using util::make_span;
using util::assign_from;
using memory::on_gpu;
using memory::on_host;

using testing::seq_almost_eq;

using std::begin;
using std::end;


// Test the flat_to_interleaved and interleaved_to_flat operations for the
// set of matrices defined by sizes and starts.
// Applies the interleave to the vector in values, and checks this against
// a reference result generated using a host side reference implementation.
// Then the interleave result is reverse_interleaved, and the result is
// compared to the original input.
//
// This is implemented in a separate function to facilitate testing on a
// broad range of BlockWidth and LoadWidth compile time parameters.
template <typename T, typename I, int BlockWidth, int LoadWidth>
::testing::AssertionResult test_interleave(
        std::vector<I> sizes,
        std::vector<I> starts,
        std::vector<T> values,
        int padded_size, 
        bool run=true)
{
    auto num_mtx = sizes.size();

    auto in       = on_gpu(memory::make_const_view(values));
    auto sizes_d  = on_gpu(memory::make_const_view(sizes));
    auto starts_d = on_gpu(memory::make_const_view(starts));

    int packed_size = padded_size * BlockWidth * gpu::impl::block_count(num_mtx, BlockWidth);
 
    // Create GPU vector and fill it 
    auto forward = memory::device_vector<T>(packed_size, npos<T>());

    hipDeviceSynchronize(); /// <--------------------------------------

    gpu::flat_to_interleaved<T, I, BlockWidth, LoadWidth>(in.data(), forward.data(), sizes_d.data(), starts_d.data(), padded_size, num_mtx);

    hipDeviceSynchronize(); /// <--------------------------------------

    // Create and fill host vector from GPU vector

    std::vector<T> result_f = assign_from(on_host(forward));

    hipDeviceSynchronize(); /// <--------------------------------------

    // CHECK RESULTS
    std::vector<T> expected = gpu::flat_to_interleaved(values, sizes, starts, BlockWidth, num_mtx, padded_size);

    hipDeviceSynchronize(); /// <--------------------------------------

    const auto forward_success = (result_f==expected);
    if (!forward_success) {
        std::cout << "  FAIL :(" << std::endl;
        return ::testing::AssertionFailure() << "interleave to flat failed: BlockWidth "
            << BlockWidth << ", LoadWidth " << LoadWidth << "\n";
    }

    // Create and fill host vector from GPU vector

    hipDeviceSynchronize(); /// <--------------------------------------

    auto backward = memory::device_vector<T>(values.size(), npos<T>());

    hipDeviceSynchronize(); /// <--------------------------------------

    gpu::interleaved_to_flat<T, I, BlockWidth, LoadWidth>(forward.data(), backward.data(), sizes_d.data(), starts_d.data(), padded_size, num_mtx);

    hipDeviceSynchronize(); /// <--------------------------------------

    // Create and fill host vector from GPU vector
    std::vector<T> result_b = assign_from(on_host(backward)); 

    hipDeviceSynchronize(); /// <--------------------------------------

    // we expect that the result of the reverse permutation is the original input vector
    const auto backward_success = (result_b==values);
    if (!backward_success) {
        std::cout << "  FAIL :(" << std::endl;
        return ::testing::AssertionFailure() << "flat to interleave failed: BlockWidth "
            << BlockWidth << ", LoadWidth " << LoadWidth << "\n";
    }
    return ::testing::AssertionSuccess();
}
__global__
void copy(int* in, int* out, unsigned size) {
    const unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size) {
       out[tid] = in[tid]; 
    }
}

__global__
void update(int* in, int* out, unsigned size) {
    const unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < size) {
       out[tid] = in[tid]*171; 
    }
}

TEST(matrix, basic) {
    unsigned size = 1000; 
    // start on host 
    std::vector<int> host_init(size); 
    for (unsigned i = 0; i < size; ++i) {
        host_init[i] = i; 
    }

    // copy to device
    auto device_init  = on_gpu(memory::make_const_view(host_init));

    // copy from device to device
    auto device_inter = memory::device_vector<int>(size);
    auto device_final = memory::device_vector<int>(size);
    copy<<<32, 32>>>  (device_init.data() , device_inter.data(), size);
    update<<<32, 32>>>(device_inter.data(), device_final.data(), size);

    // copy from device to host
    std::vector<int> host_final = assign_from(on_host(device_final));
    bool pass = true;
    for (unsigned i =0; i < 1000; i++) {
        if (host_final[i] != i*171) {
            pass = false; 
            break;
        }
    }
    EXPECT_TRUE(pass);
 
    std::cout << "SUCCESS" << std::endl;
}

template<unsigned threads>
__global__
void copy_shared(int* in, int* out, unsigned size) {
    const unsigned lid = threadIdx.x;
    const unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ int local[threads];
    if (tid < size) {
       local[lid] = in[tid]
       __syncthreads();
       out[tid] = local[lid]; 
       __syncthreads();
    }
}

TEST(matrix, basic2) {
    unsigned size = 1000; 
    // start on host 
    std::vector<int> host_init(size); 
    for (unsigned i = 0; i < size; ++i) {
        host_init[i] = i; 
    }

    // copy to device
    auto device_init  = on_gpu(memory::make_const_view(host_init));

    // copy from device to device
    auto device_final = memory::device_vector<int>(size);
    copy_shared<36> <<<32, 36>>> (device_init.data() , device_final.data(), size);

    // copy from device to host
    std::vector<int> host_final = assign_from(on_host(device_final));
    bool pass = true;
    for (unsigned i =0; i < 1000; i++) {
        if (host_final[i] != i) {
            pass = false; 
            break;
        }
    }
    EXPECT_TRUE(pass);
 
    std::cout << "SUCCESS" << std::endl;
}

// test conversion to and from interleaved back end storage format
TEST(matrix, interleave)
{
    using I = int;
    using T = int;
    using ivec = std::vector<I>;
    using tvec = std::vector<T>;

    // simple case with 4 matrices of length 2
    /*{
        const int padded_size = 2;
        const int num_mtx = 4;
        ivec sizes(num_mtx, padded_size);

        // find the start position of each matrix in the flat storage
        // we are assuming that the matrices are unpermuted
        ivec starts(num_mtx, 0);
        std::partial_sum(begin(sizes), end(sizes)-1, begin(starts)+1);

        tvec values(padded_size*num_mtx);
        std::iota(values.begin(), values.end(), 0);

        EXPECT_TRUE((test_interleave<T, I, 1, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 2, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 3, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 4, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 5, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 6, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 7, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 8, 1>(sizes, starts, values, padded_size)));

        EXPECT_TRUE((test_interleave<T, I, 1, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 2, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 3, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 4, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 5, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 6, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 7, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 8, 2>(sizes, starts, values, padded_size)));

        EXPECT_TRUE((test_interleave<T, I, 1, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 2, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 3, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 4, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 5, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 6, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 7, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 8, 3>(sizes, starts, values, padded_size)));
    }*/

    // another small example with matrices of differing lengths
    {
        const int padded_size = 8;
        const int num_mtx = 8;
        ivec sizes = {6, 5, 4, 4, 3, 2, 2, 1};

        // find the start position of each matrix in the flat storage
        // we are assuming that the matrices are unpermuted
        ivec starts(num_mtx, 0);
        std::partial_sum(begin(sizes), end(sizes)-1, begin(starts)+1);

        tvec values(util::sum(sizes));
        std::iota(values.begin(), values.end(), 0);

        /*EXPECT_TRUE((test_interleave<T, I, 1, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 2, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 3, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 4, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 5, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 6, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 7, 1>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 8, 1>(sizes, starts, values, padded_size)));

        EXPECT_TRUE((test_interleave<T, I, 1, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 2, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 3, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 4, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 5, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 6, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 7, 2>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 8, 2>(sizes, starts, values, padded_size)));

        EXPECT_TRUE((test_interleave<T, I, 1, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 2, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 3, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 4, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 5, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 6, 3>(sizes, starts, values, padded_size)));*/
        EXPECT_TRUE((test_interleave<T, I, 7, 3>(sizes, starts, values, padded_size)));
        EXPECT_TRUE((test_interleave<T, I, 8, 3>(sizes, starts, values, padded_size)));
    }

    // more interesting case...
    /*{
        const int padded_size = 256;
        const int num_mtx = 1000;
        ivec sizes(num_mtx);
        for (auto i: make_span(  0, 100)) sizes[i] = 250;
        for (auto i: make_span(100, 103)) sizes[i] = 213;
        for (auto i: make_span(103, 150)) sizes[i] = 200;
        for (auto i: make_span(150, 500)) sizes[i] = 178;
        for (auto i: make_span(500, 999)) sizes[i] = 6;

        // we are assuming that the matrices are unpermuted
        ivec starts(num_mtx, 0);
        std::partial_sum(begin(sizes), end(sizes)-1, begin(starts)+1);

        tvec values(util::sum(sizes));
        std::iota(values.begin(), values.end(), 0);

        // test in "full" 1024 thread configuration with 32 threads per matrix
        EXPECT_TRUE((test_interleave<T, I, 32, 32>(sizes, starts, values, padded_size)));
    }*/
}

// test that the flat and interleaved storage back ends produce identical results
TEST(matrix, backends)
{
    using T = fvm_value_type;
    using I = fvm_index_type;

    using state_flat = gpu::matrix_state_flat<T, I>;
    using state_intl = gpu::matrix_state_interleaved<T, I>;
    using state_fine = gpu::matrix_state_fine<T, I>;

    using gpu_array  = memory::device_vector<T>;

    // There are two matrix structures:
    //
    // p_1: 3 branches, 6 compartments
    //
    //           3
    //          /.
    // 0 - 1 - 2
    //          \.
    //           4
    //            \.
    //             5
    //
    // p_2: 5 branches, 8 compartments
    //
    //             4
    //            /.
    //           3
    //          / \.
    // 0 - 1 - 2   5
    //          \.
    //           6
    //            \.
    //             7

    // The parent indexes that define the two matrix structures
    std::vector<std::vector<I>>
        p_base = { {0,0,1,2,2,4}, {0,0,1,2,3,3,2,6} };

    // Make a set of matrices based on repeating this pattern.
    // We assign the patterns round-robin, i.e. so that the input
    // matrices will have alternating sizes of 6 and 8, which will
    // test the solver with variable matrix size, and exercise
    // solvers that reorder matrices according to size.
    const int num_mtx = 200;

    std::vector<I> p;
    std::vector<I> cell_cv_divs;
    std::vector<I> cell_to_intdom;
    for (auto m=0; m<num_mtx; ++m) {
        auto &p_ref = p_base[m%2];
        auto first = p.size();
        for (auto i: p_ref) {
            p.push_back(i + first);
        }
        cell_cv_divs.push_back(first);
        cell_to_intdom.push_back(m);
    }
    cell_cv_divs.push_back(p.size());

    auto group_size = cell_cv_divs.back();

    // Build the capacitance, (axial) conductance, voltage, current density,
    // and membrane conductance vectors. Populate them with nonzero random values.
    auto gen  = std::mt19937();
    gen.seed(100);
    auto dist = std::uniform_real_distribution<T>(1, 200);

    std::vector<T> Cm(group_size);
    std::vector<T> g(group_size);
    std::vector<T> v(group_size);
    std::vector<T> i(group_size);
    std::vector<T> mg(group_size);
    std::vector<T> area(group_size, 1e3);

    std::generate(Cm.begin(), Cm.end(), [&](){return dist(gen);});
    std::generate(g.begin(), g.end(), [&](){return dist(gen);});
    std::generate(v.begin(), v.end(), [&](){return dist(gen);});
    std::generate(i.begin(), i.end(), [&](){return dist(gen);});
    std::generate(mg.begin(), mg.end(), [&](){return dist(gen);});

    // Make the reference matrix and the gpu matrix
    auto flat = state_flat(p, cell_cv_divs, Cm, g, area, cell_to_intdom); // flat
    auto intl = state_intl(p, cell_cv_divs, Cm, g, area, cell_to_intdom); // interleaved
    auto fine = state_fine(p, cell_cv_divs, Cm, g, area, cell_to_intdom); // interleaved

    // Set the integration times for the cells to be between 0.01 and 0.02 ms.
    std::vector<T> dt(num_mtx, 0);

    auto dt_dist = std::uniform_real_distribution<T>(0.01, 0.02);
    std::generate(dt.begin(), dt.end(), [&](){return dt_dist(gen);});

    // Voltage, current, and membrane conductance values.
    auto gpu_dt = on_gpu(dt);
    auto gpu_v = on_gpu(v);
    auto gpu_i = on_gpu(i);
    auto gpu_mg = on_gpu(mg);

    flat.assemble(gpu_dt, gpu_v, gpu_i, gpu_mg);
    intl.assemble(gpu_dt, gpu_v, gpu_i, gpu_mg);
    fine.assemble(gpu_dt, gpu_v, gpu_i, gpu_mg);

    flat.solve();
    intl.solve();
    fine.solve();

    // Compare the results.
    // We expect exact equality for the two gpu matrix implementations because both
    // perform the same operations in the same order on the same inputs.
    std::vector<double> x_flat = assign_from(on_host(flat.solution()));
    std::vector<double> x_intl = assign_from(on_host(intl.solution()));
    // as the fine algorithm contains atomics the solution might be slightly
    // different from flat and interleaved
    std::vector<double> x_fine = assign_from(on_host(fine.solution()));

    auto max_diff_fine =
        util::max_value(
            util::transform_view(
                util::count_along(x_flat),
                [&](unsigned i) {return std::abs(x_flat[i] - x_fine[i]);}));

    EXPECT_EQ(x_flat, x_intl);
    EXPECT_LE(max_diff_fine, 1e-12);
}
