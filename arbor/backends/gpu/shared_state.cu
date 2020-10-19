// GPU kernels and wrappers for shared state methods.

#include <cstdint>

#include <backends/event.hpp>
#include <backends/multi_event_stream_state.hpp>

#include "gpu_api.hpp"
#include "gpu_common.hpp"

namespace arb {
namespace gpu {

namespace kernel {

template <typename T>
__global__ void update_time_to_impl(unsigned n,
                                    T* __restrict__ const time_to,
                                    const T* __restrict__ const time,
                                    T dt,
                                    T tmax) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        auto t = time[i]+dt;
        time_to[i] = t<tmax? t: tmax;
    }
}

template <typename T, typename I>
__global__ void add_gj_current_impl(unsigned n,
                                    const T* __restrict__ const gj_info,
                                    const I* __restrict__ const voltage,
                                    I* __restrict__ const current_density) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        auto gj = gj_info[i];
        auto curr = gj.weight * (voltage[gj.loc.second] - voltage[gj.loc.first]); // nA

        gpu_atomic_sub(current_density + gj.loc.first, curr);
    }
}

// Vector/scalar addition: x[i] += v ∀i
template <typename T>
__global__ void add_scalar(unsigned n,
                           T* __restrict__ const x,
                           fvm_value_type v) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        x[i] += v;
    }
}

template <typename T, typename I>
__global__ void set_dt_impl(      T* __restrict__ dt_intdom,
                            const T* __restrict__ time_to,
                            const T* __restrict__ time,
                            const unsigned ncomp,
                                  T* __restrict__ dt_comp,
                            const I* __restrict__ cv_to_intdom) {
    auto idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < ncomp) {
        const auto ind = cv_to_intdom[idx];
        const auto dt = time_to[ind] - time[ind];
        dt_intdom[ind] = dt;
        dt_comp[idx] = dt;
    }
}

__global__ void take_samples_impl(
    multi_event_stream_state<raw_probe_info> s,
    const fvm_value_type* __restrict__ const time,
    fvm_value_type* __restrict__ const sample_time,
    fvm_value_type* __restrict__ const sample_value)
{
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<s.n) {
        auto begin = s.ev_data+s.begin_offset[i];
        auto end = s.ev_data+s.end_offset[i];
        for (auto p = begin; p!=end; ++p) {
            sample_time[p->offset] = time[i];
            sample_value[p->offset] = *p->handle;
        }
    }
}

__global__ void reduce_impl(
        const fvm_value_type* local_i,
        const fvm_value_type* local_g,
        fvm_value_type* global_i,
        fvm_value_type* global_g,
        const fvm_index_type* reduction_partition,
        fvm_size_type ncv,
        fvm_size_type warp_size)
{
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<ncv) {
        fvm_value_type sum_i, sum_g;
        auto count = reduction_partition[i];
        for (unsigned offset = 0; offset < count * warp_size; offset += warp_size) {
            sum_i += local_i[i + offset];
            sum_g += local_g[i + offset];
        }
        global_i[i] += sum_i;
        global_g[i] += sum_g;
    }
}

} // namespace kernel

using impl::block_count;

void add_scalar(std::size_t n, fvm_value_type* data, fvm_value_type v) {
    if (!n) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(n, block_dim);
    kernel::add_scalar<<<nblock, block_dim>>>(n, data, v);
}

void update_time_to_impl(
    std::size_t n, fvm_value_type* time_to, const fvm_value_type* time,
    fvm_value_type dt, fvm_value_type tmax)
{
    if (!n) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(n, block_dim);
    kernel::update_time_to_impl<<<nblock, block_dim>>>(n, time_to, time, dt, tmax);
}

void set_dt_impl(
    fvm_size_type nintdom, fvm_size_type ncomp, fvm_value_type* dt_intdom, fvm_value_type* dt_comp,
    const fvm_value_type* time_to, const fvm_value_type* time, const fvm_index_type* cv_to_intdom)
{
    if (!nintdom || !ncomp) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(ncomp, block_dim);
    kernel::set_dt_impl<<<nblock, block_dim>>>(dt_intdom, time_to, time, ncomp, dt_comp, cv_to_intdom);
}

void add_gj_current_impl(
    fvm_size_type n_gj, const fvm_gap_junction* gj_info, const fvm_value_type* voltage, fvm_value_type* current_density)
{
    if (!n_gj) return;

    constexpr int block_dim = 128;
    int nblock = block_count(n_gj, block_dim);
    kernel::add_gj_current_impl<<<nblock, block_dim>>>(n_gj, gj_info, voltage, current_density);
}

void take_samples_impl(
    const multi_event_stream_state<raw_probe_info>& s,
    const fvm_value_type* time, fvm_value_type* sample_time, fvm_value_type* sample_value)
{
    if (!s.n_streams()) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(s.n_streams(), block_dim);
    kernel::take_samples_impl<<<nblock, block_dim>>>(s, time, sample_time, sample_value);
}


void reduce_impl(
    const fvm_value_type* local_i,
    const fvm_value_type* local_g,
    fvm_value_type* global_i,
    fvm_value_type* global_g,
    const fvm_index_type* reduction_part,
    fvm_size_type ncv)
{
    if (!ncv) return;

    constexpr int block_dim = 128;
    constexpr int warp_size = impl::threads_per_warp();
    const int nblock = block_count(ncv, block_dim);
    kernel::reduce_impl<<<nblock, block_dim>>>(local_i, local_g, global_i, global_g, reduction_part, ncv, warp_size);
}

} // namespace gpu
} // namespace arb
