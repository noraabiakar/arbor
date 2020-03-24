#pragma once

#include <cstdint>
#include "cuda_atomic.hpp"
#include "cuda_common.hpp"

namespace arb {
namespace gpu {

// Shuffle of double
__device__ __inline__ double shfl(double x, int lane)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"d"(x));

    lo = __shfl(lo,lane,warpSize);
    hi = __shfl(hi,lane,warpSize);

    asm volatile("mov.b64 %0,{%1,%2};":"=d"(x):"r"(lo),"r"(hi));
    return x;
}


// key_set_pos stores information required by a thread to calculate its
// contribution to a reduce by key operation.
//
// In reduce_by_key each thread performs a reduction with threads that have the same
// index in a sorted index array.
// The algorithm for reduce_by_key implemented here requires that each thread
// knows how far it is from the end of the key set, and whether it is the first
// thread in the warp with the key.
//
// As currently implemented, this could be performed inline inside the reduce_by_key
// function, however it is in a separate data type as a first step towards reusing
// the same key set information for multipe reduction kernel calls.
struct key_set_pos {
    unsigned width;         // distance to one past the end of this run
    unsigned lane_id;       // id of this warp lane
    unsigned key_mask;      // warp mask of threads participating in reduction
    unsigned is_root;       // if this lane is the first in the run

    // The constructor takes the index of each thread and a mask of which threads
    // in the warp are participating in the reduction, and the threads work cooperatively
    // using warp intrinsics and bit twiddling, so that each thread will have unique
    // information that describes its position in the key set.
    __device__
    key_set_pos(int idx, unsigned mask) {
        key_mask = mask;
        lane_id = threadIdx.x%impl::threads_per_warp();
        unsigned num_lanes = impl::threads_per_warp()-__clz(key_mask);

        // Determine if this thread is the root (i.e. first thread with this key).
        unsigned src_lane = lane_id - lane_id? 1: 0;
        //int left_idx  = shfl(key_mask, idx, src_lane);
        int left_idx  = shfl(idx, src_lane);

        is_root = lane_id? left_idx!=idx: 1;

        // Determine the range this thread contributes to.
        unsigned roots = __ballot_sync(key_mask, is_root);

        // Find the distance to the lane id one past the end of the run.
        // Take care if this is the last run in the warp.
        width = __ffs(roots>>(lane_id+1));
        if (!width) width = num_lanes-lane_id;
    }
};

template <typename T, typename I>
__device__ __inline__
void reduce_by_key(T contribution, T* target, I i, unsigned mask) {
    key_set_pos run(i, mask);
    unsigned shift = 1;
    const unsigned width = run.width;

    unsigned w = shift<width? shift: 0;

    while (__any_sync(run.key_mask, w)) {
        unsigned src_lane = run.lane_id + w;
        //T source_value = shfl(run.key_mask, contribution, src_lane);
        T source_value = shfl(contribution, src_lane);

        if (w) contribution += source_value;

        shift <<= 1;
        w = shift<width? shift: 0;
    }

    if(run.is_root) {
        // The update must be atomic, because the run may span multiple warps.
        cuda_atomic_add(target+i, contribution);
    }
}

} // namespace gpu
} // namespace arb
