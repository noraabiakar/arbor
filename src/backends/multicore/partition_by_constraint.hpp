#pragma once

#include<simd/simd.hpp>
#include<backends/multicore/multicore_common.hpp>

namespace arb {
namespace multicore {
    
namespace S = ::arb::simd;
static constexpr unsigned simd_width_ = S::simd_abi::native_width<fvm_value_type>::value;

   struct constraint_partition {

       using iarray = arb::multicore::iarray;

       iarray serial_part;
       iarray independent_part;
       iarray contiguous_part;
       iarray constant_part;
   };


   template <typename T, typename C>
   void gen_constraint(const T& node_index, C& partitioned_index) {

       for (unsigned i = 0; i < node_index.size(); i+= simd_width_) {
           index_constraint con = node_index[i] == node_index[i + 1] ?
                                  index_constraint::constant : index_constraint::contiguous;
           for (unsigned j = i + 1; j < i + simd_width_; j++) {
               switch(con) {
                   case index_constraint::independent: {
                       if(node_index[j] == node_index[j - 1])
                           con = index_constraint::none;
                   }
                   break;
                   case index_constraint::constant: {
                       if(node_index[j] != node_index[j - 1])
                           con = index_constraint::none;
                   }
                   break;
                   case index_constraint::contiguous: {
                       if (node_index[j] != node_index[j - 1] + 1)
                           con = index_constraint::independent;
                   }
                   break;
                   default: {}
                }
           }
           partitioned_index.push_back(con);
       }
   }

} // namespace util
} // namespace arb


