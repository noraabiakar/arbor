#pragma once

#include<simd/simd.hpp>

namespace arb {
namespace multicore {
    
namespace S = ::arb::simd;
static constexpr unsigned simd_width_ = S::simd_abi::native_width<fvm_value_type>::value;

   struct constraint_partition {

       using iarray = std::vector<int>;

       iarray serial_part;
       iarray independent_part;
       iarray contiguous_part;
       iarray constant_part;
   };


   template <typename T>
   void gen_constraint(const T& node_index, constraint_partition& partitioned_index) {
       using std::begin;
       using std::end;

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
           switch(con) {
               case index_constraint::none: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       partitioned_index.serial_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::independent: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       partitioned_index.independent_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::constant: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       partitioned_index.constant_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::contiguous: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       partitioned_index.contiguous_part.push_back(node_index[i + j]);
               }
               break;
           }
       }
   }

} // namespace util
} // namespace arb


