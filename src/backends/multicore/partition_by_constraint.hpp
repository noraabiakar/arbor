#pragma once

#include<vector>
#include<simd/simd.hpp>

namespace arb {
namespace multicore {
    
namespace S = ::arb::simd;
static constexpr unsigned simd_width_ = S::simd_abi::native_width<fvm_value_type>::value;

   struct constraint_partition {

       using iarray = arb::multicore::iarray;

       static constexpr int num_compartments = 4;
       iarray full_index_compartments;
       std::vector<int> compartment_sizes;
       std::vector<int> compartment_starts_and_ends;
   };


   template <typename T>
   index_constraint get_subvector_index_constraint(const T& node_index, unsigned i) {
       index_constraint con = index_constraint::contiguous;
       if(simd_width_ != 1) {
           if(node_index[i] == node_index[i + 1])
               con = index_constraint::constant;
           for (unsigned j = i + 1; j < i + simd_width_; j++) {
               switch (con) {
                   case index_constraint::independent: {
                       if (node_index[j] == node_index[j - 1])
                           con = index_constraint::none;
                   }
                       break;
                   case index_constraint::constant: {
                       if (node_index[j] != node_index[j - 1])
                           con = index_constraint::none;
                   }
                       break;
                   case index_constraint::contiguous: {
                       if (node_index[j] != node_index[j - 1] + 1) {
                           con = index_constraint::independent;
                           if(node_index[j] == node_index[j - 1])
                               con = index_constraint::none;
                       }
                   }
                       break;
                   default: {
                   }
               }
           }
       }
       return con;
   }

   template <typename T>
   void gen_constraint(const T& node_index, constraint_partition& partitioned_index) {
       using iarray = arb::multicore::iarray;

       iarray serial_part;
       iarray independent_part;
       iarray contiguous_part;
       iarray constant_part;

       std::cout<<"size is: "<<node_index.size()<<std::endl;
       for (unsigned i = 0; i < node_index.size(); i++) {
           std::cout << node_index[i] << " ";
           if(i %4 == 3)
               std::cout<<std::endl;
       }
       std::cout<<std::endl<<"_________________________"<<std::endl;

       for (unsigned i = 0; i < node_index.size(); i+= simd_width_) {
           index_constraint con = get_subvector_index_constraint(node_index, i);

           switch(con) {
               case index_constraint::none: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       serial_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::independent: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       independent_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::constant: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       constant_part.push_back(node_index[i + j]);
               }
               break;
               case index_constraint::contiguous: {
                   for (unsigned j = 0; j < simd_width_; j++)
                       contiguous_part.push_back(node_index[i + j]);
               }
               break;
           }
       }

       partitioned_index.full_index_compartments.reserve(
               serial_part.size() + independent_part.size() +
               contiguous_part.size() + constant_part.size() );// preallocate memory

       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               contiguous_part.begin(), contiguous_part.end() );

       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               independent_part.begin(), independent_part.end() );

       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               serial_part.begin(), serial_part.end() );

       partitioned_index.full_index_compartments.insert(
               partitioned_index.full_index_compartments.end(),
               constant_part.begin(), constant_part.end() );

       partitioned_index.compartment_sizes.push_back(contiguous_part.size());
       partitioned_index.compartment_sizes.push_back(independent_part.size());
       partitioned_index.compartment_sizes.push_back(serial_part.size());
       partitioned_index.compartment_sizes.push_back(constant_part.size());

       partitioned_index.compartment_starts_and_ends.push_back(0); // first partition always starts at 0
       for (int c = 1; c <= constraint_partition::num_compartments; c++) {
           int previous_partition_end = partitioned_index.compartment_starts_and_ends[c - 1];
           int size_of_partition = partitioned_index.compartment_sizes[c-1];
           partitioned_index.compartment_starts_and_ends.push_back(previous_partition_end + size_of_partition);
       }

       std::cout<<"Size of original index array: "<<node_index.size() <<" full partitions : "<<partitioned_index.full_index_compartments.size()<<std::endl;
       std::cout<<"sizes are: "<<partitioned_index.compartment_sizes[0] << " " << partitioned_index.compartment_sizes[1] << " " << partitioned_index.compartment_sizes[2] << " " << partitioned_index.compartment_sizes[3] << std::endl;
       for (unsigned i = 0; i < partitioned_index.full_index_compartments.size(); i++) {
           std::cout << partitioned_index.full_index_compartments[i] << " ";
           if(i%4 == 3)
               std::cout<<std::endl;
       }
       std::cout<<std::endl<<"_________________________"<<std::endl;
   }

} // namespace util
} // namespace arb


