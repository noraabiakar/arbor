#include "../gtest.h"

#include <array>
#include <forward_list>
#include <string>
#include <vector>

#include <simd/simd.hpp>
#include <common_types.hpp>
#include <backends/multicore/multicore_common.hpp>
#include <backends/multicore/partition_by_constraint.hpp>

using namespace arb;
using iarray = multicore::iarray;
static constexpr unsigned simd_width_ = arb::simd::simd_abi::native_width<fvm_value_type>::value;

const int input_size_ = 1024;

enum constraint_helpers {
    contiguous= 0,
    independent,
    none,
    constant
};

TEST(partition_by_constraint, partition_contiguous) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i;
    }

    multicore::gen_constraint(input_index, output_constraint);

    EXPECT_EQ(input_size_, output_constraint.compartment_sizes[contiguous]);
    EXPECT_EQ(0, output_constraint.compartment_sizes[constant]);
    EXPECT_EQ(0, output_constraint.compartment_sizes[independent]);
    EXPECT_EQ(0, output_constraint.compartment_sizes[none]);

    for (unsigned i = 0; i < input_size_; i++) {
        EXPECT_EQ(output_constraint.full_index_compartments[i],
                  input_index[i]);
    }
}

TEST(partition_by_constraint, partition_constant) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;

    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = c;
    }

    multicore::gen_constraint(input_index, output_constraint);
    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_, output_constraint.compartment_sizes[constant]);
        EXPECT_EQ(0, output_constraint.compartment_sizes[contiguous]);
    }
    else {
        EXPECT_EQ(0, output_constraint.compartment_sizes[constant]);
        EXPECT_EQ(input_size_, output_constraint.compartment_sizes[contiguous]);
    }
    EXPECT_EQ(0, output_constraint.compartment_sizes[independent]);
    EXPECT_EQ(0, output_constraint.compartment_sizes[none]);

    for (unsigned i = 0; i < input_size_; i++) {
            EXPECT_EQ(output_constraint.full_index_compartments[i], input_index[i]);
    }
}

TEST(partition_by_constraint, partition_independent) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i * 2;
    }

    multicore::gen_constraint(input_index, output_constraint);

    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_, output_constraint.compartment_sizes[independent]);
        EXPECT_EQ(0, output_constraint.compartment_sizes[contiguous]);
    }
    else {
        EXPECT_EQ(0, output_constraint.compartment_sizes[independent]);
        EXPECT_EQ(input_size_, output_constraint.compartment_sizes[contiguous]);
    }
    EXPECT_EQ(0, output_constraint.compartment_sizes[constant]);
    EXPECT_EQ(0, output_constraint.compartment_sizes[none]);


    for (unsigned i = 0; i < input_size_; i++) {
            EXPECT_EQ(output_constraint.full_index_compartments[i], input_index[i]);
    }
}

TEST(partition_by_constraint, partition_serial) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i / ((simd_width_ + 1)/ 2);
    }

    multicore::gen_constraint(input_index, output_constraint);
    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_, output_constraint.compartment_sizes[none]);
        EXPECT_EQ(0, output_constraint.compartment_sizes[contiguous]);
    }
    else {
        EXPECT_EQ(0, output_constraint.compartment_sizes[none]);
        EXPECT_EQ(input_size_, output_constraint.compartment_sizes[contiguous]);
    }

    EXPECT_EQ(0, output_constraint.compartment_sizes[independent]);
    EXPECT_EQ(0, output_constraint.compartment_sizes[constant]);

    for (unsigned i = 0; i < input_size_; i++) {
        EXPECT_EQ(output_constraint.full_index_compartments[i], input_index[i]);
    }
}

TEST(partition_by_constraint, partition_random) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;

    const int c = 5;
//shuffle here
    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = (i < input_size_ / 4 ? i / ((simd_width_ + 1)/ 2) :
                          (i < input_size_ / 2 ? i * 2 :
                           (i < input_size_* 3 / 4 ? c : i)));
    }

    multicore::gen_constraint(input_index, output_constraint);

    if (simd_width_ != 1) {
        EXPECT_EQ(input_size_ / 4, output_constraint.compartment_sizes[contiguous]);
        EXPECT_EQ(input_size_ / 4, output_constraint.compartment_sizes[constant]);
        EXPECT_EQ(input_size_ / 4, output_constraint.compartment_sizes[independent]);
        EXPECT_EQ(input_size_ / 4, output_constraint.compartment_sizes[none]);
    }
    else {
        EXPECT_EQ(input_size_, output_constraint.compartment_sizes[contiguous]);
        EXPECT_EQ(0, output_constraint.compartment_sizes[constant]);
        EXPECT_EQ(0, output_constraint.compartment_sizes[independent]);
        EXPECT_EQ(0, output_constraint.compartment_sizes[none]);
    }

    for (unsigned i = 0; i < input_size_; i++) {
        if (simd_width_ != 1) {
            if (i < input_size_ / 4)
                EXPECT_EQ(output_constraint.full_index_compartments[i + input_size_ / 2],
                          input_index[i]);
            else if (i < input_size_ / 2)
                EXPECT_EQ(output_constraint.full_index_compartments[i],
                          input_index[i]);
            else if (i < input_size_ * 3 / 4)
                EXPECT_EQ(output_constraint.full_index_compartments[i + input_size_ / 4],
                          input_index[i]);
            else
                EXPECT_EQ(output_constraint.full_index_compartments[i - input_size_ * 3 / 4],
                          input_index[i]);
        }
        else {
            EXPECT_EQ(output_constraint.full_index_compartments[i],
                      input_index[i]);
        }
    }
}

TEST(partition_by_constraint, partition_subvector_constraint_categorization) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;

    {
        iarray input_index_constant0;
        for (unsigned i = 0; i < simd_width_; i++) {
            input_index_constant0.push_back(0);
        }
        EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_constant0, 0), index_constraint::constant);
    }

    {
        iarray input_index_contiguous0;
        for (unsigned i = 0; i < simd_width_; i++) {
            input_index_contiguous0.push_back(i);
        }
        EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_contiguous0, 0), index_constraint::contiguous);
    }

    {
        iarray input_index_independent0;
        for (unsigned i = 0; i < simd_width_; i++) {
            input_index_independent0.push_back(i * 2);
        }
        EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_independent0, 0),
                  index_constraint::independent);
    }

    {
        iarray input_index_serial0;
        for (unsigned i = 0; i < simd_width_; i++) {
            input_index_serial0.push_back(i/2);
        }
        EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_serial0, 0), index_constraint::none);
    }

    {
        iarray input_index_serial1 = iarray{1,2,2,4};
        EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_serial1, 0), index_constraint::none);
    }


    {
        iarray input_index_independent1 = iarray{0,2,656,900};
        EXPECT_EQ(multicore::get_subvector_index_constraint(input_index_independent1, 0), index_constraint::independent);
    }
}