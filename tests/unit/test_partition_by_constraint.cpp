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

TEST(partition_by_constraint, partition_contiguous) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;
    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i;
    }

    multicore::gen_constraint(input_index, output_constraint);

    EXPECT_EQ(input_size_, output_constraint.contiguous_part.size());
    EXPECT_EQ(0, output_constraint.constant_part.size());
    EXPECT_EQ(0, output_constraint.independent_part.size());
    EXPECT_EQ(0, output_constraint.serial_part.size());

    multicore::constraint_partition output_constraint_expected;
    output_constraint_expected.contiguous_part.resize(input_size_);

    for (unsigned i = 0; i < input_size_; i++) {
        output_constraint_expected.contiguous_part[i] = i;
    }

    for (unsigned i = 0; i < input_size_; i++) {
        //EXPECT_EQ(output_constraint.constant_part[i], output_constraint_expected.contiguous_part[i]);
    }
}

TEST(partition_by_constraint, partition_constant) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;
    multicore::constraint_partition output_constraint_expected;
    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = c;
    }

    multicore::gen_constraint(input_index, output_constraint);
    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_, output_constraint.constant_part.size());
        EXPECT_EQ(0, output_constraint.contiguous_part.size());

        output_constraint_expected.constant_part.resize(input_size_);
    }
    else {
        EXPECT_EQ(0, output_constraint.constant_part.size());
        EXPECT_EQ(input_size_, output_constraint.contiguous_part.size());

        output_constraint_expected.contiguous_part.resize(input_size_);
    }
    EXPECT_EQ(0, output_constraint.independent_part.size());
    EXPECT_EQ(0, output_constraint.serial_part.size());

    for (unsigned i = 0; i < input_size_; i++) {
        if (simd_width_ != 1)
            output_constraint_expected.constant_part[i] = c;
        else
            output_constraint_expected.contiguous_part[i] = c;
    }

    for (unsigned i = 0; i < input_size_; i++) {
        if (simd_width_ != 1)
            EXPECT_EQ(output_constraint.constant_part[i], output_constraint_expected.constant_part[i]);
        else
            EXPECT_EQ(output_constraint.contiguous_part[i], output_constraint_expected.contiguous_part[i]);
    }
}

TEST(partition_by_constraint, partition_independent) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;
    multicore::constraint_partition output_constraint_expected;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i * 2;
    }

    multicore::gen_constraint(input_index, output_constraint);

    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_, output_constraint.independent_part.size());
        EXPECT_EQ(0, output_constraint.contiguous_part.size());

        output_constraint_expected.independent_part.resize(input_size_);
    }
    else {
        EXPECT_EQ(0, output_constraint.independent_part.size());
        EXPECT_EQ(input_size_, output_constraint.contiguous_part.size());

        output_constraint_expected.contiguous_part.resize(input_size_);
    }
    EXPECT_EQ(0, output_constraint.constant_part.size());
    EXPECT_EQ(0, output_constraint.serial_part.size());

    for (unsigned i = 0; i < input_size_; i++) {
        if (simd_width_ != 1)
            output_constraint_expected.independent_part[i] = i * 2;
        else
            output_constraint_expected.contiguous_part[i] = i * 2;
    }

    for (unsigned i = 0; i < input_size_; i++) {
        if(simd_width_ != 1)
            EXPECT_EQ(output_constraint.independent_part[i], output_constraint_expected.independent_part[i]);
        else
            EXPECT_EQ(output_constraint.contiguous_part[i], output_constraint_expected.contiguous_part[i]);
    }
}

TEST(partition_by_constraint, partition_serial) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;
    multicore::constraint_partition output_constraint_expected;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i / ((simd_width_ + 1)/ 2);
    }

    multicore::gen_constraint(input_index, output_constraint);
    if(simd_width_ != 1) {
        EXPECT_EQ(input_size_, output_constraint.serial_part.size());
        EXPECT_EQ(0, output_constraint.contiguous_part.size());

        output_constraint_expected.serial_part.resize(input_size_);
    }
    else {
        EXPECT_EQ(0, output_constraint.serial_part.size());
        EXPECT_EQ(input_size_, output_constraint.contiguous_part.size());

        output_constraint_expected.contiguous_part.resize(input_size_);
    }
    EXPECT_EQ(0, output_constraint.independent_part.size());
    EXPECT_EQ(0, output_constraint.constant_part.size());

    for (unsigned i = 0; i < input_size_; i++) {
        if (simd_width_ != 1)
            output_constraint_expected.serial_part[i] = i / ((simd_width_ + 1)/ 2);
        else
            output_constraint_expected.contiguous_part[i] = i / ((simd_width_ + 1)/ 2);
    }

    for (unsigned i = 0; i < input_size_; i++) {
        if (simd_width_ != 1)
            EXPECT_EQ(output_constraint.serial_part[i], output_constraint_expected.serial_part[i]);
        else
            EXPECT_EQ(output_constraint.contiguous_part[i], output_constraint_expected.contiguous_part[i]);
    }
}

TEST(partition_by_constraint, partition_random) {
    iarray input_index(input_size_);
    multicore::constraint_partition output_constraint;
    multicore::constraint_partition output_constraint_expected;
    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = (i < input_size_ / 4 ? i :
                          (i < input_size_ / 2 ? c :
                           (i < input_size_* 3 / 4 ? i * 2 :
                            i / ((simd_width_ + 1)/ 2))));
    }

    multicore::gen_constraint(input_index, output_constraint);

    if (simd_width_ != 1) {
        EXPECT_EQ(input_size_ / 4, output_constraint.contiguous_part.size());
        EXPECT_EQ(input_size_ / 4, output_constraint.constant_part.size());
        EXPECT_EQ(input_size_ / 4, output_constraint.independent_part.size());
        EXPECT_EQ(input_size_ / 4, output_constraint.serial_part.size());

        output_constraint_expected.contiguous_part.resize(input_size_ / 4);
        output_constraint_expected.constant_part.resize(input_size_ / 4);
        output_constraint_expected.independent_part.resize(input_size_ / 4);
        output_constraint_expected.serial_part.resize(input_size_ / 4);
    }
    else {
        EXPECT_EQ(input_size_, output_constraint.contiguous_part.size());
        EXPECT_EQ(0, output_constraint.constant_part.size());
        EXPECT_EQ(0, output_constraint.independent_part.size());
        EXPECT_EQ(0, output_constraint.serial_part.size());

        output_constraint_expected.contiguous_part.resize(input_size_);
    }

    for (unsigned i = 0; i < input_size_ / 4; i++) {
        if(simd_width_ != 1) {
            output_constraint_expected.contiguous_part[i] = i;
            output_constraint_expected.constant_part[i] = c;
            output_constraint_expected.independent_part[i] = (i + input_size_ / 2) * 2;
            output_constraint_expected.serial_part[i] = (i + input_size_ * 3 / 4)
                    / ((simd_width_ + 1) / 2);
        }
        else {
            output_constraint_expected.contiguous_part[i] = i;
            output_constraint_expected.contiguous_part[i + input_size_ / 4] = c;
            output_constraint_expected.contiguous_part[i + input_size_ / 2] =
                    (i + input_size_ / 2) * 2;
            output_constraint_expected.contiguous_part[i + input_size_ * 3 / 4] =
                    (i + input_size_ * 3 / 4) / ((simd_width_ + 1) / 2);
        }
    }

    for (unsigned i = 0; i < input_size_ / 4; i++) {
        if (simd_width_ != 1) {
            EXPECT_EQ(output_constraint.contiguous_part[i],
                  output_constraint_expected.contiguous_part[i]);
            EXPECT_EQ(output_constraint.constant_part[i],
                  output_constraint_expected.constant_part[i]);
            EXPECT_EQ(output_constraint.independent_part[i],
                  output_constraint_expected.independent_part[i]);
            EXPECT_EQ(output_constraint.serial_part[i],
                  output_constraint_expected.serial_part[i]);
        }
        else {
            EXPECT_EQ(output_constraint.contiguous_part[i],
                      output_constraint_expected.contiguous_part[i]);
            EXPECT_EQ(output_constraint.contiguous_part[i + input_size_ / 4],
                      output_constraint_expected.contiguous_part[i + input_size_ / 4]);
            EXPECT_EQ(output_constraint.contiguous_part[i + input_size_ / 2],
                      output_constraint_expected.contiguous_part[i  + input_size_ / 2]);
            EXPECT_EQ(output_constraint.contiguous_part[i + input_size_ * 3 / 4],
                      output_constraint_expected.contiguous_part[i + input_size_ * 3 / 4]);
        }

    }
}