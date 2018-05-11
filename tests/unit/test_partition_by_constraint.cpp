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

const int input_size_ = 1048576;
const int num_vectors_ = input_size_/simd_width_;

TEST(partition_by_constraint, partition_contiguous) {
    iarray input_index(input_size_);
    std::vector<index_constraint> output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i;
    }

    multicore::gen_constraint(input_index, output_constraint);

    EXPECT_EQ(num_vectors_, output_constraint.size());

    std::vector<index_constraint> output_constraint_expected(num_vectors_);
    for (unsigned i = 0; i < num_vectors_; i++) {
        output_constraint_expected[i] = index_constraint::contiguous;
    }

    for (unsigned i = 0; i < num_vectors_; i++) {
        EXPECT_EQ(output_constraint[i], output_constraint_expected[i]);
    }
}

TEST(partition_by_constraint, partition_constant) {
    iarray input_index(input_size_);
    std::vector<index_constraint> output_constraint;

    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = c;
    }

    multicore::gen_constraint(input_index, output_constraint);

    EXPECT_EQ(num_vectors_, output_constraint.size());

    std::vector<index_constraint> output_constraint_expected(num_vectors_);
    for (unsigned i = 0; i < num_vectors_; i++) {
        output_constraint_expected[i] = index_constraint::constant;
    }

    for (unsigned i = 0; i < num_vectors_; i++) {
        EXPECT_EQ(output_constraint[i], output_constraint_expected[i]);
    }
}

TEST(partition_by_constraint, partition_independent) {
    iarray input_index(input_size_);
    std::vector<index_constraint> output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i * 2;
    }

    multicore::gen_constraint(input_index, output_constraint);

    EXPECT_EQ(num_vectors_, output_constraint.size());

    std::vector<index_constraint> output_constraint_expected(num_vectors_);
    for (unsigned i = 0; i < num_vectors_; i++) {
        output_constraint_expected[i] = index_constraint::independent;
    }

    for (unsigned i = 0; i < num_vectors_; i++) {
        EXPECT_EQ(output_constraint[i], output_constraint_expected[i]);
    }
}

TEST(partition_by_constraint, partition_serial) {
    iarray input_index(input_size_);
    std::vector<index_constraint> output_constraint;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = i / 2;
    }

    multicore::gen_constraint(input_index, output_constraint);

    EXPECT_EQ(num_vectors_, output_constraint.size());

    std::vector<index_constraint> output_constraint_expected(num_vectors_);
    for (unsigned i = 0; i < num_vectors_; i++) {
        output_constraint_expected[i] = index_constraint::none;
    }

    for (unsigned i = 0; i < num_vectors_; i++) {
        EXPECT_EQ(output_constraint[i], output_constraint_expected[i]);
    }
}

TEST(partition_by_constraint, partition_random) {
    iarray input_index(input_size_);
    std::vector<index_constraint> output_constraint;

    const int c = 5;

    for (unsigned i = 0; i < input_size_; i++) {
        input_index[i] = (i < input_size_/4 ? i :
                (i < input_size_/2 ? c :
                 (i < input_size_*3/4 ? i * 2 :
                  i / 2)));
    }

    multicore::gen_constraint(input_index, output_constraint);

    EXPECT_EQ(num_vectors_, output_constraint.size());

    std::vector<index_constraint> output_constraint_expected(num_vectors_);
    for (unsigned i = 0; i < num_vectors_; i++) {
        output_constraint_expected[i] = (i < num_vectors_/4 ? index_constraint::contiguous :
                (i < num_vectors_/2 ? index_constraint::constant :
                        (i < num_vectors_*3/4 ? index_constraint::independent :
                                   index_constraint::none)));
    }

    for (unsigned i = 0; i < num_vectors_; i++) {
        EXPECT_EQ(output_constraint[i], output_constraint_expected[i]);
    }
}