#include <cstddef>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/fvm_types.hpp>

#include "backends/event.hpp"
#include "backends/gpu/gpu_common.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/shared_state.hpp"
#include "backends/multi_event_stream_state.hpp"
#include "memory/copy.hpp"
#include "memory/wrappers.hpp"
#include "util/rangeutil.hpp"

using arb::memory::make_const_view;

namespace arb {
namespace gpu {

// CUDA implementation entry points:

void update_time_to_impl(
    std::size_t n, fvm_value_type* time_to, const fvm_value_type* time,
    fvm_value_type dt, fvm_value_type tmax);

void update_time_to_impl(
    std::size_t n, fvm_value_type* time_to, const fvm_value_type* time,
    fvm_value_type dt, fvm_value_type tmax);

void set_dt_impl(
    fvm_size_type nintdom, fvm_size_type ncomp, fvm_value_type* dt_intdom, fvm_value_type* dt_comp,
    const fvm_value_type* time_to, const fvm_value_type* time, const fvm_index_type* cv_to_intdom);

void add_gj_current_impl(
    fvm_size_type n_gj, const fvm_gap_junction* gj, const fvm_value_type* v, fvm_value_type* i);

void take_samples_impl(
    const multi_event_stream_state<raw_probe_info>& s,
    const fvm_value_type* time, fvm_value_type* sample_time, fvm_value_type* sample_value);

void reduce_impl(
    const fvm_value_type* local_i,
    const fvm_value_type* local_g,
    fvm_value_type* global_i,
    fvm_value_type* global_g,
    const fvm_index_type* reduction_part,
    fvm_size_type ncv);

void add_scalar(std::size_t n, fvm_value_type* data, fvm_value_type v);

// GPU-side minmax: consider CUDA kernel replacement.
std::pair<fvm_value_type, fvm_value_type> minmax_value_impl(fvm_size_type n, const fvm_value_type* v) {
    auto v_copy = memory::on_host(memory::const_device_view<fvm_value_type>(v, n));
    return util::minmax_value(v_copy);
}

// Ion state methods:

ion_state::ion_state(
    int charge,
    const fvm_ion_config& ion_data,
    unsigned // alignment/padding ignored.
):
    node_index_(make_const_view(ion_data.cv)),
    iX_(ion_data.cv.size(), NAN),
    eX_(ion_data.cv.size(), NAN),
    Xi_(ion_data.cv.size(), NAN),
    Xo_(ion_data.cv.size(), NAN),
    init_Xi_(make_const_view(ion_data.init_iconc)),
    init_Xo_(make_const_view(ion_data.init_econc)),
    reset_Xi_(make_const_view(ion_data.reset_iconc)),
    reset_Xo_(make_const_view(ion_data.reset_econc)),
    init_eX_(make_const_view(ion_data.init_revpot)),
    charge(1u, charge)
{
    arb_assert(node_index_.size()==init_Xi_.size());
    arb_assert(node_index_.size()==init_Xo_.size());
    arb_assert(node_index_.size()==init_eX_.size());
}

void ion_state::init_concentration() {
    memory::copy(init_Xi_, Xi_);
    memory::copy(init_Xo_, Xo_);
}

void ion_state::zero_current() {
    memory::fill(iX_, 0);
}

void ion_state::reset() {
    zero_current();
    memory::copy(reset_Xi_, Xi_);
    memory::copy(reset_Xo_, Xo_);
    memory::copy(init_eX_, eX_);
}

// Shared state methods:

shared_state::shared_state(
    fvm_size_type n_intdom,
    const std::vector<fvm_index_type>& cv_to_intdom_vec,
    const std::vector<fvm_gap_junction>& gj_vec,
    const std::vector<fvm_value_type>& init_membrane_potential,
    const std::vector<fvm_value_type>& temperature_K,
    const std::vector<fvm_value_type>& diam,
    unsigned // alignment parameter ignored.
):
    n_intdom(n_intdom),
    n_cv(cv_to_intdom_vec.size()),
    n_gj(gj_vec.size()),
    cv_to_intdom(make_const_view(cv_to_intdom_vec)),
    gap_junctions(make_const_view(gj_vec)),
    time(n_intdom),
    time_to(n_intdom),
    dt_intdom(n_intdom),
    dt_cv(n_cv),
    voltage(n_cv),
    current_density(n_cv),
    conductivity(n_cv),
    init_voltage(make_const_view(init_membrane_potential)),
    temperature_degC(make_const_view(temperature_K)),
    diam_um(make_const_view(diam)),
    deliverable_events(n_intdom)
{
    add_scalar(temperature_degC.size(), temperature_degC.data(), -273.15);
}

void shared_state::add_ion(
    const std::string& ion_name,
    int charge,
    const fvm_ion_config& ion_info)
{
    ion_data.emplace(std::piecewise_construct,
        std::forward_as_tuple(ion_name),
        std::forward_as_tuple(charge, ion_info, 1u));
}

void shared_state::reset() {
    memory::copy(init_voltage, voltage);
    memory::fill(current_density, 0);
    memory::fill(conductivity, 0);
    memory::fill(time, 0);
    memory::fill(time_to, 0);

    for (auto& i: ion_data) {
        i.second.reset();
    }
}

void shared_state::zero_locals() {
    memory::fill(local_i, 0);
    memory::fill(local_g, 0);
}

void shared_state::zero_currents() {
    zero_locals();
    memory::fill(current_density, 0);
    memory::fill(conductivity, 0);
    for (auto& i: ion_data) {
        i.second.zero_current();
    }
}

void shared_state::ions_init_concentration() {
    for (auto& i: ion_data) {
        i.second.init_concentration();
    }
}

void shared_state::update_time_to(fvm_value_type dt_step, fvm_value_type tmax) {
    update_time_to_impl(n_intdom, time_to.data(), time.data(), dt_step, tmax);
}

void shared_state::set_dt() {
    set_dt_impl(n_intdom, n_cv, dt_intdom.data(), dt_cv.data(), time_to.data(), time.data(), cv_to_intdom.data());
}

void shared_state::add_gj_current() {
    add_gj_current_impl(n_gj, gap_junctions.data(), voltage.data(), current_density.data());
}

std::pair<fvm_value_type, fvm_value_type> shared_state::time_bounds() const {
    return minmax_value_impl(n_intdom, time.data());
}

std::pair<fvm_value_type, fvm_value_type> shared_state::voltage_bounds() const {
    return minmax_value_impl(n_cv, voltage.data());
}

void shared_state::take_samples(const sample_event_stream::state& s, array& sample_time, array& sample_value) {
    take_samples_impl(s, time.data(), sample_time.data(), sample_value.data());
}

void shared_state::reduce() {
    reduce_impl(local_i.data(), local_g.data(), current_density.data(), conductivity.data(), reduction_partition.data(), n_cv);
}

void shared_state::build_cv_index(std::vector<std::pair<unsigned, std::vector<fvm_index_type>>> mech_cv) {
    std::cout << "n_cv = " << n_cv << std::endl;

    std::cout << "------INPUT-----------------------------\n";

    for (auto c: mech_cv) {
        std::cout << c.first << ": ";
        for (auto e: c.second) {
            std::cout << "\t" << e << std::endl;
        }
    }

    const int vsize = impl::threads_per_warp();
    const int gapid = -1;
    if (mech_cv.empty()) return;
    struct cv_prop {
        int cv_idx;
        int mech_id;
        int id;
    };

    std::vector<fvm_index_type> reduction_part, shuffle_idx;
    mech_partition.push_back(0);
    for (auto v: mech_cv) {
        mech_partition.push_back(mech_partition.back()+v.second.size());
    }

    std::vector<cv_prop> mech_cv_props;
    mech_cv_props.reserve(mech_partition.back());

    for(auto v:mech_cv) {
        auto id  = v.first;
        auto cvs = v.second;
        mech_cv_props.reserve(cvs.size());

        for (auto cv: cvs) {
            mech_cv_props.push_back({cv, (int)id, -1});
        }
    }

    auto comp     = [](auto& lhs, auto& rhs) {return std::tie(lhs.cv_idx, lhs.mech_id, lhs.id) <
                                                     std::tie(rhs.cv_idx, rhs.mech_id, rhs.id);};
    auto comp_rev = [](auto& lhs, auto& rhs) {return std::tie(lhs.mech_id, lhs.cv_idx, lhs.id) <
                                                     std::tie(rhs.mech_id, rhs.cv_idx, rhs.id);};

    if (mech_partition.size() > 2) {
        for (unsigned i = 2; i < mech_partition.size(); ++i) {
            auto begin = mech_cv_props.begin();
            auto middle = begin + mech_partition[i-1];
            auto last   = begin + mech_partition[i];

            std::inplace_merge(begin, middle, last, comp);
        }
    };

    std::cout << "------MECH_CV_PROPS------------------------\n";

    for (auto c: mech_cv_props) {
        std::cout << c.cv_idx << " " << c.mech_id << " " << c.id;
        std::cout << std::endl;
    }

    // Get the mech_cv_prop partition of each of CV, partitioned in vectors of size `vsize`
    std::vector<std::vector<int>> idx_part_vectors;

    for (int cv_start = 0; cv_start < (int)n_cv; cv_start += vsize) {
        std::vector<int> idx_vec;
        int cv_end = cv_start + vsize;

        for (int cv = cv_start; (cv < cv_end && cv < (int)n_cv); ++cv) {
            auto first = std::lower_bound(mech_cv_props.begin(), mech_cv_props.end(), cv, [](auto& lhs, auto& rhs) { return lhs.cv_idx < rhs; });
            idx_vec.push_back((int)(first - mech_cv_props.begin()));
        }

        auto last = std::upper_bound(mech_cv_props.begin(), mech_cv_props.end(), cv_end-1, [](auto& lhs, auto& rhs) { return lhs < rhs.cv_idx; });
        while (idx_vec.size() < vsize+1) idx_vec.push_back((int)(last - mech_cv_props.begin()));

        idx_part_vectors.push_back(idx_vec);
    }
    
    std::cout << "------IDX_PARTITION_VECTOR-----------------\n";

    for (auto c: idx_part_vectors) {
        for (auto cv: c) {
            std::cout << cv << " ";
        }
        std::cout << std::endl;
    }

    // Now we want to sort the elements from mech_cv_prop such that mech updates to the same cv
    // are separated by a stride of vsize, while being optimally packed in the final update vector.
    // e.g cvs 0,1 have pas, hh mechanisms; cv 2 has only nax; cv 3 has only hh
    // We want the following occupancy in the update vectors (let pas_x be the instance of pas on CV x)
    // pas_0 pas_1 nax_2 hh_3 | hh_0 hh_1 - - |

    std::vector<fvm_index_type> reduction_count(idx_part_vectors.size());

    std::vector<cv_prop> strided_cv_prop;
    strided_cv_prop.reserve(mech_cv_props.size());

    auto idx_ptr_vectors = idx_part_vectors;
    for (unsigned i = 0; i < idx_ptr_vectors.size(); ++i) {
        auto& idx_ptr       = idx_ptr_vectors[i];
        const auto& idx_cst = idx_part_vectors[i];

        std::set<unsigned> hit_idx;
        std::vector<cv_prop> mini_shuffle(vsize);

        int count = 0;
        while (true) {
            for (unsigned j = 0; j < vsize; ++j) {
                auto& start_idx = idx_ptr[j];
                auto end_idx = idx_cst[j + 1];

                if (start_idx == end_idx) {
                    hit_idx.insert(j);
                    // insert a gap, indicated by mech_id/cv_idx = -1
                    mini_shuffle[j] = {gapid, gapid, -1};
                } else {
                    mini_shuffle[j] = mech_cv_props.at(start_idx++);
                }
            }
            if (hit_idx.size() == vsize) break;
            std::move(mini_shuffle.begin(), mini_shuffle.end(), std::back_inserter(strided_cv_prop));
            count++;
        }
        reduction_count[i] = count;
    }

    // Create reduction_part
    reduction_part.reserve(n_cv);
    for (auto c: reduction_count) {
        for (unsigned i = 0; i < vsize; ++i) {
            reduction_part.push_back(c);
        }
    }

    std::cout << "------REDUCTION_PART----------------------\n";

    for (auto c: reduction_part) {
        std::cout << c << std::endl;
    }

    std::cout << "------SHUFFLED_STRUCT_VECTOR---------------\n";

    for (auto c: strided_cv_prop) {
        std::cout << c.cv_idx << "\t" << c.mech_id << "\t" << c.id << std::endl;
    }

    // Now that we have the correct occupancy of the update vectors we can populate their index
    int id = 0;
    for (auto& s: strided_cv_prop) {
        s.id = id++;
    }

    // Save the total size of the local i and g vectors
    auto local_size = strided_cv_prop.size();

    // Sort according to the mechanism index
    std::sort(strided_cv_prop.begin(), strided_cv_prop.end(), comp_rev);

    // Erase all the first chunk of the vector with mech_idx = -1, which refer to the gaps inserted above
    auto start = std::upper_bound(strided_cv_prop.begin(), strided_cv_prop.end(), gapid, [](auto& lhs, auto& rhs) { return lhs < rhs.mech_id; });
    strided_cv_prop.erase(strided_cv_prop.begin(), start);

    std::cout << "-----SORTED_SHUFFLED_STRUCT_VECTOR---------\n";

    for (auto i: strided_cv_prop) {
        std::cout << i.id << "\t" << i.mech_id << "\t" << i.cv_idx << std::endl;
    }

    shuffle_idx.resize(strided_cv_prop.size());
    for (unsigned i = 0; i < strided_cv_prop.size(); i++) {
        shuffle_idx[i] = strided_cv_prop[i].id;
    }

    // We need the following to be saved in the shared state:
    // 1- The node indices (shuffle_idx)
    // 2- The mech_id partitioning of the node indices (mech_partition)
    // 3- The count of vector reductions per vector

    std::cout << "-----shuffle_index-------------------------\n";

    for (auto i: shuffle_idx) {
        std::cout << i << std::endl;
    }

    std::cout << "-----mech_partition-------------------------\n";

    for (auto i: mech_partition) {
        std::cout << i << std::endl;
    }

    std::cout << "------reduction_count---------------\n";

    for (auto c: util::partition_view(reduction_partition)) {
        std::cout << c.first << " - > " << c.second << std::endl;
    }

    shuffle_index  = iarray(shuffle_idx.size());
    reduction_partition = iarray(reduction_part.size());

    memory::copy(shuffle_idx, shuffle_index);
    memory::copy(reduction_part, reduction_partition);

    local_i = array(local_size);
    local_g = array(local_size);

    memory::fill(local_i, 0.0);
    memory::fill(local_g, 0.0);
}

// Debug interface
std::ostream& operator<<(std::ostream& o, shared_state& s) {
    o << " cv_to_intdom " << s.cv_to_intdom << "\n";
    o << " time         " << s.time << "\n";
    o << " time_to      " << s.time_to << "\n";
    o << " dt_intdom    " << s.dt_intdom << "\n";
    o << " dt_cv        " << s.dt_cv << "\n";
    o << " voltage      " << s.voltage << "\n";
    o << " init_voltage " << s.init_voltage << "\n";
    o << " temperature  " << s.temperature_degC << "\n";
    o << " diameter     " << s.diam_um << "\n";
    o << " current      " << s.current_density << "\n";
    o << " conductivity " << s.conductivity << "\n";
    for (auto& ki: s.ion_data) {
        auto& kn = ki.first;
        auto& i = const_cast<ion_state&>(ki.second);
        o << " " << kn << "/current_density        " << i.iX_ << "\n";
        o << " " << kn << "/reversal_potential     " << i.eX_ << "\n";
        o << " " << kn << "/internal_concentration " << i.Xi_ << "\n";
        o << " " << kn << "/external_concentration " << i.Xo_ << "\n";
        o << " " << kn << "/intconc_initial        " << i.init_Xi_ << "\n";
        o << " " << kn << "/extconc_initial        " << i.init_Xo_ << "\n";
        o << " " << kn << "/revpot_initial         " << i.init_eX_ << "\n";
        o << " " << kn << "/node_index             " << i.node_index_ << "\n";
    }
    return o;
}

} // namespace gpu
} // namespace arb
