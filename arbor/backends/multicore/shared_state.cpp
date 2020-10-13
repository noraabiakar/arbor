#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/constants.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/math.hpp>
#include <arbor/simd/simd.hpp>

#include "backends/event.hpp"
#include "io/sepval.hpp"
#include "util/padded_alloc.hpp"
#include "util/rangeutil.hpp"

#include "multi_event_stream.hpp"
#include "multicore_common.hpp"
#include "shared_state.hpp"

namespace arb {
namespace multicore {

constexpr unsigned vector_length = (unsigned) simd::simd_abi::native_width<fvm_value_type>::value;
using simd_value_type = simd::simd<fvm_value_type, vector_length, simd::simd_abi::default_abi>;
using simd_index_type = simd::simd<fvm_index_type, vector_length, simd::simd_abi::default_abi>;
const int simd_width  = simd::width<simd_value_type>();

// Pick alignment compatible with native SIMD width for explicitly
// vectorized operations below.
//
// TODO: Is SIMD use here a win? Test and compare; may be better to leave
// these up to the compiler to optimize/auto-vectorize.

inline unsigned min_alignment(unsigned align) {
    unsigned simd_align = sizeof(fvm_value_type)*simd_width;
    return math::next_pow2(std::max(align, simd_align));
}

using pad = util::padded_allocator<>;


// ion_state methods:

ion_state::ion_state(
    int charge,
    const fvm_ion_config& ion_data,
    unsigned align
):
    alignment(min_alignment(align)),
    node_index_(ion_data.cv.begin(), ion_data.cv.end(), pad(alignment)),
    iX_(ion_data.cv.size(), NAN, pad(alignment)),
    eX_(ion_data.init_revpot.begin(), ion_data.init_revpot.end(), pad(alignment)),
    Xi_(ion_data.cv.size(), NAN, pad(alignment)),
    Xo_(ion_data.cv.size(), NAN, pad(alignment)),
    init_Xi_(ion_data.init_iconc.begin(), ion_data.init_iconc.end(), pad(alignment)),
    init_Xo_(ion_data.init_econc.begin(), ion_data.init_econc.end(), pad(alignment)),
    reset_Xi_(ion_data.reset_iconc.begin(), ion_data.reset_iconc.end(), pad(alignment)),
    reset_Xo_(ion_data.reset_econc.begin(), ion_data.reset_econc.end(), pad(alignment)),
    init_eX_(ion_data.init_revpot.begin(), ion_data.init_revpot.end(), pad(alignment)),
    charge(1u, charge, pad(alignment))
{
    arb_assert(node_index_.size()==init_Xi_.size());
    arb_assert(node_index_.size()==init_Xo_.size());
    arb_assert(node_index_.size()==eX_.size());
    arb_assert(node_index_.size()==init_eX_.size());
}

void ion_state::init_concentration() {
    std::copy(init_Xi_.begin(), init_Xi_.end(), Xi_.begin());
    std::copy(init_Xo_.begin(), init_Xo_.end(), Xo_.begin());
}

void ion_state::zero_current() {
    util::fill(iX_, 0);
}

void ion_state::reset() {
    zero_current();
    std::copy(reset_Xi_.begin(), reset_Xi_.end(), Xi_.begin());
    std::copy(reset_Xo_.begin(), reset_Xo_.end(), Xo_.begin());
    std::copy(init_eX_.begin(), init_eX_.end(), eX_.begin());
}

// shared_state methods:

shared_state::shared_state(
    fvm_size_type n_intdom,
    const std::vector<fvm_index_type>& cv_to_intdom_vec,
    const std::vector<fvm_gap_junction>& gj_vec,
    const std::vector<fvm_value_type>& init_membrane_potential,
    const std::vector<fvm_value_type>& temperature_K,
    const std::vector<fvm_value_type>& diam,
    unsigned align
):
    alignment(min_alignment(align)),
    alloc(alignment),
    n_intdom(n_intdom),
    n_cv(cv_to_intdom_vec.size()),
    n_gj(gj_vec.size()),
    cv_to_intdom(math::round_up(n_cv, alignment), pad(alignment)),
    gap_junctions(math::round_up(n_gj, alignment), pad(alignment)),
    time(n_intdom, pad(alignment)),
    time_to(n_intdom, pad(alignment)),
    dt_intdom(n_intdom, pad(alignment)),
    dt_cv(n_cv, pad(alignment)),
    voltage(n_cv, pad(alignment)),
    current_density(n_cv, pad(alignment)),
    conductivity(n_cv, pad(alignment)),
    init_voltage(init_membrane_potential.begin(), init_membrane_potential.end(), pad(alignment)),
    temperature_degC(n_cv, pad(alignment)),
    diam_um(diam.begin(), diam.end(), pad(alignment)),
    deliverable_events(n_intdom)
{
    // For indices in the padded tail of cv_to_intdom, set index to last valid intdom index.
    if (n_cv>0) {
        std::copy(cv_to_intdom_vec.begin(), cv_to_intdom_vec.end(), cv_to_intdom.begin());
        std::fill(cv_to_intdom.begin() + n_cv, cv_to_intdom.end(), cv_to_intdom_vec.back());
    }
    if (n_gj>0) {
        std::copy(gj_vec.begin(), gj_vec.end(), gap_junctions.begin());
        std::fill(gap_junctions.begin()+n_gj, gap_junctions.end(), gj_vec.back());
    }

    for (unsigned i = 0; i<n_cv; ++i) {
        temperature_degC[i] = temperature_K[i] - 273.15;
    }
}

void shared_state::add_ion(
    const std::string& ion_name,
    int charge,
    const fvm_ion_config& ion_info)
{
    ion_data.emplace(std::piecewise_construct,
        std::forward_as_tuple(ion_name),
        std::forward_as_tuple(charge, ion_info, alignment));
}

void shared_state::reset() {
    std::copy(init_voltage.begin(), init_voltage.end(), voltage.begin());
    util::fill(current_density, 0);
    util::fill(conductivity, 0);
    util::fill(time, 0);
    util::fill(time_to, 0);

    for (auto& i: ion_data) {
        i.second.reset();
    }
}

void shared_state::zero_locals() {
    util::fill(local_i, 0);
    util::fill(local_g, 0);
}

void shared_state::zero_currents() {
    zero_locals();
    util::fill(current_density, 0);
    util::fill(conductivity, 0);
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
    using simd::assign;
    using simd::indirect;
    using simd::add;
    using simd::min;
    for (fvm_size_type i = 0; i<n_intdom; i+=simd_width) {
        simd_value_type t;
        assign(t, indirect(time.data()+i, simd_width));
        t = min(add(t, dt_step), tmax);
        indirect(time_to.data()+i, simd_width) = t;
    }
}

void shared_state::set_dt() {
    using simd::assign;
    using simd::indirect;
    using simd::sub;
    for (fvm_size_type j = 0; j<n_intdom; j+=simd_width) {
        simd_value_type t, t_to;
        assign(t, indirect(time.data()+j, simd_width));
        assign(t_to, indirect(time_to.data()+j, simd_width));

        auto dt = sub(t_to,t);
        indirect(dt_intdom.data()+j, simd_width) = dt;
    }

    for (fvm_size_type i = 0; i<n_cv; i+=simd_width) {
        simd_index_type intdom_idx;
        assign(intdom_idx, indirect(cv_to_intdom.data()+i, simd_width));

        simd_value_type dt;
        assign(dt, indirect(dt_intdom.data(), intdom_idx, simd_width));
        indirect(dt_cv.data()+i, simd_width) = dt;
    }
}

void shared_state::add_gj_current() {
    for (unsigned i = 0; i < n_gj; i++) {
        auto gj = gap_junctions[i];
        auto curr = gj.weight *
                    (voltage[gj.loc.second] - voltage[gj.loc.first]); // nA

        current_density[gj.loc.first] -= curr;
    }
}

std::pair<fvm_value_type, fvm_value_type> shared_state::time_bounds() const {
    return util::minmax_value(time);
}

std::pair<fvm_value_type, fvm_value_type> shared_state::voltage_bounds() const {
    return util::minmax_value(voltage);
}

void shared_state::take_samples(
    const sample_event_stream::state& s,
    array& sample_time,
    array& sample_value)
{
    for (fvm_size_type i = 0; i<s.n_streams(); ++i) {
        auto begin = s.begin_marked(i);
        auto end = s.end_marked(i);

        // (Note: probably not worth explicitly vectorizing this.)
        for (auto p = begin; p<end; ++p) {
            sample_time[p->offset] = time[i];
            sample_value[p->offset] = *p->handle;
        }
    }
}

void shared_state::build_cv_index(std::vector<std::pair<unsigned, std::vector<fvm_index_type>>> mech_cv) {
    const int vsize = 4;
    if (mech_cv.empty()) return;
    struct cv_prop {
        int cv_idx;
        int mech_id;
        int vec_idx;
    };

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
            mech_cv_props.push_back({cv, id, -1});
        }
    }

    auto comp     = [](auto& lhs, auto& rhs) {return std::tie(lhs.cv_idx, lhs.mech_id, lhs.vec_idx) <
                                                     std::tie(rhs.cv_idx, rhs.mech_id, rhs.vec_idx);};
    auto comp_rev = [](auto& lhs, auto& rhs) {return std::tie(lhs.mech_id, lhs.cv_idx, lhs.vec_idx) <
                                                     std::tie(rhs.mech_id, rhs.cv_idx, rhs.vec_idx);};
    if (mech_partition.size() > 2) {
        for (unsigned i = 2; i < mech_partition.size(); ++i) {
            auto begin = mech_cv_props.begin();
            auto middle = begin + mech_partition[i-1];
            auto last   = begin + mech_partition[i];

            std::inplace_merge(begin, middle, last, comp);
        }
    };

    // GOOD SO FAR

    std::vector<std::vector<int>> cv_idx_vectors, idx_partition_vectors;

    std::vector<int> cv_vec;
    int prev_cv=mech_cv_props.front().cv_idx;

    for (unsigned i = 1; i < mech_cv_props.size(); ++i) {
        const auto cv = mech_cv_props[i].cv_idx;
        if (cv != prev_cv) {
            cv_vec.push_back(prev_cv);
            prev_cv = cv;
        }
        if (cv_vec.size() == vsize) {
            cv_idx_vectors.push_back(std::move(cv_vec));
            cv_vec.clear();
            prev_cv = cv;
        }
    }
    cv_vec.push_back(prev_cv);
    cv_idx_vectors.push_back(std::move(cv_vec));

    for (auto cv_idx: cv_idx_vectors) {
        std::vector<int> idx_vec;
        for (auto cv: cv_idx) {
            auto it = std::lower_bound(mech_cv_props.begin(), mech_cv_props.end(), cv, [](auto& lhs, auto& rhs) { return lhs.cv_idx < rhs; });
            idx_vec.push_back(it - mech_cv_props.begin());
        }
        auto it = std::upper_bound(mech_cv_props.begin(), mech_cv_props.end(), cv_idx.back(), [](auto& lhs, auto& rhs) { return lhs < rhs.cv_idx; });
        while (idx_vec.size() < vsize+1) idx_vec.push_back(it - mech_cv_props.begin());
        idx_partition_vectors.push_back(idx_vec);
    }
    
    std::cout << "------HERE--------\n";

    for (auto c: cv_idx_vectors) {
        for (auto cv: c) {
            std::cout << cv << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "------HERE--------\n";

    for (auto c: idx_partition_vectors) {
        for (auto cv: c) {
            std::cout << cv << " ";
        }
        std::cout << std::endl;
    }

    // SURE 

    std::vector<cv_prop> shuffle;
    auto idx_ptr_vectors = idx_partition_vectors;

    for (unsigned i = 0; i < idx_ptr_vectors.size(); ++i) {
        auto& idx_ptr       = idx_ptr_vectors[i];
        const auto& idx_cst = idx_partition_vectors[i];

        std::set<unsigned> hit_idx;
        while (true) {
            std::vector<cv_prop> mini_shuffle;
            for (unsigned j = 0; j < idx_ptr.size() - 1; ++j) {
                auto& start_idx = idx_ptr[j];
                auto end_idx = idx_cst[j + 1];
                if (start_idx == end_idx) {
                    hit_idx.insert(j);
                    mini_shuffle.push_back({-1, -1, -1});
                } else {
                    mini_shuffle.push_back(mech_cv_props[start_idx++]);
                }
            }
            if (hit_idx.size() == vsize) break;
            std::move(mini_shuffle.begin(), mini_shuffle.end(), std::back_inserter(shuffle));
        }
    }

    std::cout << "------HERE--------\n";

    for (auto c: shuffle) {
        std::cout << c.cv_idx << "\t" << c.mech_id << "\t" << c.vec_idx << std::endl;
    }

    // YEAH I GUESS ? Now fill and shuffle
    int idx = 0;
    for (auto& s: shuffle) {
        s.vec_idx = idx++;
    }
    std::sort(shuffle.begin(), shuffle.end(), comp_rev);

    auto it = shuffle.begin();
    for (; it != shuffle.end(); ++it) {
        if((*it).mech_id != -1) break;
    }
    shuffle.erase(shuffle.begin(), it);

    std::cout << "--------------\n";

    for (auto i: shuffle) {
        std::cout << i.vec_idx << "\t" << i.mech_id << "\t" << i.cv_idx << std::endl;
    }

    std::cout << "--------------\n";


    // Next we need the actual updated CVs to be stored somewhere,
    // We need to calculate the limits for mech indices,
    // We need to calculate how many vector additions per vector of CVs
    // ex
    // CV      : | 0 1 2 5 | 6 7 9 10 | 11 12 - - |
    // num adds: |    3    |   2      | 1         |
    local_i = array(mech_cv_props.size(), pad(alignment));
    local_g = array(mech_cv_props.size(), pad(alignment));
}

void shared_state::reduce() {
    auto partition = util::partition_view(node_partition);

    unsigned i = 0;
    fvm_value_type sum_i, sum_g;
    for (auto range: partition) {
        sum_i = 0.0;
        sum_g = 0.0;
        for (auto j = range.first; j < range.second; ++j) {
            sum_i += local_i[j];
            sum_g += local_g[j];
        }
        current_density[i] += sum_i;
        conductivity[i] += sum_g;
        i++;
    }
}

// (Debug interface only.)
std::ostream& operator<<(std::ostream& out, const shared_state& s) {
    using io::csv;

    out << "n_intdom     " << s.n_intdom << "\n";
    out << "n_cv         " << s.n_cv << "\n";
    out << "cv_to_intdom " << csv(s.cv_to_intdom) << "\n";
    out << "time         " << csv(s.time) << "\n";
    out << "time_to      " << csv(s.time_to) << "\n";
    out << "dt_intdom    " << csv(s.dt_intdom) << "\n";
    out << "dt_cv        " << csv(s.dt_cv) << "\n";
    out << "voltage      " << csv(s.voltage) << "\n";
    out << "init_voltage " << csv(s.init_voltage) << "\n";
    out << "temperature  " << csv(s.temperature_degC) << "\n";
    out << "diameter     " << csv(s.diam_um) << "\n";
    out << "current      " << csv(s.current_density) << "\n";
    out << "conductivity " << csv(s.conductivity) << "\n";
    for (const auto& ki: s.ion_data) {
        auto& kn = ki.first;
        auto& i = const_cast<ion_state&>(ki.second);
        out << kn << "/current_density        " << csv(i.iX_) << "\n";
        out << kn << "/reversal_potential     " << csv(i.eX_) << "\n";
        out << kn << "/internal_concentration " << csv(i.Xi_) << "\n";
        out << kn << "/external_concentration " << csv(i.Xo_) << "\n";
        out << kn << "/intconc_initial        " << csv(i.init_Xi_) << "\n";
        out << kn << "/extconc_initial        " << csv(i.init_Xo_) << "\n";
        out << kn << "/revpot_initial         " << csv(i.init_eX_) << "\n";
        out << kn << "/node_index             " << csv(i.node_index_) << "\n";
    }

    return out;
}

} // namespace multicore
} // namespace arb
