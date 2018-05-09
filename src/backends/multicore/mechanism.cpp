#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <backends/fvm_types.hpp>
#include <common_types.hpp>

#include <math.hpp>
#include <mechanism.hpp>
#include <util/index_into.hpp>
#include <util/optional.hpp>
#include <util/maputil.hpp>
#include <util/padded_alloc.hpp>
#include <util/range.hpp>

#include <backends/multicore/mechanism.hpp>
#include <backends/multicore/multicore_common.hpp>
#include <backends/multicore/fvm.hpp>
#include <backends/multicore/partition_by_constraint.hpp>

namespace arb {
namespace multicore {

using util::make_range;
using util::value_by_key;

// Copy elements from source sequence into destination sequence,
// and fill the remaining elements of the destination sequence
// with the given fill value.
//
// Assumes that the iterators for these sequences are at least
// forward iterators.

template <typename Source, typename Dest, typename Fill>
void copy_extend(const Source& source, Dest&& dest, const Fill& fill) {
    using std::begin;
    using std::end;

    auto dest_n = util::size(dest);
    auto source_n = util::size(source);

    auto n = source_n<dest_n? source_n: dest_n;
    auto tail = std::copy_n(begin(source), n, begin(dest));
    std::fill(tail, end(dest), fill);
}

// The derived class (typically generated code from modcc) holds pointers that need
// to be set to point inside the shared state, or into the allocated parameter/variable
// data block.
//
// In ths SIMD case, there may be a 'tail' of values that correspond to a partial
// SIMD value when the width is not a multiple of the SIMD data width. In this
// implementation we do not use SIMD masking to avoid tail values, but instead
// extend the vectors to a multiple of the SIMD width: sites/CVs corresponding to
// these past-the-end values are given a weight of zero, and any corresponding
// indices into shared state point to the last valid slot.

void mechanism::instantiate(fvm_size_type id, backend::shared_state& shared, const layout& pos_data) {
    using util::make_range;

    util::padded_allocator<> pad(shared.alignment);
    mechanism_id_ = id;
    width_ = pos_data.cv.size();

    // Assign non-owning views onto shared state:

    vec_ci_   = shared.cv_to_cell.data();
    vec_t_    = shared.time.data();
    vec_t_to_ = shared.time_to.data();
    vec_dt_   = shared.dt_cv.data();

    vec_v_    = shared.voltage.data();
    vec_i_    = shared.current_density.data();

    auto ion_state_tbl = ion_state_table();
    n_ion_ = ion_state_tbl.size();
    for (auto i: ion_state_tbl) {
        util::optional<ion_state&> oion = value_by_key(shared.ion_data, i.first);
        if (!oion) {
            throw std::logic_error("mechanism holds ion with no corresponding shared state");
        }

        ion_state_view& ion_view = *i.second;
        ion_view.current_density = oion->iX_.data();
        ion_view.reversal_potential = oion->eX_.data();
        ion_view.internal_concentration = oion->Xi_.data();
        ion_view.external_concentration = oion->Xo_.data();
    }

    event_stream_ptr_ = &shared.deliverable_events;

    // If there are no sites (is this ever meaningful?) there is nothing more to do.
    if (width_==0) {
        return;
    }

    // Extend width to account for requisite SIMD padding.
    width_padded_ = math::round_up(width_, shared.alignment);

    // Allocate and initialize state and parameter vectors with default values.

    auto fields = field_table();
    std::size_t n_field = fields.size();

    // (First sub-array of data_ is used for width_, below.)
    data_ = array((1+n_field)*width_padded_, NAN, pad);
    for (std::size_t i = 0; i<n_field; ++i) {
        // Take reference to corresponding derived (generated) mechanism value pointer member.
        fvm_value_type*& field_ptr = *(fields[i].second);
        field_ptr = data_.data()+(i+1)*width_padded_;

        if (auto opt_value = value_by_key(field_default_table(), fields[i].first)) {
            std::fill(field_ptr, field_ptr+width_padded_, *opt_value);
        }
    }
    weight_ = data_.data();

    // Allocate and copy local state: weight, node indices, ion indices.
    // The tail comprises those elements between width_ and width_padded_:
    //
    // * For entries in the padded tail of weight_, set weight to zero.
    // * For indices in the padded tail of node_index_, set index to last valid CV index.
    // * For indices in the padded tail of ion index maps, set index to last valid ion index.

    node_index_ = iarray(width_padded_, pad);
    copy_extend(pos_data.cv, node_index_, pos_data.cv.back());

    //TODO: call function that returns std::vector<index_constraint> and copy into index_constraint_
    gen_constraint(node_index_, constraint_index_);

    copy_extend(pos_data.weight, make_range(data_.data(), data_.data()+width_padded_), 0);

    for (auto i: ion_index_table()) {
        util::optional<ion_state&> oion = value_by_key(shared.ion_data, i.first);
        if (!oion) {
            throw std::logic_error("mechanism holds ion with no corresponding shared state");
        }

        auto indices = util::index_into(node_index_, oion->node_index_);

        // Take reference to derived (generated) mechanism ion index member.
        auto& ion_index = *i.second;
        ion_index = iarray(width_padded_, pad);
        copy_extend(indices, ion_index, util::back(indices));
    }

}

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (auto opt_ptr = value_by_key(field_table(), key)) {
        if (values.size()!=width_) {
            throw std::logic_error("internal error: mechanism parameter size mismatch");
        }

        if (width_>0) {
            // Retrieve corresponding derived (generated) mechanism value pointer member.
            value_type* field_ptr = *opt_ptr.value();
            util::range<value_type*> field(field_ptr, field_ptr+width_padded_);

            copy_extend(values, field, values.back());
        }
    }
    else {
        throw std::logic_error("internal error: no such mechanism parameter");
    }
}

void mechanism::set_global(const std::string& key, fvm_value_type value) {
    if (auto opt_ptr = value_by_key(global_table(), key)) {
        // Take reference to corresponding derived (generated) mechanism value member.
        value_type& global = *opt_ptr.value();
        global = value;
    }
    else {
        throw std::logic_error("internal error: no such mechanism global");
    }
}

} // namespace multicore
} // namespace arb
