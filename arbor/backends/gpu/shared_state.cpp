#include <cstddef>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/ion.hpp>

#include "backends/event.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/shared_state.hpp"
#include "backends/multi_event_stream_state.hpp"
#include "memory/wrappers.hpp"
#include "util/rangeutil.hpp"

using arb::memory::make_const_view;

namespace arb {
namespace gpu {

// CUDA implementation entry points:

void init_concentration_impl(
    std::size_t n, fvm_value_type* Xi, fvm_value_type* Xo, const fvm_value_type* weight_Xi,
    const fvm_value_type* weight_Xo, fvm_value_type iconc, fvm_value_type econc, cudaStream_t* stream);

void nernst_impl(
    std::size_t n, fvm_value_type factor,
    const fvm_value_type* Xi, const fvm_value_type* Xo, fvm_value_type* eX,
    cudaStream_t* stream);

void update_time_to_impl(
    std::size_t n, fvm_value_type* time_to, const fvm_value_type* time,
    fvm_value_type dt, fvm_value_type tmax, cudaStream_t* stream);

void set_dt_impl(
    fvm_size_type ncell, fvm_size_type ncomp, fvm_value_type* dt_cell, fvm_value_type* dt_comp,
    const fvm_value_type* time_to, const fvm_value_type* time, const fvm_index_type* cv_to_cell,
    cudaStream_t* stream);

void take_samples_impl(
    const multi_event_stream_state<raw_probe_info>& s,
    const fvm_value_type* time, fvm_value_type* sample_time, fvm_value_type* sample_value,
    cudaStream_t* stream);

// GPU-side minmax: consider CUDA kernel replacement.
std::pair<fvm_value_type, fvm_value_type> minmax_value_impl(fvm_size_type n, const fvm_value_type* v) {
    auto v_copy = memory::on_host(memory::const_device_view<fvm_value_type>(v, n));
    return util::minmax_value(v_copy);
}

// Ion state methods:

ion_state::ion_state(
    ion_info info,
    const std::vector<fvm_index_type>& cv,
    const std::vector<fvm_value_type>& iconc_norm_area,
    const std::vector<fvm_value_type>& econc_norm_area,
    unsigned, // alignment/padding ignored.
    gpu_context_handle gpu_context
):
    node_index_(make_const_view(cv)),
    iX_(cv.size(), NAN),
    eX_(cv.size(), NAN),
    Xi_(cv.size(), NAN),
    Xo_(cv.size(), NAN),
    weight_Xi_(make_const_view(iconc_norm_area)),
    weight_Xo_(make_const_view(econc_norm_area)),
    charge(info.charge),
    default_int_concentration(info.default_int_concentration),
    default_ext_concentration(info.default_ext_concentration),
    gpu_context_(gpu_context)
{
    arb_assert(node_index_.size()==weight_Xi_.size());
    arb_assert(node_index_.size()==weight_Xo_.size());
}

void ion_state::nernst(fvm_value_type temperature_K) {
    // Nernst equation: reversal potenial eX given by:
    //
    //     eX = RT/zF * ln(Xo/Xi)
    //
    // where:
    //     R: universal gas constant 8.3144598 J.K-1.mol-1
    //     T: temperature in Kelvin
    //     z: valency of species (K, Na: +1) (Ca: +2)
    //     F: Faraday's constant 96485.33289 C.mol-1
    //     Xo/Xi: ratio of out/in concentrations

    // 1e3 factor required to scale from V -> mV.
    constexpr fvm_value_type RF = 1e3*constant::gas_constant/constant::faraday;

    fvm_value_type factor = RF*temperature_K/charge;
    nernst_impl(Xi_.size(), factor, Xo_.data(), Xi_.data(), eX_.data(),
            gpu_context_->get_thread_stream(std::this_thread::get_id()));
}

void ion_state::init_concentration() {
    init_concentration_impl(
        Xi_.size(),
        Xi_.data(), Xo_.data(),
        weight_Xi_.data(), weight_Xo_.data(),
        default_int_concentration, default_ext_concentration,
        gpu_context_->get_thread_stream(std::this_thread::get_id()));
}

void ion_state::zero_current() {
    memory::fill(iX_, 0);
}

void ion_state::zero_current(cudaStream_t* stream) {
    auto iX_v = make_view(iX_);
    using iX_t = std::decay_t<array>;

    if (iX_v.size()) {
        arb::gpu::fill<iX_t::value_type>(iX_v.data(), 0, iX_v.size(), stream);
    }
}

// Shared state methods:

shared_state::shared_state(
    fvm_size_type n_cell,
    const std::vector<fvm_index_type>& cv_to_cell_vec,
    unsigned, // alignment parameter ignored.
    execution_context context
):
    n_cell(n_cell),
    n_cv(cv_to_cell_vec.size()),
    cv_to_cell(make_const_view(cv_to_cell_vec)),
    time(n_cell),
    time_to(n_cell),
    dt_cell(n_cell),
    dt_cv(n_cv),
    voltage(n_cv),
    current_density(n_cv),
    gpu_context(context.gpu),
    deliverable_events(n_cell, context.gpu)
{}

void shared_state::add_ion(
    ion_info info,
    const std::vector<fvm_index_type>& cv,
    const std::vector<fvm_value_type>& iconc_norm_area,
    const std::vector<fvm_value_type>& econc_norm_area)
{
    ion_data.emplace(std::piecewise_construct,
        std::forward_as_tuple(info.kind),
        std::forward_as_tuple(info, cv, iconc_norm_area, econc_norm_area, 1u, gpu_context));
}

void shared_state::reset(fvm_value_type initial_voltage, fvm_value_type temperature_K) {
    memory::fill(voltage, initial_voltage);
    memory::fill(current_density, 0);
    memory::fill(time, 0);
    memory::fill(time_to, 0);

    for (auto& i: ion_data) {
        i.second.reset(temperature_K);
    }
}

void shared_state::zero_currents() {
    auto current_density_v = make_view(current_density);
    using current_t = std::decay_t<array>;

    if (current_density_v.size()) {
        arb::gpu::fill<current_t::value_type>(current_density_v.data(), 0, current_density_v.size(), 
                                              gpu_context->get_thread_stream(std::this_thread::get_id()));
    }
    for (auto& i: ion_data) {
        i.second.zero_current(gpu_context->get_thread_stream(std::this_thread::get_id()));
    }
}

void shared_state::ions_init_concentration() {
    for (auto& i: ion_data) {
        i.second.init_concentration();
    }
}

void shared_state::ions_nernst_reversal_potential(fvm_value_type temperature_K) {
    for (auto& i: ion_data) {
        i.second.nernst(temperature_K);
    }
}

void shared_state::update_time_to(fvm_value_type dt_step, fvm_value_type tmax) {
    update_time_to_impl(n_cell, time_to.data(), time.data(), dt_step, tmax,
            gpu_context->get_thread_stream(std::this_thread::get_id()));
}

void shared_state::set_dt() {
    set_dt_impl(n_cell, n_cv, dt_cell.data(), dt_cv.data(), time_to.data(), time.data(), cv_to_cell.data(),
                gpu_context->get_thread_stream(std::this_thread::get_id()));
}

std::pair<fvm_value_type, fvm_value_type> shared_state::time_bounds() const {
    return minmax_value_impl(n_cell, time.data());
}

std::pair<fvm_value_type, fvm_value_type> shared_state::voltage_bounds() const {
    return minmax_value_impl(n_cv, voltage.data());
}

void shared_state::take_samples(const sample_event_stream::state& s, array& sample_time, array& sample_value) {
    take_samples_impl(s, time.data(), sample_time.data(), sample_value.data(),
            gpu_context->get_thread_stream(std::this_thread::get_id()));
}

// Debug interface
std::ostream& operator<<(std::ostream& o, shared_state& s) {
    o << " cv_to_cell " << s.cv_to_cell << "\n";
    o << " time       " << s.time << "\n";
    o << " time_to    " << s.time_to << "\n";
    o << " dt_cell    " << s.dt_cell << "\n";
    o << " dt_cv      " << s.dt_cv << "\n";
    o << " voltage    " << s.voltage << "\n";
    o << " current    " << s.current_density << "\n";
    for (auto& ki: s.ion_data) {
        auto kn = to_string(ki.first);
        auto& i = const_cast<ion_state&>(ki.second);
        o << " " << kn << ".current_density        " << i.iX_ << "\n";
        o << " " << kn << ".reversal_potential     " << i.eX_ << "\n";
        o << " " << kn << ".internal_concentration " << i.Xi_ << "\n";
        o << " " << kn << ".external_concentration " << i.Xo_ << "\n";
        o << " " << kn << ".node_index             " << i.node_index_ << "\n";
    }
    return o;
}

} // namespace gpu
} // namespace arb
