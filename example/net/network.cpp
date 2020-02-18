#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>

#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/swcio.hpp>
#include <arbor/simulation.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/version.hpp>

#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/json_params.hpp>
#include <hdf5_lib.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

#include "parameters.hpp"

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

class net_recipe: public arb::recipe {
public:
    net_recipe(const sim_params& params):
            num_cells_(121277),
            params_(params),
            tree_(read_swc(params_.morph_file)),
            conn_data_(h5_file(params.network_file))
    {
        cell_locs_ = conn_data_["Locs"].get<std::vector<std::pair<double,double>>>("HC");

        cell_gprop_.default_parameters = arb::neuron_parameter_defaults;
        cell_gprop_.default_parameters.temperature_K = params_.temp + 273.15;
        cell_gprop_.default_parameters.init_membrane_potential = params_.v_init;
        cell_gprop_.default_parameters.ion_data["k"].init_reversal_potential = -90;
        cell_gprop_.default_parameters.ion_data["ca"].init_ext_concentration = 2;
    }

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (gid >=0 && gid < 66000) {
            //MEC cell
            std::mt19937_64 G(gid);
            return arb::spike_source_cell{arb::poisson_schedule(3.0/1000.0, G)};
        } else if (gid >=66000 && gid < 112000) {
            //LEC cell
            std::mt19937_64 G(gid);
            return arb::spike_source_cell{arb::poisson_schedule(3.0/1000.0, G)};
        } else if (gid >=112000 && gid < 113200) {
            //GC
            return granule_cell(tree_, gid);
        } else if (gid >=113200 && gid < num_cells()) {
            //BC
            return basket_cell();
        }
        return {};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        if (gid >=0 && gid < 112000) {
            return cell_kind::spike_source;
        } else if (gid >=112000 && gid < num_cells()) {
            return cell_kind::cable;
        }
        return {};
    }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override {
        return 1;
    }

    cell_size_type num_targets(cell_gid_type gid) const override {
        if (gid >=0 && gid < 112000) {
            return 0;
        } else if (gid >=112000 && gid < 113200) {
            return 3;
        } else if (gid >=113200 && gid < num_cells()) {
            return 1;
        }
        return 0;
    }

    // There is one probe (for measuring voltage at the soma) on the cell.
    cell_size_type num_probes(cell_gid_type gid) const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        arb::mlocation mid_soma = {0, 0.5};
        arb::cell_probe_address probe = {mid_soma, arb::cell_probe_address::membrane_voltage};
        return {id, 0, probe};
    }

    arb::util::any get_global_properties(cell_kind k) const override {
        return cell_gprop_;
    }

    void add_ion(const std::string& ion_name, int charge, double init_iconc, double init_econc, double init_revpot) {
        cell_gprop_.add_ion(ion_name, charge, init_iconc, init_econc, init_revpot);
    }

    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        std::lock_guard<std::mutex> l(mtx_);
        std::vector<arb::cell_connection> conns;
        if (gid >=0 && gid < 112000) {
            return {};
        } else if (gid >=112000 && gid < 113200) {
            auto dest_loc = cell_locs_[gid];
            //GC <- MEC
            {
                auto sources = conn_data_["Con"]["DG"]["GC"][std::to_string(gid)]["EC"]["MEC"].get<std::vector<int>>("middleThird");
                for (auto src: sources) {
                    auto src_loc = cell_locs_[src];
                    auto dist = distance(dest_loc, src_loc);
                    conns.emplace_back(
                            arb::cell_connection({(cell_gid_type) src, 0}, {gid, 1}, 1.52e-5, (2.4 + dist) / 0.32));
                }
            }
            //GC <- LEC
            {
                auto sources = conn_data_["Con"]["DG"]["GC"][std::to_string(gid)]["EC"]["LEC"].get<std::vector<int>>("outerThird");
                for (auto src: sources) {
                    auto src_loc = cell_locs_[src];
                    auto dist = distance(dest_loc, src_loc);
                    conns.emplace_back(
                            arb::cell_connection({(cell_gid_type) src, 0}, {gid, 2}, 1.63e-5, (2.4 + dist) / 0.32));
                }
            }
            // GC <- BC
            {
                auto sources = conn_data_["Con"]["DG"]["GC"][std::to_string(gid)]["DG"]["BC"].get<std::vector<int>>("soma");
                for (auto src: sources) {
                    auto src_loc = cell_locs_[src];
                    auto dist = distance(dest_loc, src_loc);
                    conns.emplace_back(
                            arb::cell_connection({(cell_gid_type) src, 0}, {gid, 0}, 1.074458e-03, (1.7 + dist) / 0.3));
                }
            }

        } else if (gid >=113200 && gid < num_cells()) {
            auto dest_loc = cell_locs_[gid];
            //BC <- MEC
            {
                auto sources = conn_data_["Con"]["DG"]["BC"][std::to_string(gid)]["EC"]["MEC"].get<std::vector<int>>("middleThird");
                for (auto src: sources) {
                    auto src_loc = cell_locs_[src];
                    auto dist = distance(dest_loc, src_loc);
                    conns.emplace_back(
                            arb::cell_connection({(cell_gid_type) src, 0}, {gid, 1}, 1.52e-5, (2.4 + dist) / 0.32));
                }
            }
            //BC <- LEC
            {
                auto sources = conn_data_["Con"]["DG"]["BC"][std::to_string(gid)]["EC"]["LEC"].get<std::vector<int>>("outerThird");
                for (auto src: sources) {
                    auto src_loc = cell_locs_[src];
                    auto dist = distance(dest_loc, src_loc);
                    conns.emplace_back(
                            arb::cell_connection({(cell_gid_type) src, 0}, {gid, 2}, 1.63e-5, (2.4 + dist) / 0.32));
                }
            }
            // BC <- GC
            {
                auto sources = conn_data_["Con"]["DG"]["BC"][std::to_string(gid)]["DG"]["GC"].get<std::vector<int>>("hilus");
                for (auto src: sources) {
                    auto src_loc = cell_locs_[src];
                    auto dist = distance(dest_loc, src_loc);
                    conns.emplace_back(
                            arb::cell_connection({(cell_gid_type) src, 0}, {gid, 0}, 3.298706e-05, (1.7 + dist) / 0.27));
                }
            }
        }
        return conns;
    }

private:
    cell_size_type num_cells_;
    sim_params params_;
    arb::sample_tree tree_;
    h5_wrapper conn_data_;
    std::vector<std::pair<double,double>> cell_locs_;
    arb::cable_cell_global_properties cell_gprop_;
    mutable std::mutex mtx_;
};


int main(int argc, char** argv) {
    try {
        bool root = true;

        arb::proc_allocation resources;
        if (auto nt = arbenv::get_env_num_threads()) {
            resources.num_threads = nt;
        }
        else {
            resources.num_threads = arbenv::thread_concurrency();
        }

#ifdef ARB_MPI_ENABLED
        arbenv::with_mpi guard(argc, argv, false);
        resources.gpu_id = arbenv::find_private_gpu(MPI_COMM_WORLD);
        auto context = arb::make_context(resources, MPI_COMM_WORLD);
        root = arb::rank(context) == 0;
#else
        resources.gpu_id = arbenv::default_gpu();
        auto context = arb::make_context(resources);
#endif

#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(context);
#endif

        // Print a banner with information about hardware configuration
        std::cout << "gpu:      " << (has_gpu(context)? "yes": "no") << "\n";
        std::cout << "threads:  " << num_threads(context) << "\n";
        std::cout << "mpi:      " << (has_mpi(context)? "yes": "no") << "\n";
        std::cout << "ranks:    " << num_ranks(context) << "\n" << std::endl;

        arb::profile::meter_manager meters;
        meters.start(context);

        auto params = read_params(argc, argv);

        // Create an instance of our recipe.
        net_recipe recipe(params);
        recipe.add_ion("nca", 2, 1.0, 1.0, 0);
        recipe.add_ion("lca", 2, 1.0, 1.0, 130);
        recipe.add_ion("tca", 2, 1.0, 1.0, 130);
        recipe.add_ion("nat", 1, 1.0, 1.0, 45);
        recipe.add_ion("kf",  1, 1.0, 1.0, -90);
        recipe.add_ion("ks",  1, 1.0, 1.0, -90);
        recipe.add_ion("sk",  1, 1.0, 1.0, -90);

        auto partition_network = [&recipe, &context]() {
            arb::domain_decomposition decomp;
            struct opt_ {
                opt_(unsigned nd, unsigned id, bool has_gpu): num_domains(nd), domain_id(id) {
                    backend = has_gpu ? arb::backend_kind::gpu : arb::backend_kind::multicore;
                }
                unsigned num_domains;
                unsigned domain_id;
                arb::backend_kind backend;

                unsigned mec_start = 0;
                unsigned lec_start = 66000;
                unsigned gc_start = 112000;
                unsigned bc_start = 113200;

                unsigned num_mec = 66000;
                unsigned num_lec = 46000;
                unsigned num_gc = 1200;
                unsigned num_bc = 8077;

                unsigned gc_per_domain = num_gc/num_domains;
                unsigned gc_rem            = num_gc%num_domains;

                unsigned bc_per_domain = num_bc/num_domains;
                unsigned bc_rem            = num_bc%num_domains;

                unsigned lec_per_domain = num_lec/num_domains;
                unsigned lec_rem            = num_lec%num_domains;

                unsigned mec_per_domain = num_mec/num_domains;
                unsigned mec_rem            = num_mec%num_domains;

                unsigned base_count = gc_per_domain + bc_per_domain + lec_per_domain + mec_per_domain;
                unsigned extra_count = gc_rem + bc_rem + lec_rem + mec_rem;

                unsigned num_local_cells = domain_id == 0 ? base_count + extra_count: base_count;
            } opt(num_ranks(context), rank(context), has_gpu(context));

            std::vector<arb::group_description> groups;
            for (unsigned i = opt.mec_start + opt.mec_per_domain*opt.domain_id;
                          i < opt.mec_start + opt.mec_per_domain*(opt.domain_id+1); ++i) {
                groups.push_back({cell_kind::spike_source, {(cell_gid_type)i}, opt.backend});
            }
            if (opt.domain_id == 0) {
                for (unsigned i = opt.mec_start + opt.mec_per_domain * opt.num_domains; 
                              i < opt.mec_start + opt.num_mec; ++i) {
                    groups.push_back({cell_kind::spike_source, {(cell_gid_type)i}, opt.backend});
                }
            }

            for (unsigned i = opt.lec_start + opt.lec_per_domain * opt.domain_id;
                          i < opt.lec_start + opt.lec_per_domain * (opt.domain_id + 1); ++i) {
                groups.push_back({cell_kind::spike_source, {(cell_gid_type)i}, opt.backend});
            }
            if (opt.domain_id == 0) {
                for (unsigned i = opt.lec_start + opt.lec_per_domain * opt.num_domains; 
                              i < opt.lec_start + opt.num_lec; ++i) {
                    groups.push_back({cell_kind::spike_source, {(cell_gid_type)i}, opt.backend});
                }
            }

            for (unsigned i = opt.gc_start + opt.gc_per_domain * opt.domain_id;
                          i < opt.gc_start + opt.gc_per_domain * (opt.domain_id + 1); ++i) {
                groups.push_back({cell_kind::cable, {(cell_gid_type)i}, opt.backend});
            }
            if (opt.domain_id == 0) {
                for (unsigned i = opt.gc_start + opt.gc_per_domain * opt.num_domains; 
                              i < opt.gc_start + opt.num_gc; ++i) {
                    groups.push_back({cell_kind::cable, {(cell_gid_type)i}, opt.backend});
                }
            }

            for (unsigned i = opt.bc_start + opt.bc_per_domain * opt.domain_id;
                          i < opt.bc_start + opt.bc_per_domain * (opt.domain_id + 1); ++i) {
                groups.push_back({cell_kind::cable, {(cell_gid_type)i}, opt.backend});
            }
            if (opt.domain_id == 0) {
                for (unsigned i = opt.bc_start + opt.bc_per_domain * opt.num_domains; 
                              i < opt.bc_start + opt.num_bc; ++i) {
                    groups.push_back({cell_kind::cable, {(cell_gid_type)i}, opt.backend});
                }
            }

            struct partition_gid_domain {
                partition_gid_domain(const opt_& opt): opt(opt) {}

                int operator()(cell_gid_type gid) const {
                    int dom = 0;
                    if (gid >= opt.mec_start && gid < opt.lec_start) {
                        dom = ((gid - opt.mec_start)/opt.mec_per_domain);
                    } else if (gid >= opt.lec_start && gid < opt.gc_start) {
                        dom = ((gid - opt.lec_start)/opt.lec_per_domain);
                    } else if (gid >= opt.gc_start && gid < opt.bc_start) {
                        dom = ((gid - opt.gc_start)/opt.gc_per_domain);
                    } else if (gid >= opt.bc_start && gid){
                        dom = ((gid - opt.bc_start)/opt.bc_per_domain);
                    }
                    return dom >= opt.num_domains ? 0 : dom;
                }

                opt_ opt;
            };

            decomp.num_domains = opt.num_domains;
            decomp.domain_id = opt.domain_id;
            decomp.num_local_cells = opt.num_local_cells;
            decomp.num_global_cells = 121277;
            decomp.groups = std::move(groups);
            decomp.gid_domain = partition_gid_domain(opt);

            return decomp;
        };

        auto decomp = partition_network();

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);
        sim.set_binning_policy(arb::binning_kind::regular, params.dt);
        if (root) std::cout << params.run_time << std::endl;
        // Set up the probe that will measure voltage in the cell.

        // The id of the only probe on the cell: the cell_member type points to (cell 0, probe 0)
        auto probe_id = cell_member_type{112000, 0};

        auto sched = arb::regular_schedule(params.dt);

        // This is where the voltage samples will be stored as (time, value) pairs
        arb::trace_data<double> voltage;

        // Now attach the sampler at probe_id, with sampling schedule sched, writing to voltage
        sim.add_sampler(arb::one_probe(probe_id), sched, arb::make_simple_sampler(voltage));

        // Set up recording of spikes to a vector on the root process.
        std::vector<arb::spike> recorded_spikes;
        if (root) {
            sim.set_global_spike_callback(
                    [&recorded_spikes](const std::vector<arb::spike>& spikes) {
                        recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                    });
        }

        meters.checkpoint("model-init", context);

        std::cout << "running simulation" << std::endl;

        sim.run(params.run_time, params.dt);

        meters.checkpoint("model-run", context);

        auto ns = sim.num_spikes();

        // Write spikes to file
        if (root) {
            std::cout << "\n" << ns << " spikes generated at rate of "
                      << params.run_time << " ms between spikes\n";
            std::ofstream fid("spikes.gdf");
            if (!fid.good()) {
                std::cerr << "Warning: unable to open file spikes.gdf for spike output\n";
            }
            else {
                char linebuf[45];
                for (auto spike: recorded_spikes) {
                    auto n = std::snprintf(
                            linebuf, sizeof(linebuf), "%u %.4f\n",
                            unsigned{spike.source.gid}, float(spike.time));
                    fid.write(linebuf, n);
                }
            }
        }

        auto profile = arb::profile::profiler_summary();
        std::cout << profile << "\n";

        // Write the samples to a json file.
        if (decomp.gid_domain(112000) == arb::rank(context)) {
           std::cout << "rank " << arb::rank(context) << " is writing the voltages" << std::endl;
           write_trace_json(voltage);
        }

        auto report = arb::profile::make_meter_report(meters, context);
        std::cout << report;
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in granule miniapp: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
