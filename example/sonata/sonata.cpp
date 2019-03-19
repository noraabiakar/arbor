#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>

#include "parameters.hpp"
#include "data_management_lib.hpp"

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

// Generate a cell.
arb::cable_cell dummy_cell(
        std::vector<std::pair<arb::segment_location, double>>,
        std::vector<std::pair<arb::segment_location, arb::mechanism_desc>>);

void write_trace_json(const arb::trace_data<double>& trace);

class sonata_recipe: public arb::recipe {
public:
    sonata_recipe(hdf5_record nodes, hdf5_record edges, csv_record node_types, csv_record edge_types):
            database_(nodes, edges, node_types, edge_types),
            num_cells_(database_.num_cells()) {}

    //sonata_recipe(): num_cells_(12) {}

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {

        std::vector<std::pair<arb::segment_location,double>> src_types;
        std::vector<std::pair<arb::segment_location,arb::mechanism_desc>> tgt_types;

        std::lock_guard<std::mutex> l(mtx_);
        database_.get_sources_and_targets(gid, src_types, tgt_types);
        /*switch(gid) {
            case 0: {
                src_types.push_back(std::make_pair(arb::segment_location(0, 0.0), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.1), 10));
                src_types.push_back(std::make_pair(arb::segment_location(2, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                break;
            }
            case 1: {
                src_types.push_back(std::make_pair(arb::segment_location(0, 0.3), 10));
                src_types.push_back(std::make_pair(arb::segment_location(2, 0.4), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.5), 10));
                break;
            }
            case 2: {
                src_types.push_back(std::make_pair(arb::segment_location(0, 0.6), 10));
                src_types.push_back(std::make_pair(arb::segment_location(2, 0.7), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.2), 10));
                break;
            }
            case 3: {
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.8), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.9), 10));
                break;
            }
            case 4: {
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.8), 10));
                src_types.push_back(std::make_pair(arb::segment_location(2, 0.2), 10));
                src_types.push_back(std::make_pair(arb::segment_location(0, 0.6), 10));
                break;
            }
            case 5: {
                src_types.push_back(std::make_pair(arb::segment_location(2, 0.3), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.4), 10));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                break;
            }
            case 6: {
                src_types.push_back(std::make_pair(arb::segment_location(2, 0.3), 10));
                src_types.push_back(std::make_pair(arb::segment_location(0, 0.2), 10));
                break;
            }
            case 7: {
                src_types.push_back(std::make_pair(arb::segment_location(0, 0.1), 10));
                src_types.push_back(std::make_pair(arb::segment_location(1, 0.0), 10));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.5), arb::mechanism_desc("expsyn")));
                break;
            }
            case 8: {
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.01), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.02), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.06), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(1, 0.08), arb::mechanism_desc("exp2syn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(2, 0.05), arb::mechanism_desc("exp2syn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(1, 0.06), arb::mechanism_desc("exp2syn")));
                break;
            }
            case 9: {
                tgt_types.push_back(std::make_pair(arb::segment_location(1, 0.02), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(1, 0.02), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(2, 0.02), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(2, 0.04), arb::mechanism_desc("expsyn")));
                break;
            }
            case 10: {
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.05), arb::mechanism_desc("exp2syn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(2, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(2, 0.2), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(1, 0.5), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(2, 0.01), arb::mechanism_desc("exp2syn")));
                break;
            }
            case 11: {
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.05), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(2, 0.03), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(1, 0.07), arb::mechanism_desc("expsyn")));
                tgt_types.push_back(std::make_pair(arb::segment_location(0, 0.09), arb::mechanism_desc("expsyn")));
                break;
            }
            default: break;
        }*/

        return dummy_cell(src_types, tgt_types);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    cell_size_type num_sources(cell_gid_type gid) const override {
        std::lock_guard<std::mutex> l(mtx_);
        return database_.num_sources(gid);
        /*switch(gid) {
            case 0: return 6;
            case 1: return 3;
            case 2: return 7;
            case 3: return 2;
            case 4: return 3;
            case 5: return 2;
            case 6: return 2;
            case 7: return 2;
            default: return 0;
        }*/
    }

    cell_size_type num_targets(cell_gid_type gid) const override {
        std::lock_guard<std::mutex> l(mtx_);
        return database_.num_targets(gid);
        /*switch(gid) {
            case 5: return 3;
            case 7: return 5;
            case 8: return 6;
            case 9: return 4;
            case 10: return 5;
            case 11: return 4;
            default: return 0;
        }*/
    }

    // Each cell has one incoming connection, from cell with gid-1.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<arb::cell_connection> conns;

        std::lock_guard<std::mutex> l(mtx_);
        database_.get_connections(gid, conns);

        /*switch(gid) {
            case 5: {
                conns.emplace_back(cell_member_type{0,3}, cell_member_type{5,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{0,3}, cell_member_type{5,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{0,3}, cell_member_type{5,0}, 10.0, 0.5);
                break;
            }
            case 7: {
                conns.emplace_back(cell_member_type{2,2}, cell_member_type{7,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{2,2}, cell_member_type{7,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{2,2}, cell_member_type{7,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{2,2}, cell_member_type{7,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{2,2}, cell_member_type{7,0}, 10.0, 0.5);
                break;
            }
            case 8: {
                conns.emplace_back(cell_member_type{0,0}, cell_member_type{8,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{0,1}, cell_member_type{8,1}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{0,2}, cell_member_type{8,2}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{1,0}, cell_member_type{8,3}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{1,1}, cell_member_type{8,4}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{1,2}, cell_member_type{8,5}, 10.0, 0.5);
                break;
            }
            case 9: {
                conns.emplace_back(cell_member_type{2,0}, cell_member_type{9,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{2,1}, cell_member_type{9,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{3,0}, cell_member_type{9,2}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{3,1}, cell_member_type{9,3}, 10.0, 0.5);
                break;
            }
            case 10: {
                conns.emplace_back(cell_member_type{4,0}, cell_member_type{10,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{4,1}, cell_member_type{10,1}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{4,2}, cell_member_type{10,2}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{5,0}, cell_member_type{10,3}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{5,1}, cell_member_type{10,4}, 10.0, 0.5);
                break;
            }
            case 11: {
                conns.emplace_back(cell_member_type{6,0}, cell_member_type{11,0}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{6,1}, cell_member_type{11,1}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{7,0}, cell_member_type{11,2}, 10.0, 0.5);
                conns.emplace_back(cell_member_type{7,1}, cell_member_type{11,3}, 10.0, 0.5);
                break;
            }
            default: break;
        }*/

        return conns;
    }

    // Return one event generator on gid 0. This generates a single event that will
    // kick start the spiking.
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        std::vector<arb::event_generator> gens;
        if (num_targets(gid) > 0) {
            gens.push_back(arb::explicit_generator(arb::pse_vector{{{gid, 0}, 1.0, 0.05}}));
        }
        return gens;
    }

    cell_size_type num_probes(cell_gid_type gid)  const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        // Get the appropriate kind for measuring voltage.
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma.
        arb::segment_location loc(0, 0.01);

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }

private:
    mutable std::mutex mtx_;
    mutable database database_;
    cell_size_type num_cells_;
};

int main(int argc, char **argv)
{
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

        std::cout << sup::mask_stream(root);

        // Print a banner with information about hardware configuration
        std::cout << "gpu:      " << (has_gpu(context)? "yes": "no") << "\n";
        std::cout << "threads:  " << num_threads(context) << "\n";
        std::cout << "mpi:      " << (has_mpi(context)? "yes": "no") << "\n";
        std::cout << "ranks:    " << num_ranks(context) << "\n" << std::endl;

        arb::profile::meter_manager meters;
        meters.start(context);

        // Create an instance of our recipe.
        using h5_file_handle = std::shared_ptr<h5_file>;

        h5_file_handle nodes = std::make_shared<h5_file>("nodes.h5");
        h5_file_handle edges = std::make_shared<h5_file>("edges.h5");
        csv_file edge_def("edge_types.csv");
        csv_file node_def("node_types.csv");

       /* nodes->print();

        std::cout << std::endl;

        edges->print();

        std::cout << std::endl;*/

        hdf5_record n(nodes);
        hdf5_record e(edges);
        csv_record e_t(edge_def);
        csv_record n_t(node_def);

        sonata_recipe recipe(n, e, n_t, e_t);
        //sonata_recipe recipe;

        auto decomp = arb::partition_load_balance(recipe, context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);

        // Set up the probe that will measure voltage in the cell.

        // The id of the only probe on the cell: the cell_member type points to (cell 0, probe 0)
        auto probe_id = cell_member_type{4, 0};
        // The schedule for sampling is 10 samples every 1 ms.
        auto sched = arb::regular_schedule(0.1);
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
        // Run the simulation for 100 ms, with time steps of 0.025 ms.
        sim.run(100, 0.025);

        meters.checkpoint("model-run", context);

        auto ns = sim.num_spikes();


        // Write spikes to file
        if (root) {
            std::cout << "\n" << ns << " spikes generated \n";
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

        // Write the samples to a json file.
        if (root) write_trace_json(voltage);

        auto report = arb::profile::make_meter_report(meters, context);
        std::cout << report;

        /*std::cout << std::endl << std::endl;
        std::cout << "***************" <<std::endl;
        std::cout << " QUERY RECIPE" <<std::endl;
        std::cout << "***************" <<std::endl;

        std::cout << "Number of cells = " << recipe.num_cells() << std::endl;

        for(unsigned i =0; i < 500; i++) {
            std::cout << "Cell " << i << std::endl;
            auto cell = arb::util::any_cast<arb::cable_cell>(recipe.get_cell_description(i));
            std::cout << "Synapses: " << std::endl;
            for (auto s: cell.synapses()) {
                std::cout << "\t" << s.location.segment << ", " << s.location.position << ": " << s.mechanism.name()
                          << std::endl;
            }
            std::cout << "Detectors: " << std::endl;
            for (auto d: cell.detectors()) {
                std::cout << "\t" << d.location.segment << ", " << d.location.position << ": " << d.threshold << std::endl;
            }
        }
        for(unsigned i =0; i < 500; i++) {
            auto conns = recipe.connections_on(i);
            for (auto j: conns) {
                std::cout << "(" << j.source.gid << ", "<< j.source.index << "); (" << j.dest.gid << ", " << j.dest.index << ")" <<
                ", " << j.delay << ", " << j.weight << std::endl;
            }
        }*/
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in SONATA miniapp: " << e.what() << "\n";
        return 1;
    }
    return 0;
}


arb::cable_cell dummy_cell(
        std::vector<std::pair<arb::segment_location, double>> detectors,
        std::vector<std::pair<arb::segment_location, arb::mechanism_desc>> synapses) {

    arb::cable_cell cell;

    // Add soma.
    auto soma = cell.add_soma(12.6157/2.0); // For area of 500 μm².
    soma->rL = 100;
    soma->add_mechanism("hh");

    auto dend = cell.add_cable(0, arb::section_kind::dendrite, 3.0/2.0, 3.0/2.0, 300); //cable 1
    dend->set_compartments(100);
    dend->add_mechanism("pas");

    auto dend1 = cell.add_cable(1, arb::section_kind::dendrite, 3.0/2.0, 3.0/2.0, 300); //cable 2
    dend1->set_compartments(100);
    dend1->add_mechanism("pas");

    // Add spike threshold detector at the soma.
    for (auto d: detectors) {
        cell.add_detector(d.first, d.second);
    }

    for (auto s: synapses) {
        cell.add_synapse(s.first, s.second);
    }

    return cell;
}


void write_trace_json(const arb::trace_data<double>& trace) {
    std::string path = "./voltages.json";

    nlohmann::json json;
    json["name"] = "ring demo";
    json["units"] = "mV";
    json["cell"] = "0.0";
    json["probe"] = "0";

    auto& jt = json["data"]["time"];
    auto& jy = json["data"]["voltage"];

    for (const auto& sample: trace) {
        jt.push_back(sample.t);
        jy.push_back(sample.v);
    }

    std::ofstream file(path);
    file << std::setw(1) << json << "\n";
}