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

    //sonata_recipe(): num_cells_(500) {}

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {

        std::vector<std::pair<arb::segment_location,double>> src_types;
        std::vector<std::pair<arb::segment_location,arb::mechanism_desc>> tgt_types;

        std::lock_guard<std::mutex> l(mtx_);
        database_.get_sources_and_targets(gid, src_types, tgt_types);
        //src_types.emplace_back(arb::segment_location(0,0.9), 10);
        //tgt_types.emplace_back(arb::segment_location(0,0.5), arb::mechanism_desc("expsyn"));

        return dummy_cell(src_types, tgt_types);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    cell_size_type num_sources(cell_gid_type gid) const override {
        std::lock_guard<std::mutex> l(mtx_);
        return database_.num_sources(gid);
        //return 25;
    }

    cell_size_type num_targets(cell_gid_type gid) const override {
        std::lock_guard<std::mutex> l(mtx_);
        return database_.num_targets(gid);
        //return 25;
    }

    // Each cell has one incoming connection, from cell with gid-1.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<arb::cell_connection> conns;

        /*if (gid < 400) {
            for (unsigned i = 0; i < 20; i++) {
                conns.push_back(arb::cell_connection({(gid+i+1)%400, 0}, {gid, 0}, 0.01, 0.1));
            }
            for (unsigned i = 0; i < 5; i++) {
                conns.push_back(arb::cell_connection({(gid+i)%100 + 400, 0}, {gid, 0}, -0.01, 0.1));
            }
        }
        else {
            for (unsigned i = 0; i < 5; i++) {
                conns.push_back(arb::cell_connection({((gid - 400)+i+1)%100 + 400, 0}, {gid, 0}, -0.01, 0.1));
            }
            for (unsigned i = 0; i < 5; i++) {
                for (unsigned j = 0; j < 4; j++) {
                    conns.push_back(arb::cell_connection({((gid - 400)*4 + j)%400, 0}, {gid, 0}, 0.01, 0.1));
                }
            }
        }*/
        std::lock_guard<std::mutex> l(mtx_);
        database_.get_connections(gid, conns);
        return conns;
    }

    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        std::vector<arb::event_generator> gens;
        if (gid == 1) {
            float t = ((float)gid/num_cells())*3.0;
            gens.push_back(arb::explicit_generator(arb::pse_vector{{{gid, 0}, t, 0.009}}));
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

        hdf5_record n(nodes);
        n.verify_nodes();

        hdf5_record e(edges);
        e.verify_edges();

        csv_record e_t(edge_def);
        csv_record n_t(node_def);

        sonata_recipe recipe(n, e, n_t, e_t);
        //sonata_recipe recipe;

        auto decomp = arb::partition_load_balance(recipe, context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);

        // Set up the probe that will measure voltage in the cell.

        // The id of the only probe on the cell: the cell_member type points to (cell 0, probe 0)
        auto probe_id = cell_member_type{421, 0};
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

        auto summary = arb::profile::profiler_summary();
        std::cout << summary << "\n";


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
            std::cout << "Cell " << i << std::endl;
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