/*
 * Miniapp that uses the artificial benchmark cell type to test
 * the simulator infrastructure.
 */
#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/profile/meter_manager.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/version.hpp>


#include <aux/ioutil.hpp>
#include <aux/json_meter.hpp>
#ifdef ARB_MPI_ENABLED
#include <aux/with_mpi.hpp>
#endif

#include "parameters.hpp"
#include "recipe.hpp"

namespace profile = arb::profile;

int main(int argc, char** argv) {
    bool is_root = true;

    try {
        bench_params params = read_options(argc, argv);
        auto resources = arb::proc_allocation();
        auto context = arb::make_context(resources);
        if (params.dry_run) {
            context = arb::make_context(resources, arb::dry_run_info(params.num_tiles, params.num_cells));
        }
#ifdef ARB_MPI_ENABLED
            else {
            context = arb::make_context(resources, MPI_COMM_WORLD);
            {
                int rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                is_root = rank==0;
            }
        }
#endif
        arb_assert(arb::num_ranks(context)==params.num_tiles);

#ifdef ARB_PROFILE_ENABLED
        profile::profiler_initialize(context);
#endif

        std::cout << aux::mask_stream(is_root);

        std::cout << params << "\n";

        profile::meter_manager meters;
        meters.start(context);

        // Create an instance of our symmetric recipe.
        auto tile = std::make_unique<bench_tile>(params);
        arb::symmetric_recipe recipe(std::move(tile));

        meters.checkpoint("recipe-build", context);

        // Make the domain decomposition for the model
        auto decomp = arb::partition_load_balance(recipe, context);
        meters.checkpoint("domain-decomp", context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);
        meters.checkpoint("model-build", context);

        // Set up recording of spikes to a vector on the root process.
        std::vector<arb::spike> recorded_spikes;
        if (is_root) {
            sim.set_global_spike_callback(
                    [&recorded_spikes](const std::vector<arb::spike>& spikes) {
                        recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                    });
        }

        // Run the simulation for 100 ms, with time steps of 0.01 ms.
        sim.run(params.duration, 0.01);
        meters.checkpoint("model-run", context);

        // write meters
        auto report = profile::make_meter_report(meters, context);
        std::cout << report << "\n";

        if (is_root) {
            std::ofstream meter_file;
            meter_file.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            meter_file.open("meters.json");
            meter_file << std::setw(1) << aux::to_json(report) << "\n";
            meter_file.close();

            std::ofstream spike_file;
            spike_file.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            spike_file.open("spikes.gdf");
            char linebuf[45];
            for (auto spike: recorded_spikes) {
                auto n = std::snprintf(
                        linebuf, sizeof(linebuf), "%u %.4f\n",
                        unsigned{spike.source.gid}, float(spike.time));
                spike_file.write(linebuf, n);
            }
            spike_file.close();
        }

        // output profile and diagnostic feedback
        auto summary = profile::profiler_summary();
        std::cout << summary << "\n";

        std::cout << "there were " << sim.num_spikes() << " spikes\n";
    }
    catch (std::exception& e) {
        std::cerr << "exception caught running benchmark miniapp:\n" << e.what() << std::endl;
    }
}
