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
#include "hdf5_lib.hpp"

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

class sonata_recipe: public arb::recipe {
public:
    sonata_recipe(unsigned num_cells, std::vector<cell_size_type> partition):
            num_cells_(num_cells),
            gid_partition(partition)
    {}

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {}

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override { return 0; }

    // The cell has one target synapse, which will be connected to cell gid-1.
    cell_size_type num_targets(cell_gid_type gid) const override { return 0; }

    // Each cell has one incoming connection, from cell with gid-1.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override { return {}; }


private:
    cell_size_type num_cells_;
    std::vector<cell_size_type> gid_partition;
};


int main(int argc, char **argv)
{
    h5_file nodes("edges.h5");

    std::cout << nodes.top_group_.name() <<std::endl;
    for (auto &g0: nodes.top_group_.groups_) {
        std::cout << "\t" << g0.name() <<std::endl;
        for (auto &g1: g0.groups_) {
            std::cout << "\t\t" << g1.name() <<std::endl;
            for (auto &g2: g1.groups_) {
                std::cout << "\t\t\t" << g2.name() <<std::endl;
                for (auto &g3: g2.groups_) {
                    std::cout << "\t\t\t\t" << g3.name() <<std::endl;
                    for (auto &g4: g3.groups_) {
                        std::cout << "\t\t\t\t\t" << g4.name() <<std::endl;
                    }
                    for (auto &d4: g3.datasets_) {
                        std::cout << "\t\t\t\t\t" << d4.name() <<std::endl;
                    }
                }
                for (auto &d3: g2.datasets_) {
                    std::cout << "\t\t\t\t" << d3.name() <<std::endl;
                }
            }
            for (auto &d2: g1.datasets_) {
                std::cout << "\t\t\t" << d2.name() <<std::endl;
            }
        }
        for (auto &d1: g0.datasets_) {
            std::cout << "\t\t" << d1.name() <<std::endl;
        }
    }
    for (auto &d0: nodes.top_group_.datasets_) {
        std::cout << "\t" << d0.name() <<std::endl;
    }

    return 0;
}
