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
    h5_file nodes("nodes.h5");

    std::cout << "Populations in file: " << std::endl;
    for (auto i: nodes.get_populations()) {
        std::cout << "\t" << i.get_name() << std::endl;
        std::cout << "\tGroups in populations: " << std::endl;
        for (auto g: i.get_groups()) {
            std::cout << "\t\t" << g.get_name() << std::endl;
        }
        std::cout << "\tDatasets in populations: " << std::endl;
        for (auto d: i.get_datasets()) {
            std::cout << "\t\t" << d.get_name() << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "total number of cells in file: "<< nodes.get_num_elements() << std::endl;
    std::cout << "partition of cells: " << std::endl;
    for(auto e: nodes.get_partition()) {
        std::cout << e << " ";
    }
    std::cout << std::endl;

    h5_file edges("edges.h5");

    std::cout << "Populations in file: " << std::endl;
    for (auto i: edges.get_populations()) {
        std::cout << "\t" << i.get_name() << std::endl;
        std::cout << "\tGroups in populations: " << std::endl;
        for (auto g: i.get_groups()) {
            std::cout << "\t\t" << g.get_name() << std::endl;
        }
        std::cout << "\tDatasets in populations: " << std::endl;
        for (auto d: i.get_datasets()) {
            std::cout << "\t\t" << d.get_name() << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "total number of edges in file: "<< edges.get_num_elements() << std::endl;
    std::cout << "partition of edges: " << std::endl;
    for(auto e: edges.get_partition()) {
        std::cout << e << " ";
    }
    std::cout << std::endl;

    std::cout << std::endl << "Testing: open a dataet" << std::endl;
    for(auto p: nodes.get_populations()) {
        for (auto d: p.get_datasets()) {
            d.get_value_at(0);
        }
    }

    return 0;
}
