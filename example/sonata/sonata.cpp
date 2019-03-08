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

// Generate a cell.
arb::cable_cell dummy_cell(
        std::vector<std::pair<arb::segment_location, double>>,
        std::vector<std::pair<arb::segment_location, arb::mechanism_desc>>);

class sonata_recipe: public arb::recipe {
public:
    sonata_recipe(record nodes, record edges):
            nodes_(nodes),
            edges_(edges),
            num_cells_(nodes_.num_elements())
    {
        edge_tables.reserve(edges.populations().size());
        for (unsigned i = 0; i < edges.populations().size(); i++) {
            edge_tables.emplace_back(edges.partitions()[i]);
        }

        unsigned sum = 0;
        gid_partition.push_back(sum);
        for(auto n: nodes_.partitions()) {
            gid_partition.push_back(sum+=n);
        }
    }

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        unsigned i = 0;
        for (; i < gid_partition.size(); i++) {
            if (gid < gid_partition[i]) {
                break;
            }
        }

        unsigned node_pop = i-1;
        unsigned pop_loc = gid - gid_partition[node_pop];
        std::cout << node_pop << " " << pop_loc << std::endl;

        // Get edge populations where target_pop==nodes_[node_pop]
        // Let's say that's edges_[0]
        
        return dummy_cell({}, {});
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override { return 0; }

    // The cell has one target synapse, which will be connected to cell gid-1.
    cell_size_type num_targets(cell_gid_type gid) const override { return 0; }

    // Each cell has one incoming connection, from cell with gid-1.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override { return {}; }


private:
    struct edge_info{
        std::pair<unsigned, unsigned> source;
        std::pair<unsigned, unsigned> target;
        unsigned type;
    };

    record nodes_;
    record edges_;
    cell_size_type num_cells_;
    std::vector<cell_size_type> gid_partition;

    std::vector<std::vector<edge_info>> edge_tables;
};

void print(const std::shared_ptr<h5_file>& file) {
    std::cout << file->top_group_->name() << std::endl;
    for (auto g0: file->top_group_->groups_) {
        std::cout << "\t" << g0->name() << std::endl;
        for (auto g1: g0->groups_) {
            std::cout << "\t\t" << g1->name() << std::endl;
            for (auto g2: g1->groups_) {
                std::cout << "\t\t\t" << g2->name() << std::endl;
                for (auto g3: g2->groups_) {
                    std::cout << "\t\t\t\t" << g3->name() << std::endl;
                    for (auto g4: g3->groups_) {
                        std::cout << "\t\t\t\t\t" << g4->name() << std::endl;
                    }
                    for (auto d4: g3->datasets_) {
                        std::cout << "\t\t\t\t\t" << d4->name() << " " << d4->size() << std::endl;
                    }
                }
                for (auto d3: g2->datasets_) {
                    std::cout << "\t\t\t\t" << d3->name() << " " << d3->size() << std::endl;
                }
            }
            for (auto d2: g1->datasets_) {
                std::cout << "\t\t\t" << d2->name() << " " << d2->size() << std::endl;
            }
        }
        for (auto d1: g0->datasets_) {
            std::cout << "\t\t" << d1->name() << " " << d1->size() << std::endl;
        }
    }
    for (auto d0: file->top_group_->datasets_) {
        std::cout << "\t" << d0->name() << " " << d0->size() << std::endl;
    }
}

int main(int argc, char **argv)
{
    using h5_file_handle = std::shared_ptr<h5_file>;

    h5_file_handle nodes = std::make_shared<h5_file>("nodes.h5");
    h5_file_handle edges = std::make_shared<h5_file>("edges.h5");

    print(nodes);

    std::cout << std::endl;

    print(edges);

    std::cout << std::endl;

    record n(nodes);
    std::cout << "Nodes\n" << n.partitions().size() << std::endl << "{ ";
    for(auto p: n.partitions()) {
        std::cout << p << " ";
    }
    std::cout << " }" << std::endl;

    record e(edges);
    std::cout << "Edges\n" << e.partitions().size() << std::endl << "{ ";
    for(auto p: e.partitions()) {
        std::cout << p << " ";
    }
    std::cout << " }" << std::endl;


    sonata_recipe recipe(n, e);

    std::cout << std::endl << std::endl;
    std::cout << "***************" <<std::endl;
    std::cout << " QUERY RECIPE" <<std::endl;
    std::cout << "***************" <<std::endl;

    std::cout << "Number of cells = " << recipe.num_cells() << std::endl;

    for(unsigned i =0; i < 12; i++) {
        recipe.get_cell_description(i);
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

    // Add spike threshold detector at the soma.
    for (auto d: detectors) {
        cell.add_detector(d.first, d.second);
    }

    for (auto s: synapses) {
        cell.add_synapse(s.first, s.second);
    }

    return cell;
}
