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


struct edge_info{
    std::pair<unsigned, unsigned> source;
    std::pair<unsigned, unsigned> target;
    unsigned type;
};

class database {
public:
    database(record nodes, record edges): nodes_(nodes), edges_(edges) {
        edge_tables.reserve(edges.populations().size());
        for (unsigned i = 0; i < edges.populations().size(); i++) {
            edge_tables.emplace_back(edges.partitions()[i]);
        }

        unsigned sum = 0;
        gid_partition.push_back(sum);
        for(auto n: nodes_.partitions()) {
            gid_partition.push_back(sum+=n);
        }

        arb::mechanism_desc expsyn("expsyn");
        arb::mechanism_desc exp2syn("exp2syn");
        type_to_conn_info[100] = conn_info(arb::segment_location{0, 0.1}, arb::segment_location{1, 0.2}, 0.5, expsyn);
        type_to_conn_info[101] = conn_info(arb::segment_location{1, 0.1}, arb::segment_location{0, 0.1}, 0.1, expsyn);
        type_to_conn_info[102] = conn_info(arb::segment_location{0, 0.5}, arb::segment_location{0, 0.5}, 0.3, exp2syn);
    }

    cell_size_type num_cells() {
        return nodes_.num_elements();
    }

    cell_size_type num_edges() {
        return edges_.num_elements();
    }

    void get_sources_and_targets(cell_gid_type gid,
            std::vector<std::pair<arb::segment_location, double>>& src,
            std::vector<std::pair<arb::segment_location, arb::mechanism_desc>>& tgt) {
        unsigned i = 0;
        for (; i < gid_partition.size(); i++) {
            if (gid < gid_partition[i]) {
                break;
            }
        }

        unsigned node_pop = i-1;
        unsigned pop_loc = gid - gid_partition[node_pop];

        // Replace with CSV read
        std::vector<unsigned> target_pops;
        if (gid < 8){
            target_pops.push_back(1);
        } else {
            target_pops.push_back(0);
        }

        for (auto tp: target_pops) {
            auto indices_id = edges_[tp].find_group("indicies");
            auto target_to_source_id = edges_[tp][indices_id].find_group("target_to_source");

            std::pair<unsigned, unsigned> range;
            range.first = edges_[tp][indices_id][target_to_source_id].dataset_at("node_id_to_ranges", pop_loc, 0).value();
            range.second = edges_[tp][indices_id][target_to_source_id].dataset_at("node_id_to_ranges", pop_loc, 1).value();

            for (unsigned i = range.first; i < range.second; i++) {
                std::pair<unsigned, unsigned> edge_ids;
                edge_ids.first = edges_[tp][indices_id][target_to_source_id].dataset_at("range_to_edge_id", i, 0).value();
                edge_ids.second = edges_[tp][indices_id][target_to_source_id].dataset_at("range_to_edge_id", i, 1).value();

                unsigned loc_idx = 0;
                for (unsigned j = edge_ids.first; j < edge_ids.second; j++, loc_idx++) {
                    auto type = edges_[tp].dataset_at("edge_type_id", j).value();
                    edge_tables[tp][j].target = {gid, loc_idx};
                    edge_tables[tp][j].type = type;
                    auto conn_info = type_to_conn_info[type];
                    tgt.push_back(std::make_pair(conn_info.tgt_, conn_info.syn_));
                }
            }
        }

        // Replace with CSV read
        std::vector<unsigned> source_pops;
        if (gid < 8) {
            source_pops.push_back(0);
            source_pops.push_back(1);
        }

        for (auto sp: source_pops) {
            auto indices_id = edges_[sp].find_group("indicies");
            auto source_to_target_id = edges_[sp][indices_id].find_group("source_to_target");

            std::pair<unsigned, unsigned> range;
            range.first = edges_[sp][indices_id][source_to_target_id].dataset_at("node_id_to_ranges", pop_loc, 0).value();
            range.second = edges_[sp][indices_id][source_to_target_id].dataset_at("node_id_to_ranges", pop_loc, 1).value();

            for (unsigned i = range.first; i < range.second; i++) {
                std::pair<unsigned, unsigned> edge_ids;
                edge_ids.first = edges_[sp][indices_id][source_to_target_id].dataset_at("range_to_edge_id", i, 0).value();
                edge_ids.second = edges_[sp][indices_id][source_to_target_id].dataset_at("range_to_edge_id", i, 1).value();

                unsigned loc_idx = 0;
                for (unsigned j = edge_ids.first; j < edge_ids.second; j++, loc_idx++) {
                    auto type = edges_[sp].dataset_at("edge_type_id", j).value();
                    edge_tables[sp][j].source = {gid, loc_idx};
                    edge_tables[sp][j].type = type;
                    auto conn_info = type_to_conn_info[type];
                    src.push_back(std::make_pair(conn_info.src_, conn_info.weight_));
                }
            }
        }
    }

    void print_tables() {
        int n= 0;
        for(auto e: edge_tables) {
            std::cout << n++ << ":" << std::endl;
            for(auto entry: e) {
                std::cout << "\t{ [" << entry.source.first << ", " << entry.source.second << "], ["
                          << entry.target.first << ", " << entry.target.second << "], " << entry.type << " }" <<std::endl;
            }
        }
    }

private:

    record nodes_;
    record edges_;
    std::vector<std::vector<edge_info>> edge_tables;
    std::vector<cell_size_type> gid_partition;

    struct conn_info {
        conn_info() : src_(0,0), tgt_(0,0), weight_(0), syn_("expsyn") {};
        conn_info (
            arb::segment_location src,
            arb::segment_location tgt,
            double weight,
            arb::mechanism_desc syn):
            src_(src), tgt_(tgt), weight_(weight), syn_(syn) {};

        arb::segment_location src_;
        arb::segment_location tgt_;
        double weight_;
        arb::mechanism_desc syn_;
    };
    std::unordered_map<cell_size_type, conn_info> type_to_conn_info;

};

class sonata_recipe: public arb::recipe {
public:
    sonata_recipe(record nodes, record edges):
            database_(nodes, edges),
            num_cells_(database_.num_cells()) {}

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {

        std::vector<std::pair<arb::segment_location,double>> src_types;
        std::vector<std::pair<arb::segment_location,arb::mechanism_desc>> tgt_types;

        database_.get_sources_and_targets(gid, src_types, tgt_types);

        return dummy_cell(src_types, tgt_types);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override { return 0; }

    // The cell has one target synapse, which will be connected to cell gid-1.
    cell_size_type num_targets(cell_gid_type gid) const override { return 0; }

    // Each cell has one incoming connection, from cell with gid-1.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override { return {}; }

    void print_tables() {
        database_.print_tables();
    }


private:
    mutable database database_;
    cell_size_type num_cells_;
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
                        std::cout << "\t\t\t\t\t" << d4->name() << " " << d4->size() << " ";
                        std::cout << d4->at(0, 0) << ", " << d4->at(0, 1) << std::endl;
                    }
                }
                for (auto d3: g2->datasets_) {
                    std::cout << "\t\t\t\t" << d3->name() << " " << d3->size() << " ";
                    std::cout << d3->at(0) << std::endl;
                }
            }
            for (auto d2: g1->datasets_) {
                std::cout << "\t\t\t" << d2->name() << " " << d2->size() << " ";
                std::cout << d2->at(0) << std::endl;
            }
        }
        for (auto d1: g0->datasets_) {
            std::cout << "\t\t" << d1->name() << " " << d1->size() << " ";
            std::cout << d1->at(0) << std::endl;
        }
    }
    for (auto d0: file->top_group_->datasets_) {
        std::cout << "\t" << d0->name() << " " << d0->size() << " ";
        std::cout << d0->at(0) << std::endl;
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
        std::cout << "Cell " << i << std::endl;
        auto cell = arb::util::any_cast<arb::cable_cell>(recipe.get_cell_description(i));
        std::cout << "Synapses: " << std::endl;
        for (auto s: cell.synapses()) {
            std::cout << "\t" << s.location.segment << ", " << s.location.position << ": " << s.mechanism.name() <<std::endl;
        }
        std::cout << "Detectors: " << std::endl;
        for (auto d: cell.detectors()) {
            std::cout << "\t" << d.location.segment << ", " << d.location.position << ": " << d.threshold <<std::endl;
        }
    }

    recipe.print_tables();

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
