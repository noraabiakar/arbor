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
#include "csv_lib.hpp"

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
    database(hdf5_record nodes, hdf5_record edges, csv_record node_types, csv_record edge_types):
    nodes_(nodes), edges_(edges), node_types_(node_types), edge_types_(edge_types)
    {
        // Initialize members
        source_.resize(edges_.num_populations());
        target_.resize(edges_.num_populations());
        syn_weight_.resize(edges_.num_populations());
        syn_delay_.resize(edges_.num_populations());
        synapse_.resize(edges_.num_populations());

        // For every edge population, find node source population and node target population
        for (auto edges: edges_.populations()) {
            std::cout << "**************************" << std::endl;

            std::string edge_pop = edges.name();
            unsigned edge_idx = edges_.map()[edge_pop];

            // CSV read
            unsigned csv_src = edge_types_.map()["source_pop_name"];
            unsigned csv_tgt = edge_types_.map()["target_pop_name"];
            unsigned csv_edge = edge_types_.map()["pop_name"];

            /// Not needed yet

            /*for (unsigned i = 0; i < edge_types_.data()[csv_edge].size(); i++) {
                if (edge_types_.data()[csv_edge][i] == edge_pop) {
                    auto source_pop = edge_types_.data()[csv_src][i];
                    src_pop = nodes_.map()[source_pop];
                    auto target_pop = edge_types_.data()[csv_tgt][i];
                    tgt_pop = nodes_.map()[target_pop];
                    break;
                }
            }*/

            // Start navigating the edge population - first, read the source ranges
            /*auto ind_id = edges.find_group("indicies");
            auto s2t_id = edges[ind_id].find_group("source_to_target");

            auto n2r = edges[ind_id][s2t_id].dataset_2d("node_id_to_ranges");
            auto r2e = edges[ind_id][s2t_id].dataset_2d("range_to_edge_id");*/

            // First read edge_group_id and edge_group_index and edge_type

            auto edges_grp_id = edges.dataset_1d("edge_group_id");
            auto edges_grp_idx = edges.dataset_1d("edge_group_index");
            auto edges_type = edges.dataset_1d("edge_type_id");

            for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
                auto loc_grp_id = edges_grp_id.value()[i];

                int source_branch, target_branch;
                double source_pos, target_pos, syn_weight, syn_delay;
                std::string synapse;

                bool found_source_branch = false;
                bool found_source_pos = false;
                bool found_target_branch = false;
                bool found_target_pos = false;
                bool found_syn_weight = false;
                bool found_syn_delay = false;
                bool found_synapse = false;

                // if the edges are in groups, for each edge find the group, if it exists
                if (edges.find_group(std::to_string(loc_grp_id)) != -1 ) {
                    auto group = edges[std::to_string(loc_grp_id)].value();
                    auto loc_grp_idx = edges_grp_idx.value()[i];

                    if (group.find_dataset("afferent_section_id") != -1) {
                        source_branch = group.dataset_i_at("afferent_section_id", loc_grp_idx).value();
                        found_source_branch = true;
                    }
                    if (group.find_dataset("afferent_section_pos") != -1) {
                        source_pos = group.dataset_d_at("afferent_section_pos", loc_grp_idx).value();
                        found_source_pos = true;
                    }
                    if (group.find_dataset("efferent_section_id") != -1) {
                        target_branch = group.dataset_i_at("efferent_section_id", loc_grp_idx).value();
                        found_target_branch = true;
                    }
                    if (group.find_dataset("efferent_section_pos") != -1) {
                        target_pos = group.dataset_d_at("efferent_section_pos", loc_grp_idx).value();
                        found_target_pos = true;
                    }
                    if (group.find_dataset("syn_weight") != -1) {
                        syn_weight = group.dataset_d_at("syn_weight", loc_grp_idx).value();
                        found_syn_weight = true;
                    }
                    if (group.find_dataset("delay") != -1) {
                        syn_delay = group.dataset_d_at("delay", loc_grp_idx).value();
                        found_syn_delay = true;
                    }
                    if (group.find_dataset("model_template") != -1) {
                        synapse = group.dataset_s_at("model_template", loc_grp_idx).value();
                        found_synapse = true;
                    }
                }

                // name and index of edge_type_id
                auto e_type = edges_type.value()[i];
                unsigned type_idx = edge_types_.map()["edge_type_id"];

                // find specific index of edge_type in type_idx
                unsigned loc_type_idx;
                for ( loc_type_idx = 0; loc_type_idx < edge_types_.data()[type_idx].size(); loc_type_idx++) {
                    if (e_type == std::atoi(edge_types_.data()[type_idx][loc_type_idx].c_str())) {
                        break;
                    }
                }

                if (!found_source_branch) {
                    unsigned source_branch_idx = edge_types_.map()["afferent_section_id"];
                    source_branch = std::atoi(edge_types_.data()[source_branch_idx][loc_type_idx].c_str());
                }
                if (!found_source_pos) {
                    unsigned source_pos_idx = edge_types_.map()["afferent_section_pos"];
                    source_pos = std::atof(edge_types_.data()[source_pos_idx][loc_type_idx].c_str());
                }
                if (!found_target_branch) {
                    unsigned target_branch_idx = edge_types_.map()["efferent_section_id"];
                    target_branch = std::atoi(edge_types_.data()[target_branch_idx][loc_type_idx].c_str());
                }
                if (!found_target_pos) {
                    unsigned target_pos_idx = edge_types_.map()["efferent_section_pos"];
                    target_pos = std::atof(edge_types_.data()[target_pos_idx][loc_type_idx].c_str());
                }
                if (!found_syn_weight) {
                    unsigned syn_weight_idx = edge_types_.map()["syn_weight"];
                    syn_weight = std::atof(edge_types_.data()[syn_weight_idx][loc_type_idx].c_str());
                }
                if (!found_syn_delay) {
                    unsigned syn_delay_idx = edge_types_.map()["delay"];
                    syn_delay = std::atof(edge_types_.data()[syn_delay_idx][loc_type_idx].c_str());
                }
                if (!found_synapse) {
                    unsigned synapse_idx = edge_types_.map()["model_template"];
                    synapse = edge_types_.data()[synapse_idx][loc_type_idx];
                }

                source_[edge_idx].push_back({(unsigned)source_branch, source_pos});
                std::cout << "source_branch of " << i  << ": " << source_branch << " " << source_pos  << std::endl;

                target_[edge_idx].push_back({(unsigned)target_branch, target_pos});
                std::cout << "target_branch of " << i  << ": " << target_branch << " " << target_pos  << std::endl;

                syn_weight_[edge_idx].push_back(syn_weight);
                std::cout << "syn_weight of " << i  << ": " << syn_weight << std::endl;

                syn_delay_[edge_idx].push_back(syn_weight);
                std::cout << "syn_delay of " << i  << ": " << syn_delay << std::endl;

                synapse_[edge_idx].push_back(synapse);
                std::cout << "synapse of " << i  << ": " << synapse << std::endl;
            }
        }
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

        /*unsigned i = 0;
        for (; i < gid_partition.size(); i++) {
            if (gid < gid_partition[i]) {
                break;
            }
        }

        unsigned node_pop = i-1;
        unsigned pop_loc = gid - gid_partition[node_pop];

        std::vector<unsigned> target_pops;
        std::vector<unsigned> source_pops;

        // First, find the name of the population we belong to, from node_
        auto node_pop_name = nodes_.populations()[node_pop].name();

        // Second, search for that population name in the corresponding column in edge_types.csv
        unsigned target_pop_idx = edge_types_.map()["target_pop_name"];
        unsigned edge_pop_idx = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[target_pop_idx].size(); i++) {
            if (edge_types_.data()[target_pop_idx][i] == node_pop_name) {
                auto s = edge_types_.data()[edge_pop_idx][i];
                target_pops.push_back(edges_.map()[s]);
            }
        }

        unsigned source_pop_idx = edge_types_.map()["source_pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[source_pop_idx].size(); i++) {
            if (edge_types_.data()[source_pop_idx][i] == node_pop_name) {
                auto s = edge_types_.data()[edge_pop_idx][i];
                source_pops.push_back(edges_.map()[s]);
            }
        }

        std::sort(target_pops.begin(), target_pops.end());
        target_pops.erase(unique( target_pops.begin(), target_pops.end() ), target_pops.end());

        std::sort(source_pops.begin(), source_pops.end());
        source_pops.erase(unique( source_pops.begin(), source_pops.end() ), source_pops.end());


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
        }*/
    }

    void get_connections(cell_gid_type gid) {
        /*unsigned i = 0;
        for (; i < gid_partition.size(); i++) {
            if (gid < gid_partition[i]) {
                break;
            }
        }

        unsigned node_pop = i-1;
        unsigned pop_loc = gid - gid_partition[node_pop];

        std::vector<arb::cell_connection> cons;

        edge_tables[node_pop];*/
    }

    void print_tables() {
        /*int n= 0;
        for(auto e: edge_tables) {
            std::cout << n++ << ":" << std::endl;
            for(auto entry: e) {
                std::cout << "\t{ [" << entry.source.first << ", " << entry.source.second << "], ["
                          << entry.target.first << ", " << entry.target.second << "], " << entry.type << " }" <<std::endl;
            }
        }*/
    }

private:

    hdf5_record nodes_;
    hdf5_record edges_;
    csv_record node_types_;
    csv_record edge_types_;

    // Edge tables
    std::vector<std::vector<arb::segment_location>> source_;
    std::vector<std::vector<arb::segment_location>> target_;
    std::vector<std::vector<double>> syn_weight_;
    std::vector<std::vector<double>> syn_delay_;
    std::vector<std::vector<std::string>> synapse_;




    /*struct conn_info {
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
    std::unordered_map<cell_size_type, conn_info> type_to_conn_info;*/

};

class sonata_recipe: public arb::recipe {
public:
    sonata_recipe(hdf5_record nodes, hdf5_record edges, csv_record node_types, csv_record edge_types):
            database_(nodes, edges, node_types, edge_types),
            num_cells_(database_.num_cells()) {}

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {

        std::vector<std::pair<arb::segment_location,double>> src_types;
        std::vector<std::pair<arb::segment_location,arb::mechanism_desc>> tgt_types;

        database_.get_sources_and_targets(gid, src_types, tgt_types);

        database_.print_tables();

        return dummy_cell(src_types, tgt_types);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override { return 0; }

    // The cell has one target synapse, which will be connected to cell gid-1.
    cell_size_type num_targets(cell_gid_type gid) const override { return 0; }

    // Each cell has one incoming connection, from cell with gid-1.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        return {};
    }

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
                        std::cout << d4->int2_at(0, 0) << ", " << d4->int2_at(0, 1) << std::endl;
                    }
                }
                for (auto d3: g2->datasets_) {
                    std::cout << "\t\t\t\t" << d3->name() << " " << d3->size() << " ";
                    std::cout << d3->int_at(0) << std::endl;
                }
            }
            for (auto d2: g1->datasets_) {
                std::cout << "\t\t\t" << d2->name() << " " << d2->size() << " ";
                std::cout << d2->int_at(0) << std::endl;
            }
        }
        for (auto d1: g0->datasets_) {
            std::cout << "\t\t" << d1->name() << " " << d1->size() << " ";
            std::cout << d1->int_at(0) << std::endl;
        }
    }
    for (auto d0: file->top_group_->datasets_) {
        std::cout << "\t" << d0->name() << " " << d0->size() << " ";
        std::cout << d0->int_at(0) << std::endl;
    }
}

int main(int argc, char **argv)
{
    using h5_file_handle = std::shared_ptr<h5_file>;

    h5_file_handle nodes = std::make_shared<h5_file>("nodes.h5");
    h5_file_handle edges = std::make_shared<h5_file>("edges.h5");
    csv_file edge_def("edge_types.csv");
    csv_file node_def("node_types.csv");

    print(nodes);

    std::cout << std::endl;

    print(edges);

    std::cout << std::endl;

    hdf5_record n(nodes);
    hdf5_record e(edges);
    csv_record e_t(edge_def);
    csv_record n_t(node_def);

    e_t.print();
    n_t.print();

    sonata_recipe recipe(n, e, n_t, e_t);

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

    //recipe.print_tables();


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
