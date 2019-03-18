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

struct cell_loc{
    cell_gid_type gid;
    arb::segment_location loc;

    bool operator==(const cell_loc &rhs) const { return (gid == rhs.gid) && (loc == rhs.loc); }
};

namespace std {
    template<> struct hash<cell_loc> {
        std::size_t operator()(const cell_loc &f) const {
            return ((hash<cell_gid_type>()(f.gid)
                  ^ (hash<cell_lid_type>()(f.loc.segment) << 1)) >> 1)
                  ^ (hash<double>()(f.loc.position) << 1);
        }
    };
}

class database {
public:
    database(hdf5_record nodes, hdf5_record edges, csv_record node_types, csv_record edge_types):
    nodes_(nodes), edges_(edges), node_types_(node_types), edge_types_(edge_types) {}

    cell_size_type num_cells() {
        return nodes_.num_elements();
    }

    cell_size_type num_edges() {
        return edges_.num_elements();
    }

    void get_connections(cell_gid_type gid,
            std::vector<arb::cell_connection>& conns) {

        unsigned i = 0;
        for (; i < nodes_.partitions().size(); i++) {
            if (gid < nodes_.partitions()[i]) {
                break;
            }
        }

        unsigned node_pop = i-1;
        unsigned pop_id = gid - nodes_.partitions()[node_pop];
        std::string pop_name = nodes_.populations()[node_pop].name();

        std::vector<unsigned> target_edge_pops;

        // CSV read
        unsigned src_vec_id = edge_types_.map()["source_pop_name"];
        unsigned tgt_vec_id = edge_types_.map()["target_pop_name"];
        unsigned edge_vec_id = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[src_vec_id].size(); i++) {
            if (edge_types_.data()[tgt_vec_id][i] == pop_name) {
                auto edge_pop = edge_types_.data()[edge_vec_id][i];
                target_edge_pops.push_back(edges_.map()[edge_pop]);
            }
        }

        std::sort(source_edge_pops.begin(), source_edge_pops.end());
        std::sort(target_edge_pops.begin(), target_edge_pops.end());

        source_edge_pops.erase(std::unique(source_edge_pops.begin(), source_edge_pops.end()), source_edge_pops.end() );
        target_edge_pops.erase(std::unique(target_edge_pops.begin(), target_edge_pops.end()), target_edge_pops.end() );
    }

    void get_sources_and_targets(cell_gid_type gid,
            std::vector<std::pair<arb::segment_location, double>>& src,
            std::vector<std::pair<arb::segment_location, arb::mechanism_desc>>& tgt) {

        unsigned i = 0;
        for (; i < nodes_.partitions().size(); i++) {
            if (gid < nodes_.partitions()[i]) {
                break;
            }
        }

        unsigned node_pop = i-1;
        unsigned pop_id = gid - nodes_.partitions()[node_pop];
        std::string pop_name = nodes_.populations()[node_pop].name();

        std::vector<unsigned> source_edge_pops;
        std::vector<unsigned> target_edge_pops;

        // CSV read
        unsigned src_vec_id = edge_types_.map()["source_pop_name"];
        unsigned tgt_vec_id = edge_types_.map()["target_pop_name"];
        unsigned edge_vec_id = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[src_vec_id].size(); i++) {
            if (edge_types_.data()[src_vec_id][i] == pop_name) {
                auto edge_pop = edge_types_.data()[edge_vec_id][i];
                source_edge_pops.push_back(edges_.map()[edge_pop]);
            }
            if (edge_types_.data()[tgt_vec_id][i] == pop_name) {
                auto edge_pop = edge_types_.data()[edge_vec_id][i];
                target_edge_pops.push_back(edges_.map()[edge_pop]);
            }
        }

        std::sort(source_edge_pops.begin(), source_edge_pops.end());
        std::sort(target_edge_pops.begin(), target_edge_pops.end());

        source_edge_pops.erase(std::unique(source_edge_pops.begin(), source_edge_pops.end()), source_edge_pops.end() );
        target_edge_pops.erase(std::unique(target_edge_pops.begin(), target_edge_pops.end()), target_edge_pops.end() );

        for (auto i: source_edge_pops) {
            std::vector<std::pair<int, int>> source_edge_ranges;

            auto ind_id = edges_[i].find_group("indicies");
            auto s2t_id = edges_[i][ind_id].find_group("source_to_target");
            auto n2r_range = edges_[i][ind_id][s2t_id].dataset_2i_at("node_id_to_ranges", pop_id);
            for (auto j = n2r_range->first; j< n2r_range->second; j++) {
                auto r2e = edges_[i][ind_id][s2t_id].dataset_2i_at("range_to_edge_id", j);
                source_edge_ranges.push_back(r2e.value());
                auto id = edges_[i].dataset_int_range("edge_id", r2e->first, r2e->second);
            }
            for (auto r: source_edge_ranges) {
                get_sources_range(gid, i, r, src);
            }
        }

        for (auto i: target_edge_pops) {
            std::vector<std::pair<int, int>> target_edge_ranges;

            auto ind_id = edges_[i].find_group("indicies");
            auto t2s_id = edges_[i][ind_id].find_group("target_to_source");
            auto n2r = edges_[i][ind_id][t2s_id].dataset_2i_at("node_id_to_ranges", pop_id);
            for (auto j = n2r->first; j< n2r->second; j++) {
                auto r2e = edges_[i][ind_id][t2s_id].dataset_2i_at("range_to_edge_id", j);
                target_edge_ranges.push_back(r2e.value());
            }
            // for all edges in this edge population
            for (auto r: target_edge_ranges) {
                get_targets_range(gid, i, r, tgt);
            }
        }
    }

    void print_tables() {}

private:

    void get_targets_range(
            cell_gid_type gid,
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range,
            std::vector<std::pair<arb::segment_location, arb::mechanism_desc>>& tgt)
    {
        // First read edge_group_id and edge_group_index and edge_type
        auto edges_grp_id = edges_[edge_pop_id].dataset_int_range("edge_group_id", edge_range.first, edge_range.second);
        auto edges_grp_idx = edges_[edge_pop_id].dataset_int_range("edge_group_index", edge_range.first, edge_range.second);
        auto edges_type = edges_[edge_pop_id].dataset_int_range("edge_type_id", edge_range.first, edge_range.second);

        std::vector<std::pair<arb::segment_location, arb::mechanism_desc>> temp;

        for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
            auto loc_grp_id = edges_grp_id.value()[i];

            int target_branch;
            double target_pos;
            std::string synapse;

            bool found_target_branch = false;
            bool found_target_pos = false;
            bool found_synapse = false;

            // if the edges are in groups, for each edge find the group, if it exists
            if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
                std::cout << "found_group!\n";
                auto group = edges_[edge_pop_id][std::to_string(loc_grp_id)].value();
                auto loc_grp_idx = edges_grp_idx.value()[i];

                if (group.find_dataset("efferent_section_id") != -1) {
                    target_branch = group.dataset_i_at("efferent_section_id", loc_grp_idx).value();
                    found_target_branch = true;
                }
                if (group.find_dataset("efferent_section_pos") != -1) {
                    target_pos = group.dataset_d_at("efferent_section_pos", loc_grp_idx).value();
                    found_target_pos = true;
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
            for (loc_type_idx = 0; loc_type_idx < edge_types_.data()[type_idx].size(); loc_type_idx++) {
                if (e_type == std::atoi(edge_types_.data()[type_idx][loc_type_idx].c_str())) {
                    break;
                }
            }

            if (!found_target_branch) {
                unsigned target_branch_idx = edge_types_.map()["efferent_section_id"];
                target_branch = std::atoi(edge_types_.data()[target_branch_idx][loc_type_idx].c_str());
            }
            if (!found_target_pos) {
                unsigned target_pos_idx = edge_types_.map()["efferent_section_pos"];
                target_pos = std::atof(edge_types_.data()[target_pos_idx][loc_type_idx].c_str());
            }

            if (!found_synapse) {
                unsigned synapse_idx = edge_types_.map()["model_template"];
                synapse = edge_types_.data()[synapse_idx][loc_type_idx];
            }

            temp.emplace_back(arb::segment_location((unsigned)target_branch, target_pos), arb::mechanism_desc(synapse));
        }

        std::sort(temp.begin(), temp.end(), [](const auto &a, const auto& b) -> bool
        {
            return std::tie(a.first.segment, a.first.position) > std::tie(b.first.segment, b.first.position);
        });

        for (auto t : target_lists_[gid]) {
            auto t_pos = std::lower_bound(temp.begin(), temp.end(), t,
                    [](const std::pair<arb::segment_location, arb::mechanism_desc>& rhs, const arb::segment_location& lhs) -> bool
                    { return std::tie(lhs.segment, lhs.position) > std::tie(rhs.first.segment, rhs.first.position); });
            if (t_pos != temp.end()) {
                if ((*t_pos).first == t) {
                    tgt.push_back(*t_pos);
                    temp.erase(t_pos);
                }
            }
        }

        for (auto t : temp) {
            tgt.push_back(t);
            target_lists_[gid].push_back(t.first);
        }
    }

    void get_sources_range(
            cell_gid_type gid,
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range,
            std::vector<std::pair<arb::segment_location, double>>& src)
    {
        // First read edge_group_id and edge_group_index and edge_type
        auto edges_grp_id = edges_[edge_pop_id].dataset_int_range("edge_group_id", edge_range.first, edge_range.second);
        auto edges_grp_idx = edges_[edge_pop_id].dataset_int_range("edge_group_index", edge_range.first, edge_range.second);
        auto edges_type = edges_[edge_pop_id].dataset_int_range("edge_type_id", edge_range.first, edge_range.second);

        std::vector<std::pair<arb::segment_location, double>> temp;

        for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
            auto loc_grp_id = edges_grp_id.value()[i];

            int source_branch;
            double source_pos;

            bool found_source_branch = false;
            bool found_source_pos = false;

            // if the edges are in groups, for each edge find the group, if it exists
            if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
                std::cout << "found_group!\n";
                auto group = edges_[edge_pop_id][std::to_string(loc_grp_id)].value();
                auto loc_grp_idx = edges_grp_idx.value()[i];

                if (group.find_dataset("afferent_section_id") != -1) {
                    source_branch = group.dataset_i_at("afferent_section_id", loc_grp_idx).value();
                    found_source_branch = true;
                }
                if (group.find_dataset("afferent_section_pos") != -1) {
                    source_pos = group.dataset_d_at("afferent_section_pos", loc_grp_idx).value();
                    found_source_pos = true;
                }
            }

            // name and index of edge_type_id
            auto e_type = edges_type.value()[i];
            unsigned type_idx = edge_types_.map()["edge_type_id"];

            // find specific index of edge_type in type_idx
            unsigned loc_type_idx;
            for (loc_type_idx = 0; loc_type_idx < edge_types_.data()[type_idx].size(); loc_type_idx++) {
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

            temp.emplace_back(arb::segment_location((unsigned)source_branch, source_pos), 10);
        }

        std::sort(temp.begin(), temp.end(), [](const auto &a, const auto& b) -> bool
        {
            return std::tie(a.first.segment, a.first.position) > std::tie(b.first.segment, b.first.position);
        });

        for (auto t : source_lists_[gid]) {
            auto t_pos = std::lower_bound(temp.begin(), temp.end(), t,
                                          [](const std::pair<arb::segment_location, double>& rhs, const arb::segment_location& lhs) -> bool
                                          { return std::tie(lhs.segment, lhs.position) > std::tie(rhs.first.segment, rhs.first.position); });
            if (t_pos != temp.end()) {
                if ((*t_pos).first == t) {
                    src.push_back(*t_pos);
                    temp.erase(t_pos);
                }
            }
        }

        for (auto t : temp) {
            src.push_back(t);
            source_lists_[gid].push_back(t.first);
        }
    }

    hdf5_record nodes_;
    hdf5_record edges_;
    csv_record node_types_;
    csv_record edge_types_;

    std::unordered_map<cell_gid_type , std::vector<arb::segment_location>> source_lists_;
    std::unordered_map<cell_gid_type , std::vector<arb::segment_location>> target_lists_;
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
                        std::cout << d4->int2_at(0).first << ", " << d4->int2_at(0).second << std::endl;
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
