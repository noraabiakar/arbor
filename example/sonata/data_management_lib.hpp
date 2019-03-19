#include <arbor/common_types.hpp>
#include <arbor/util/optional.hpp>
#include <string.h>
#include <stdio.h>
#include <hdf5.h>
#include <assert.h>

#include "hdf5_lib.hpp"
#include "csv_lib.hpp"

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

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

    void get_connections(cell_gid_type gid, std::vector<arb::cell_connection>& conns);

    void get_sources_and_targets(cell_gid_type gid,
                                 std::vector<std::pair<arb::segment_location, double>>& src,
                                 std::vector<std::pair<arb::segment_location, arb::mechanism_desc>>& tgt);


private:

    std::vector<arb::segment_location> get_afferent_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range);

    std::vector<arb::segment_location> get_efferent_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range);

    std::vector<arb::mechanism_desc> get_synapses_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range);

    std::vector<double> get_weight_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range);

    std::vector<double> get_delay_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range);

    hdf5_record nodes_;
    hdf5_record edges_;
    csv_record node_types_;
    csv_record edge_types_;

    std::unordered_map<cell_gid_type , std::vector<arb::segment_location>> source_lists_;
    std::unordered_map<cell_gid_type , std::vector<arb::segment_location>> target_lists_;
};

void database::get_connections(cell_gid_type gid,
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

    std::unordered_map <unsigned, unsigned> edge_to_source;
    // CSV read
    unsigned src_vec_id = edge_types_.map()["source_pop_name"];
    unsigned tgt_vec_id = edge_types_.map()["target_pop_name"];
    unsigned edge_vec_id = edge_types_.map()["pop_name"];

    for (unsigned i = 0; i < edge_types_.data()[src_vec_id].size(); i++) {
        if (edge_types_.data()[tgt_vec_id][i] == pop_name) {
            auto edge_pop = edge_types_.data()[edge_vec_id][i];
            auto source_pop = edge_types_.data()[src_vec_id][i];
            edge_to_source[edges_.map()[edge_pop]] = edges_.map()[source_pop];
        }
    }

    for (auto i: edge_to_source) {
        auto edge_pop = i.first;
        auto source_pop = i.second;

        std::vector<std::pair<int, int>> source_edge_ranges;

        auto ind_id = edges_[edge_pop].find_group("indicies");
        auto s2t_id = edges_[edge_pop][ind_id].find_group("target_to_source");
        auto n2r_range = edges_[edge_pop][ind_id][s2t_id].dataset_2i_at("node_id_to_ranges", pop_id);
        for (auto j = n2r_range->first; j< n2r_range->second; j++) {
            auto r2e = edges_[edge_pop][ind_id][s2t_id].dataset_2i_at("range_to_edge_id", j);
            source_edge_ranges.push_back(r2e.value());
        }

        for (unsigned i = 0; i< source_edge_ranges.size(); i++) {
            auto targets_t = get_efferent_range(edge_pop, source_edge_ranges[i]);
            auto sources_t = get_afferent_range(edge_pop, source_edge_ranges[i]);
            auto weights = get_weight_range(edge_pop, source_edge_ranges[i]);
            auto delays = get_delay_range(edge_pop, source_edge_ranges[i]);

            auto src_id = edges_[edge_pop].dataset_int_range("source_node_id",
                                                             source_edge_ranges[i].first, source_edge_ranges[i].second);

            if (targets_t.size()!= sources_t.size()) {
                std::cout << "ERROR!" <<std::endl;
            }

            std::vector<cell_member_type> sources, targets;

            for(unsigned i = 0; i < targets_t.size(); i++) {
                // if we can't find the target in target_lists, its cell hasn't been constructed yet
                bool found = false;
                unsigned j = 0;
                for (; j < target_lists_[gid].size(); j++) {
                    if (targets_t[i] == target_lists_[gid][j]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    target_lists_[gid].push_back(targets_t[i]);
                    targets.push_back({gid, (unsigned)(target_lists_[gid].size()-1)});
                }
                else {
                    targets.push_back({gid, j});
                }
            }

            for(unsigned i = 0; i < sources_t.size(); i++) {
                auto source_gid = src_id.value()[i] + nodes_.partitions()[source_pop];
                // if we can't find the source in sources_lists, its cell hasn't been constructed yet
                bool found = false;
                unsigned j = 0;
                for (; j < source_lists_[source_gid].size(); j++) {
                    if (sources_t[i] == source_lists_[source_gid][j]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    source_lists_[source_gid].push_back(sources_t[i]);
                    sources.push_back({source_gid, (unsigned)(source_lists_[source_gid].size()-1)});
                }
                else {
                    sources.push_back({source_gid, j});
                }
            }

            for (unsigned i = 0; i < sources_t.size(); i++) {
                conns.emplace_back(sources[i], targets[i], weights[i], delays[i]);
            }
        }
    }
}

void database::get_sources_and_targets(cell_gid_type gid,
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

    std::vector<std::pair<arb::segment_location, double>> gather_src;
    for (auto i: source_edge_pops) {
        std::vector<std::pair<int, int>> source_edge_ranges;

        auto ind_id = edges_[i].find_group("indicies");
        auto s2t_id = edges_[i][ind_id].find_group("source_to_target");
        auto n2r_range = edges_[i][ind_id][s2t_id].dataset_2i_at("node_id_to_ranges", pop_id);
        for (auto j = n2r_range->first; j< n2r_range->second; j++) {
            auto r2e = edges_[i][ind_id][s2t_id].dataset_2i_at("range_to_edge_id", j);
            source_edge_ranges.push_back(r2e.value());
        }
        for (auto r: source_edge_ranges) {
            auto temp_srcs = get_afferent_range(i, r);

            for (unsigned i = 0; i< temp_srcs.size(); i++) {
                gather_src.push_back(std::make_pair(temp_srcs[i], 10));
            }
        }
    }

    // Order sources

    std::sort(gather_src.begin(), gather_src.end(), [](const auto &a, const auto& b) -> bool
    {
        return std::tie(a.first.segment, a.first.position) < std::tie(b.first.segment, b.first.position);
    });

    std::cout << std::endl;
    for (auto s : source_lists_[gid]) {
        auto s_pos = std::lower_bound(gather_src.begin(), gather_src.end(), s,
                                      [](const std::pair<arb::segment_location, double>& lhs, const arb::segment_location& rhs) -> bool
                                      { return std::tie(lhs.first.segment, lhs.first.position) < std::tie(rhs.segment, rhs.position); });
        if (s_pos != gather_src.end()) {
            if ((*s_pos).first == s) {
                src.push_back(*s_pos);
                gather_src.erase(s_pos);
            }
        }
    }

    for (auto s : gather_src) {
        src.push_back(s);
        source_lists_[gid].push_back(s.first);
    }

    std::vector<std::pair<arb::segment_location, arb::mechanism_desc>> gather_tgt;
    for (auto i: target_edge_pops) {
        std::vector<std::pair<int, int>> target_edge_ranges;

        auto ind_id = edges_[i].find_group("indicies");
        auto t2s_id = edges_[i][ind_id].find_group("target_to_source");
        auto n2r = edges_[i][ind_id][t2s_id].dataset_2i_at("node_id_to_ranges", pop_id);
        for (auto j = n2r->first; j< n2r->second; j++) {
            auto r2e = edges_[i][ind_id][t2s_id].dataset_2i_at("range_to_edge_id", j);
            target_edge_ranges.push_back(r2e.value());
        }
        for (auto r: target_edge_ranges) {
            auto temp_tgts = get_efferent_range(i, r);
            auto temp_syn = get_synapses_range(i, r);

            for (unsigned i = 0; i< temp_tgts.size(); i++) {
                gather_tgt.push_back(std::make_pair(temp_tgts[i], temp_syn[i]));
            }
        }
    }

    // Order targets

    std::sort(gather_tgt.begin(), gather_tgt.end(), [](const auto &a, const auto& b) -> bool
    {
        return std::tie(a.first.segment, a.first.position) < std::tie(b.first.segment, b.first.position);
    });

    for (auto t : target_lists_[gid]) {
        auto t_pos = std::lower_bound(gather_tgt.begin(), gather_tgt.end(), t,
                                      [](const std::pair<arb::segment_location, arb::mechanism_desc>& lhs, const arb::segment_location& rhs) -> bool
                                      { return std::tie(lhs.first.segment, lhs.first.position) < std::tie(rhs.segment, rhs.position); });
        if (t_pos != gather_tgt.end()) {
            if ((*t_pos).first == t) {
                tgt.push_back(*t_pos);
                gather_tgt.erase(t_pos);
            }
        }
    }

    for (auto t : gather_tgt) {
        tgt.push_back(t);
        target_lists_[gid].push_back(t.first);
    }
}

std::vector<arb::segment_location> database::get_afferent_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range)
{
    std::vector<arb::segment_location> out;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].dataset_int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].dataset_int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].dataset_int_range("edge_type_id", edge_range.first, edge_range.second);

    for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
        auto loc_grp_id = edges_grp_id.value()[i];

        int source_branch;
        double source_pos;

        bool found_source_branch = false;
        bool found_source_pos = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
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

        out.emplace_back(arb::segment_location((unsigned)source_branch, source_pos));
    }
    return out;
}

std::vector<arb::segment_location> database::get_efferent_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range)
{
    std::vector<arb::segment_location> out;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].dataset_int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].dataset_int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].dataset_int_range("edge_type_id", edge_range.first, edge_range.second);


    for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
        auto loc_grp_id = edges_grp_id.value()[i];

        int source_branch;
        double source_pos;

        bool found_source_branch = false;
        bool found_source_pos = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto group = edges_[edge_pop_id][std::to_string(loc_grp_id)].value();
            auto loc_grp_idx = edges_grp_idx.value()[i];

            if (group.find_dataset("efferent_section_id") != -1) {
                source_branch = group.dataset_i_at("efferent_section_id", loc_grp_idx).value();
                found_source_branch = true;
            }
            if (group.find_dataset("efferent_section_pos") != -1) {
                source_pos = group.dataset_d_at("efferent_section_pos", loc_grp_idx).value();
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
            unsigned source_branch_idx = edge_types_.map()["efferent_section_id"];
            source_branch = std::atoi(edge_types_.data()[source_branch_idx][loc_type_idx].c_str());
        }
        if (!found_source_pos) {
            unsigned source_pos_idx = edge_types_.map()["efferent_section_pos"];
            source_pos = std::atof(edge_types_.data()[source_pos_idx][loc_type_idx].c_str());
        }

        out.emplace_back(arb::segment_location((unsigned)source_branch, source_pos));
    }
    return out;
}

std::vector<arb::mechanism_desc> database::get_synapses_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range)
{
    std::vector<arb::mechanism_desc> out;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].dataset_int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].dataset_int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].dataset_int_range("edge_type_id", edge_range.first, edge_range.second);

    std::vector<std::pair<arb::segment_location, arb::mechanism_desc>> temp;

    for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
        auto loc_grp_id = edges_grp_id.value()[i];

        std::string synapse;

        bool found_synapse = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto group = edges_[edge_pop_id][std::to_string(loc_grp_id)].value();
            auto loc_grp_idx = edges_grp_idx.value()[i];
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

        if (!found_synapse) {
            unsigned synapse_idx = edge_types_.map()["model_template"];
            synapse = edge_types_.data()[synapse_idx][loc_type_idx];
        }

        out.emplace_back(arb::mechanism_desc(synapse));
    }

    return out;
}

std::vector<double> database::get_weight_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range)
{
    std::vector<double> out;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].dataset_int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].dataset_int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].dataset_int_range("edge_type_id", edge_range.first, edge_range.second);

    std::vector<std::pair<arb::segment_location, arb::mechanism_desc>> temp;

    for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
        auto loc_grp_id = edges_grp_id.value()[i];

        double weight;

        bool found_weight = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto group = edges_[edge_pop_id][std::to_string(loc_grp_id)].value();
            auto loc_grp_idx = edges_grp_idx.value()[i];
            if (group.find_dataset("syn_weight") != -1) {
                weight = std::atof(group.dataset_s_at("syn_weight", loc_grp_idx).value().c_str());
                found_weight = true;
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

        if (!found_weight) {
            unsigned weight_idx = edge_types_.map()["syn_weight"];
            weight = std::atof(edge_types_.data()[weight_idx][loc_type_idx].c_str());
        }

        out.emplace_back(weight);
    }

    return out;
}

std::vector<double> database::get_delay_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range)
{
    std::vector<double> out;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].dataset_int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].dataset_int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].dataset_int_range("edge_type_id", edge_range.first, edge_range.second);

    std::vector<std::pair<arb::segment_location, arb::mechanism_desc>> temp;

    for (unsigned i = 0; i < edges_grp_id.value().size(); i++) {
        auto loc_grp_id = edges_grp_id.value()[i];

        double delay;

        bool found_delay = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto group = edges_[edge_pop_id][std::to_string(loc_grp_id)].value();
            auto loc_grp_idx = edges_grp_idx.value()[i];
            if (group.find_dataset("delay") != -1) {
                delay = std::atof(group.dataset_s_at("delay", loc_grp_idx).value().c_str());
                found_delay = true;
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

        if (!found_delay) {
            unsigned weight_idx = edge_types_.map()["delay"];
            delay = std::atof(edge_types_.data()[weight_idx][loc_type_idx].c_str());
        }

        out.emplace_back(delay);
    }

    return out;
}