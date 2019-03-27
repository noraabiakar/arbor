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

using source_type = std::pair<arb::segment_location,double>;
template<> struct std::hash<source_type>
{
    std::size_t operator()(const source_type& s) const noexcept
    {
        std::size_t const h1(std::hash<unsigned>{}(s.first.segment));
        std::size_t const h2(std::hash<double>{}(s.first.position));
        std::size_t const h3(std::hash<double>{}(s.second));
        auto h1_2 = h1 ^ (h2 << 1);
        return (h1_2 >> 1) ^ (h3 << 1);
    }
};

using target_type = std::pair<arb::segment_location,std::string>;
template<> struct std::hash<target_type>
{
    std::size_t operator()(const target_type& s) const noexcept
    {
        std::size_t const h1(std::hash<unsigned>{}(s.first.segment));
        std::size_t const h2(std::hash<double>{}(s.first.position));
        std::size_t const h3(std::hash<std::string>{}(s.second));
        auto h1_2 = h1 ^ (h2 << 1);
        return (h1_2 >> 1) ^ (h3 << 1);
    }
};

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
    unsigned num_sources(cell_gid_type gid);
    unsigned num_targets(cell_gid_type gid);

private:

    struct local_node{
        cell_gid_type pop_id;
        cell_gid_type node_id;
    };

    local_node localize(cell_gid_type gid) {
        unsigned i = 0;
        for (; i < nodes_.partitions().size(); i++) {
            if (gid < nodes_.partitions()[i]) {
                return {i-1, gid - nodes_.partitions()[i-1]};
            }
        }
        return local_node();
    }

    cell_gid_type globalize(local_node n) {
        return n.node_id + nodes_.partitions()[n.pop_id];
    }

    std::unordered_map <unsigned, unsigned> edge_to_source_of_target(unsigned target_pop) {
        std::unordered_map <unsigned, unsigned> edge_to_source;
        unsigned src_vec_id = edge_types_.map()["source_pop_name"];
        unsigned tgt_vec_id = edge_types_.map()["target_pop_name"];
        unsigned edge_vec_id = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[src_vec_id].size(); i++) {
            if (edge_types_.data()[tgt_vec_id][i] == nodes_[target_pop].name()) {
                auto edge_pop = edge_types_.data()[edge_vec_id][i];
                auto source_pop = edge_types_.data()[src_vec_id][i];
                edge_to_source[edges_.map()[edge_pop]] = nodes_.map()[source_pop];
            }
        }
        return edge_to_source;
    }

    std::vector<unsigned> edges_of_target(unsigned target_pop) {
        std::vector<unsigned> target_edge_pops;
        unsigned tgt_vec_id = edge_types_.map()["target_pop_name"];
        unsigned edge_vec_id = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[tgt_vec_id].size(); i++) {
            if (edge_types_.data()[tgt_vec_id][i] == nodes_[target_pop].name()) {
                auto e_pop = edge_types_.data()[edge_vec_id][i];
                target_edge_pops.push_back(edges_.map()[e_pop]);
            }
        }
        std::sort(target_edge_pops.begin(), target_edge_pops.end());
        target_edge_pops.erase(std::unique(target_edge_pops.begin(), target_edge_pops.end()), target_edge_pops.end() );
        return target_edge_pops;
    }

    std::vector<unsigned> edges_of_source(unsigned source_pop) {
        std::vector<unsigned> source_edge_pops;
        unsigned src_vec_id = edge_types_.map()["source_pop_name"];
        unsigned edge_vec_id = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[src_vec_id].size(); i++) {
            if (edge_types_.data()[src_vec_id][i] == nodes_[source_pop].name()) {
                auto e_pop = edge_types_.data()[edge_vec_id][i];
                source_edge_pops.push_back(edges_.map()[e_pop]);
            }
        }
        std::sort(source_edge_pops.begin(), source_edge_pops.end());
        source_edge_pops.erase(std::unique(source_edge_pops.begin(), source_edge_pops.end()), source_edge_pops.end() );
        return source_edge_pops;
    }

    enum datatype { string_t, int_t, double_t };
    // list of targets
    void fill_target_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range,
            std::vector<arb::segment_location>& targets );

    // list of sources
    void fill_source_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range,
            std::vector<arb::segment_location>& sources);

    void fill_conn_range(
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range,
            std::vector<arb::segment_location>& sources,
            std::vector<arb::segment_location>& targets,
            std::vector<double>& weights,
            std::vector<double>& delays);

    void fill_range_of(
            std::string field,
            unsigned edge_pop_id,
            std::pair<unsigned, unsigned> edge_range,
            datatype t,
            std::vector<arb::util::any>& out);

    hdf5_record nodes_;
    hdf5_record edges_;
    csv_record node_types_;
    csv_record edge_types_;

    std::unordered_map<cell_gid_type, std::unordered_map<source_type, unsigned>> source_maps_;
    std::unordered_map<cell_gid_type, std::unordered_map<target_type, unsigned>> target_maps_;
};

void database::get_connections(cell_gid_type gid, std::vector<arb::cell_connection>& conns) {

    // Find cell local index in population
    auto loc_node = localize(gid);
    auto edge_to_source = edge_to_source_of_target(loc_node.pop_id);

    for (auto i: edge_to_source) {
        auto edge_pop = i.first;
        auto source_pop = i.second;

        auto ind_id = edges_[edge_pop].find_group("indicies");
        auto s2t_id = edges_[edge_pop][ind_id].find_group("target_to_source");
        auto n2r_range = edges_[edge_pop][ind_id][s2t_id].int_pair_at("node_id_to_ranges", loc_node.node_id);

        for (auto j = n2r_range.first; j< n2r_range.second; j++) {
            auto r2e = edges_[edge_pop][ind_id][s2t_id].int_pair_at("range_to_edge_id", j);

            std::vector<arb::segment_location> targets_t, sources_t;
            std::vector<double> weights, delays;
            fill_conn_range(edge_pop, r2e, sources_t, targets_t, weights, delays);

            std::vector<arb::util::any> thresholds;
            fill_range_of("threshold", edge_pop, r2e, datatype::double_t, thresholds);

            std::vector<arb::util::any> syns;
            fill_range_of("model_template", edge_pop, r2e, datatype::string_t, syns);

            auto src_id = edges_[edge_pop].int_range("source_node_id", r2e.first, r2e.second);

            std::vector<cell_member_type> sources, targets;

            for(unsigned s = 0; s < sources_t.size(); s++) {
                auto source_gid = globalize({source_pop, (cell_gid_type)src_id[s]});

                // if we can't find the target in target_lists, its cell hasn't been constructed yet
                source_type source_pair = std::make_pair(sources_t[s], arb::util::any_cast<double>(thresholds[s]));
                auto loc = source_maps_[gid].find(source_pair);
                if (loc == source_maps_[gid].end()) {
                    auto p = source_maps_[gid].size();
                    source_maps_[gid][source_pair] = p;
                    sources.push_back({source_gid, (unsigned)p});
                }
                else {
                    sources.push_back({source_gid, loc->second});
                }
            }

            for(unsigned t = 0; t < targets_t.size(); t++) {
                // if we can't find the target in target_lists, its cell hasn't been constructed yet
                target_type target_pair = std::make_pair(targets_t[t], arb::util::any_cast<std::string>(syns[t]));
                auto loc = target_maps_[gid].find(target_pair);
                if (loc == target_maps_[gid].end()) {
                    auto p = target_maps_[gid].size();
                    target_maps_[gid][target_pair] = p;
                    targets.push_back({gid, (unsigned)p});
                }
                else {
                    targets.push_back({gid, loc->second});
                }
            }

            for (unsigned k = 0; k < sources_t.size(); k++) {
                conns.emplace_back(sources[k], targets[k], weights[k], delays[k]);
            }
        }
    }
}

void database::get_sources_and_targets(cell_gid_type gid,
                             std::vector<std::pair<arb::segment_location, double>>& src,
                             std::vector<std::pair<arb::segment_location, arb::mechanism_desc>>& tgt) {

    auto loc_node = localize(gid);
    auto source_edge_pops = edges_of_source(loc_node.pop_id);
    auto target_edge_pops = edges_of_target(loc_node.pop_id);

    for (auto i: source_edge_pops) {
        auto ind_id = edges_[i].find_group("indicies");
        auto s2t_id = edges_[i][ind_id].find_group("source_to_target");
        auto n2r_range = edges_[i][ind_id][s2t_id].int_pair_at("node_id_to_ranges", loc_node.node_id);

        for (auto j = n2r_range.first; j< n2r_range.second; j++) {
            auto r2e = edges_[i][ind_id][s2t_id].int_pair_at("range_to_edge_id", j);
            std::vector<arb::segment_location> sources;
            fill_source_range(i, r2e, sources);

            std::vector<arb::util::any> thresholds;
            fill_range_of("threshold", i, r2e, datatype::double_t, thresholds);

            for (unsigned j = 0; j< sources.size(); j++) {
                source_type source_pair = std::make_pair(sources[j], arb::util::any_cast<double>(thresholds[j]));
                auto loc = source_maps_[gid].find(source_pair);
                if (loc == source_maps_[gid].end()) {
                    source_maps_[gid][source_pair] = source_maps_[gid].size();
                }
            }
        }
    }
    src.resize(source_maps_[gid].size(), std::make_pair(arb::segment_location(0, 0.0), 0.0));
    for (auto s: source_maps_[gid]) {
        src[s.second] = s.first;
    }

    for (auto i: target_edge_pops) {
        auto ind_id = edges_[i].find_group("indicies");
        auto t2s_id = edges_[i][ind_id].find_group("target_to_source");
        auto n2r = edges_[i][ind_id][t2s_id].int_pair_at("node_id_to_ranges", loc_node.node_id);
        for (auto j = n2r.first; j< n2r.second; j++) {
            auto r2e = edges_[i][ind_id][t2s_id].int_pair_at("range_to_edge_id", j);
            std::vector<arb::segment_location> targets;
            fill_target_range(i, r2e, targets);

            std::vector<arb::util::any> syns;
            fill_range_of("model_template", i, r2e, datatype::string_t, syns);

            for (unsigned j = 0; j< targets.size(); j++) {
                target_type target_pair = std::make_pair(targets[j], arb::util::any_cast<std::string>(syns[j]));
                auto loc = target_maps_[gid].find(target_pair);
                if (loc == target_maps_[gid].end()) {
                    target_maps_[gid][target_pair] = target_maps_[gid].size();
                }
            };
        }
    }
    tgt.resize(target_maps_[gid].size(), std::make_pair(arb::segment_location(0, 0.0), arb::mechanism_desc("")));
    for (auto t: target_maps_[gid]) {
        tgt[t.second] = std::make_pair(t.first.first, arb::mechanism_desc(t.first.second));
    }
}

void database::fill_source_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range,
        std::vector<arb::segment_location>& out)
{
    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);

    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        int source_branch;
        double source_pos;

        bool found_source_branch = false;
        bool found_source_pos = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto lgi = edges_[edge_pop_id].find_group(std::to_string(loc_grp_id));
            auto group = edges_[edge_pop_id][lgi];
            auto loc_grp_idx = edges_grp_idx[i];

            if (group.find_dataset("efferent_section_id") != -1) {
                source_branch = group.int_at("efferent_section_id", loc_grp_idx);
                found_source_branch = true;
            }
            if (group.find_dataset("efferent_section_pos") != -1) {
                source_pos = group.double_at("efferent_section_pos", loc_grp_idx);
                found_source_pos = true;
            }
        }

        // name and index of edge_type_id
        auto e_type = edges_type[i];
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
}

void database::fill_target_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range,
        std::vector<arb::segment_location>& out)
{
    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);


    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        int target_branch;
        double target_pos;

        bool found_target_branch = false;
        bool found_target_pos = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto lgi = edges_[edge_pop_id].find_group(std::to_string(loc_grp_id));
            auto group = edges_[edge_pop_id][lgi];
            auto loc_grp_idx = edges_grp_idx[i];

            if (group.find_dataset("afferent_section_id") != -1) {
                target_branch = group.int_at("afferent_section_id", loc_grp_idx);
                found_target_branch = true;
            }
            if (group.find_dataset("afferent_section_pos") != -1) {
                target_pos = group.double_at("afferent_section_pos", loc_grp_idx);
                found_target_pos = true;
            }
        }

        // name and index of edge_type_id
        auto e_type = edges_type[i];
        unsigned type_idx = edge_types_.map()["edge_type_id"];

        // find specific index of edge_type in type_idx
        unsigned loc_type_idx;
        for (loc_type_idx = 0; loc_type_idx < edge_types_.data()[type_idx].size(); loc_type_idx++) {
            if (e_type == std::atoi(edge_types_.data()[type_idx][loc_type_idx].c_str())) {
                break;
            }
        }

        if (!found_target_branch) {
            unsigned target_branch_idx = edge_types_.map()["afferent_section_id"];
            target_branch = std::atoi(edge_types_.data()[target_branch_idx][loc_type_idx].c_str());
        }
        if (!found_target_pos) {
            unsigned target_pos_idx = edge_types_.map()["afferent_section_pos"];
            target_pos = std::atof(edge_types_.data()[target_pos_idx][loc_type_idx].c_str());
        }

        out.emplace_back(arb::segment_location((unsigned)target_branch, target_pos));
    }
}

void database::fill_conn_range(
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range,
        std::vector<arb::segment_location>& sources,
        std::vector<arb::segment_location>& targets,
        std::vector<double>& weights,
        std::vector<double>& delays)
{
    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);

    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        int source_branch, target_branch;
        double source_pos, target_pos, weight, delay;

        bool found_source_branch = false;
        bool found_source_pos = false;

        bool found_target_branch = false;
        bool found_target_pos = false;

        bool found_weight = false;
        bool found_delay = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto lgi = edges_[edge_pop_id].find_group(std::to_string(loc_grp_id));
            auto group = edges_[edge_pop_id][lgi];
            auto loc_grp_idx = edges_grp_idx[i];

            if (group.find_dataset("afferent_section_id") != -1) {
                target_branch = group.int_at("afferent_section_id", loc_grp_idx);
                found_target_branch = true;
            }
            if (group.find_dataset("afferent_section_pos") != -1) {
                target_pos = group.double_at("afferent_section_pos", loc_grp_idx);
                found_target_pos = true;
            }
            if (group.find_dataset("efferent_section_id") != -1) {
                source_branch = group.int_at("efferent_section_id", loc_grp_idx);
                found_source_branch = true;
            }
            if (group.find_dataset("efferent_section_pos") != -1) {
                source_pos = group.double_at("efferent_section_pos", loc_grp_idx);
                found_source_pos = true;
            }
            if (group.find_dataset("syn_weight") != -1) {
                weight = group.double_at("syn_weight", loc_grp_idx);
                found_weight = true;
            }
            if (group.find_dataset("delay") != -1) {
                delay = group.double_at("delay", loc_grp_idx);
                found_delay = true;
            }
        }

        // name and index of edge_type_id
        auto e_type = edges_type[i];
        unsigned type_idx = edge_types_.map()["edge_type_id"];

        // find specific index of edge_type in type_idx
        unsigned loc_type_idx;
        for (loc_type_idx = 0; loc_type_idx < edge_types_.data()[type_idx].size(); loc_type_idx++) {
            if (e_type == std::atoi(edge_types_.data()[type_idx][loc_type_idx].c_str())) {
                break;
            }
        }

        if (!found_target_branch) {
            unsigned target_branch_idx = edge_types_.map()["afferent_section_id"];
            target_branch = std::atoi(edge_types_.data()[target_branch_idx][loc_type_idx].c_str());
        }
        if (!found_target_pos) {
            unsigned target_pos_idx = edge_types_.map()["afferent_section_pos"];
            target_pos = std::atof(edge_types_.data()[target_pos_idx][loc_type_idx].c_str());
        }
        if (!found_source_branch) {
            unsigned source_branch_idx = edge_types_.map()["efferent_section_id"];
            source_branch = std::atoi(edge_types_.data()[source_branch_idx][loc_type_idx].c_str());
        }
        if (!found_source_pos) {
            unsigned source_pos_idx = edge_types_.map()["efferent_section_pos"];
            source_pos = std::atof(edge_types_.data()[source_pos_idx][loc_type_idx].c_str());
        }
        if (!found_weight) {
            unsigned weight_idx = edge_types_.map()["syn_weight"];
            weight = std::atof(edge_types_.data()[weight_idx][loc_type_idx].c_str());
        }
        if (!found_delay) {
            unsigned delay_idx = edge_types_.map()["delay"];
            delay = std::atof(edge_types_.data()[delay_idx][loc_type_idx].c_str());
        }

        sources.emplace_back(arb::segment_location((unsigned)source_branch, source_pos));
        targets.emplace_back(arb::segment_location((unsigned)target_branch, target_pos));
        weights.emplace_back(weight);
        delays.emplace_back(delay);

    }
}

void database::fill_range_of(
        std::string field,
        unsigned edge_pop_id,
        std::pair<unsigned, unsigned> edge_range,
        datatype type,
        std::vector<arb::util::any>& out)
{
    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);

    std::vector<std::pair<arb::segment_location, arb::mechanism_desc>> temp;

    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        arb::util::any data;
        bool found_field = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto lgi = edges_[edge_pop_id].find_group(std::to_string(loc_grp_id));
            auto group = edges_[edge_pop_id][lgi];
            auto loc_grp_idx = edges_grp_idx[i];
            if (group.find_dataset(field) != -1) {
                switch (type) {
                    case string_t:
                        data = group.string_at(field, loc_grp_idx);
                        break;
                    case int_t:
                        data = group.int_at(field, loc_grp_id);
                        break;
                    case double_t:
                        data = group.double_at(field, loc_grp_id);
                        break;
                    default: break;
                }
                found_field = true;
            }
        }

        // name and index of edge_type_id
        auto e_type = edges_type[i];
        unsigned type_idx = edge_types_.map()["edge_type_id"];

        // find specific index of edge_type in type_idx
        unsigned loc_type_idx;
        for (loc_type_idx = 0; loc_type_idx < edge_types_.data()[type_idx].size(); loc_type_idx++) {
            if (e_type == std::atoi(edge_types_.data()[type_idx][loc_type_idx].c_str())) {
                break;
            }
        }

        if (!found_field) {
            unsigned synapse_idx = edge_types_.map()[field];
            switch (type) {
                case string_t:
                    data = edge_types_.data()[synapse_idx][loc_type_idx];
                    break;
                case int_t:
                    data = std::atoi(edge_types_.data()[synapse_idx][loc_type_idx].c_str());
                    break;
                case double_t:
                    data = std::atof(edge_types_.data()[synapse_idx][loc_type_idx].c_str());
                    break;
                default: break;
            }
        }

        out.emplace_back(data);
    }
}

unsigned database::num_sources(cell_gid_type gid) {
    auto loc_node = localize(gid);
    auto source_edge_pops = edges_of_source(loc_node.pop_id);

    unsigned sum = 0;
    for (auto i: source_edge_pops) {
        std::vector<std::pair<int, int>> source_edge_ranges;

        auto ind_id = edges_[i].find_group("indicies");
        auto s2t_id = edges_[i][ind_id].find_group("source_to_target");
        auto n2r_range = edges_[i][ind_id][s2t_id].int_pair_at("node_id_to_ranges", loc_node.node_id);
        for (auto j = n2r_range.first; j< n2r_range.second; j++) {
            auto r2e = edges_[i][ind_id][s2t_id].int_pair_at("range_to_edge_id", j);
            source_edge_ranges.push_back(r2e);
        }
        for (auto r: source_edge_ranges) {
            sum += (r.second - r.first);
        }
    }
    return sum;
}

unsigned database::num_targets(cell_gid_type gid) {
    auto loc_node = localize(gid);
    auto target_edge_pops = edges_of_target(loc_node.pop_id);

    unsigned sum = 0;
    for (auto i: target_edge_pops) {
        std::vector<std::pair<int, int>> target_edge_ranges;

        auto ind_id = edges_[i].find_group("indicies");
        auto s2t_id = edges_[i][ind_id].find_group("target_to_source");
        auto n2r_range = edges_[i][ind_id][s2t_id].int_pair_at("node_id_to_ranges", loc_node.node_id);
        for (auto j = n2r_range.first; j< n2r_range.second; j++) {
            auto r2e = edges_[i][ind_id][s2t_id].int_pair_at("range_to_edge_id", j);
            target_edge_ranges.push_back(r2e);
        }
        for (auto r: target_edge_ranges) {
            sum += (r.second - r.first);
        }
    }
    return sum;
}