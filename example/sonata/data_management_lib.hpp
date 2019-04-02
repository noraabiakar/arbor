#include <arbor/common_types.hpp>
#include <arbor/util/optional.hpp>
#include <string.h>
#include <stdio.h>
#include <hdf5.h>
#include <assert.h>
#include <unordered_set>
#include <malloc.h>

#include "hdf5_lib.hpp"
#include "csv_lib.hpp"

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::segment_location;

using source_type = std::pair<segment_location,double>;
using target_type = std::pair<segment_location,std::string>;

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
                                 std::vector<std::pair<segment_location, double>>& src,
                                 std::vector<std::pair<segment_location, arb::mechanism_desc>>& tgt);
    unsigned num_sources(cell_gid_type gid);
    unsigned num_targets(cell_gid_type gid);

private:

    /* Read relevant information from HDF5 or CSV */
    std::vector<source_type> source_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range);
    std::vector<target_type> target_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range);
    std::vector<double> weight_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range);
    std::vector<double> delay_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range);

    /* Helper functions */
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

    std::unordered_set<unsigned> edges_of_target(unsigned target_pop) {
        std::unordered_set<unsigned> target_edge_pops;
        unsigned tgt_vec_id = edge_types_.map()["target_pop_name"];
        unsigned edge_vec_id = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[tgt_vec_id].size(); i++) {
            if (edge_types_.data()[tgt_vec_id][i] == nodes_[target_pop].name()) {
                auto e_pop = edge_types_.data()[edge_vec_id][i];
                target_edge_pops.insert(edges_.map()[e_pop]);
            }
        }
        return target_edge_pops;
    }

    std::unordered_set<unsigned> edges_of_source(unsigned source_pop) {
        std::unordered_set<unsigned> source_edge_pops;
        unsigned src_vec_id = edge_types_.map()["source_pop_name"];
        unsigned edge_vec_id = edge_types_.map()["pop_name"];

        for (unsigned i = 0; i < edge_types_.data()[src_vec_id].size(); i++) {
            if (edge_types_.data()[src_vec_id][i] == nodes_[source_pop].name()) {
                auto e_pop = edge_types_.data()[edge_vec_id][i];
                source_edge_pops.insert(edges_.map()[e_pop]);
            }
        }
        return source_edge_pops;
    }

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

            auto src_rng = source_range(edge_pop, r2e);
            auto tgt_rng = target_range(edge_pop, r2e);
            auto weights = weight_range(edge_pop, r2e);
            auto delays = delay_range(edge_pop, r2e);

            auto src_id = edges_[edge_pop].int_range("source_node_id", r2e.first, r2e.second);

            std::vector<cell_member_type> sources, targets;

            for(unsigned s = 0; s < src_rng.size(); s++) {
                auto source_gid = globalize({source_pop, (cell_gid_type)src_id[s]});

                auto loc = source_maps_[gid].find(src_rng[s]);
                if (loc == source_maps_[gid].end()) {
                    auto p = source_maps_[gid].size();
                    source_maps_[gid][src_rng[s]] = p;
                    sources.push_back({source_gid, (unsigned)p});
                }
                else {
                    sources.push_back({source_gid, loc->second});
                }
            }

            for(unsigned t = 0; t < tgt_rng.size(); t++) {
                auto loc = target_maps_[gid].find(tgt_rng[t]);
                if (loc == target_maps_[gid].end()) {
                    auto p = target_maps_[gid].size();
                    target_maps_[gid][tgt_rng[t]] = p;
                    targets.push_back({gid, (unsigned)p});
                }
                else {
                    targets.push_back({gid, loc->second});
                }
            }

            for (unsigned k = 0; k < sources.size(); k++) {
                conns.emplace_back(sources[k], targets[k], weights[k], delays[k]);
            }
        }
    }
}

void database::get_sources_and_targets(cell_gid_type gid,
                             std::vector<std::pair<segment_location, double>>& src,
                             std::vector<std::pair<segment_location, arb::mechanism_desc>>& tgt) {

    auto loc_node = localize(gid);
    auto source_edge_pops = edges_of_source(loc_node.pop_id);
    auto target_edge_pops = edges_of_target(loc_node.pop_id);

    for (auto i: source_edge_pops) {
        auto ind_id = edges_[i].find_group("indicies");
        auto s2t_id = edges_[i][ind_id].find_group("source_to_target");
        auto n2r_range = edges_[i][ind_id][s2t_id].int_pair_at("node_id_to_ranges", loc_node.node_id);

        for (auto j = n2r_range.first; j< n2r_range.second; j++) {
            auto r2e = edges_[i][ind_id][s2t_id].int_pair_at("range_to_edge_id", j);
            auto src_rng = source_range(i, r2e);
            for (auto s: src_rng) {
                auto loc = source_maps_[gid].find(s);
                if (loc == source_maps_[gid].end()) {
                    source_maps_[gid][s] = source_maps_[gid].size();
                }
            }
        }
    }
    src.resize(source_maps_[gid].size(), std::make_pair(segment_location(0, 0.0), 0.0));
    for (auto s: source_maps_[gid]) {
        src[s.second] = s.first;
    }

    for (auto i: target_edge_pops) {
        auto ind_id = edges_[i].find_group("indicies");
        auto t2s_id = edges_[i][ind_id].find_group("target_to_source");
        auto n2r = edges_[i][ind_id][t2s_id].int_pair_at("node_id_to_ranges", loc_node.node_id);
        for (auto j = n2r.first; j< n2r.second; j++) {
            auto r2e = edges_[i][ind_id][t2s_id].int_pair_at("range_to_edge_id", j);
            auto tgt_rng = target_range(i, r2e);
            for (auto t: tgt_rng) {
                auto loc = target_maps_[gid].find(t);
                if (loc == target_maps_[gid].end()) {
                    target_maps_[gid][t] = target_maps_[gid].size();
                }
            }
        }
    }
    tgt.resize(target_maps_[gid].size(), std::make_pair(segment_location(0, 0.0), arb::mechanism_desc("")));
    for (auto t: target_maps_[gid]) {
        tgt[t.second] = std::make_pair(t.first.first, arb::mechanism_desc(t.first.second));
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Private helper functions

// Read from HDF5 file/ CSV file depending on where the information is available

std::vector<source_type> database::source_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range) {
    std::vector<source_type> ret;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);

    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        int source_branch;
        double source_pos, threshold;

        bool found_source_branch = false;
        bool found_source_pos = false;
        bool found_threshold =false;

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
            if (group.find_dataset("threshold") != -1) {
                threshold = group.double_at("threshold", loc_grp_idx);
                found_threshold = true;
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
        if (!found_threshold) {
            unsigned synapse_idx = edge_types_.map()["threshold"];
            threshold = std::atof(edge_types_.data()[synapse_idx][loc_type_idx].c_str());
        }

        ret.emplace_back(segment_location((unsigned)source_branch, source_pos), threshold);
    }
    return ret;
}

std::vector<target_type> database::target_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range) {
    std::vector<target_type> ret;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);

    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        int target_branch;
        double target_pos;
        std::string synapse;

        bool found_target_branch = false;
        bool found_target_pos = false;
        bool found_synapse = false;

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
            if (group.find_dataset("model_template") != -1) {
                synapse = group.string_at("model_template", loc_grp_idx);
                found_synapse = true;
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
        if (!found_synapse) {
            unsigned synapse_idx = edge_types_.map()["model_template"];
            synapse = edge_types_.data()[synapse_idx][loc_type_idx];
        }

        ret.emplace_back(segment_location((unsigned)target_branch, target_pos), synapse);
    }
    return ret;
}

std::vector<double> database::weight_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range) {
    std::vector<double> ret;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);

    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        double weight;
        bool found_weight = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto lgi = edges_[edge_pop_id].find_group(std::to_string(loc_grp_id));
            auto group = edges_[edge_pop_id][lgi];
            auto loc_grp_idx = edges_grp_idx[i];

            if (group.find_dataset("syn_weight") != -1) {
                weight = group.double_at("syn_weight", loc_grp_idx);
                found_weight = true;
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

        if (!found_weight) {
            unsigned weight_idx = edge_types_.map()["syn_weight"];
            weight = std::atof(edge_types_.data()[weight_idx][loc_type_idx].c_str());
        }
        ret.emplace_back(weight);
    }
    return ret;
}

std::vector<double> database::delay_range(unsigned edge_pop_id, std::pair<unsigned, unsigned> edge_range) {
    std::vector<double> ret;

    // First read edge_group_id and edge_group_index and edge_type
    auto edges_grp_id = edges_[edge_pop_id].int_range("edge_group_id", edge_range.first, edge_range.second);
    auto edges_grp_idx = edges_[edge_pop_id].int_range("edge_group_index", edge_range.first, edge_range.second);
    auto edges_type = edges_[edge_pop_id].int_range("edge_type_id", edge_range.first, edge_range.second);

    for (unsigned i = 0; i < edges_grp_id.size(); i++) {
        auto loc_grp_id = edges_grp_id[i];

        double delay;
        bool found_delay = false;

        // if the edges are in groups, for each edge find the group, if it exists
        if (edges_[edge_pop_id].find_group(std::to_string(loc_grp_id)) != -1) {
            auto lgi = edges_[edge_pop_id].find_group(std::to_string(loc_grp_id));
            auto group = edges_[edge_pop_id][lgi];
            auto loc_grp_idx = edges_grp_idx[i];

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

        if (!found_delay) {
            unsigned delay_idx = edge_types_.map()["delay"];
            delay = std::atof(edge_types_.data()[delay_idx][loc_type_idx].c_str());
        }
        ret.emplace_back(delay);
    }
    return ret;
}