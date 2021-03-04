#pragma once

#include <array>
#include <random>

#include <nlohmann/json.hpp>

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/common_types.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/string_literals.hpp>

#include <arborio/cableio.hpp>

#include <string>
#include <sup/json_params.hpp>

using namespace arb::literals;

// Parameters used to generate the random cell morphologies.
struct cell_parameters {
    cell_parameters() = default;

    //  Maximum number of levels in the cell (not including the soma)
    unsigned max_depth = 5;

    // The following parameters are described as ranges.
    // The first value is at the soma, and the last value is used on the last level.
    // Values at levels in between are found by linear interpolation.
    std::array<double,2> branch_probs = {1.0, 0.5}; //  Probability of a branch occuring.
    std::array<unsigned,2> compartments = {20, 2};  //  Compartment count on a branch.
    std::array<double,2> lengths = {200, 20};       //  Length of branch in μm.

    // The number of synapses per cell.
    unsigned synapses = 1;
};

cell_parameters parse_cell_parameters(nlohmann::json& json) {
    cell_parameters params;
    sup::param_from_json(params.max_depth, "depth", json);
    sup::param_from_json(params.branch_probs, "branch-probs", json);
    sup::param_from_json(params.compartments, "compartments", json);
    sup::param_from_json(params.lengths, "lengths", json);
    sup::param_from_json(params.synapses, "synapses", json);

    return params;
}

// Helper used to interpolate in branch_cell.
template <typename T>
double interp(const std::array<T,2>& r, unsigned i, unsigned n) {
    double p = i * 1./(n-1);
    double r0 = r[0];
    double r1 = r[1];
    return r[0] + p*(r1-r0);
}

std::ostream& operator<<(std::ostream& out, const arb::cv_policy&) {
    return out;
};

arb::cable_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params) {
    arb::segment_tree tree;

    // Add soma.
    double srad = 12.6157/2.0; // soma radius
    int stag = 1; // soma tag
    tree.append(arb::mnpos, {0, 0,-srad, srad}, {0, 0, srad, srad}, stag); // For area of 500 μm².

    std::vector<std::vector<unsigned>> levels;
    levels.push_back({0});

    // Standard mersenne_twister_engine seeded with gid.
    std::mt19937 gen(gid);
    std::uniform_real_distribution<double> dis(0, 1);

    double drad = 0.5; // Diameter of 1 μm for each dendrite cable.
    int dtag = 3;      // Dendrite tag.

    double dist_from_soma = srad; // Start dendrite at the edge of the soma.
    for (unsigned i=0; i<params.max_depth; ++i) {
        // Branch prob at this level.
        double bp = interp(params.branch_probs, i, params.max_depth);
        // Length at this level.
        double l = interp(params.lengths, i, params.max_depth);
        // Number of compartments at this level.
        unsigned nc = std::round(interp(params.compartments, i, params.max_depth));

        std::vector<unsigned> sec_ids;
        for (unsigned sec: levels[i]) {
            for (unsigned j=0; j<2; ++j) {
                if (dis(gen)<bp) {
                    auto z = dist_from_soma;
                    auto dz = l/nc;
                    auto p = sec;
                    for (unsigned k=0; k<nc; ++k) {
                        p = tree.append(p, {0,0,z+(k+1)*dz, drad}, dtag);
                    }
                    sec_ids.push_back(p);
                }
            }
        }
        if (sec_ids.empty()) {
            break;
        }
        levels.push_back(sec_ids);

        dist_from_soma += l;
    }

    arb::label_dict labels;

    using arb::reg::tagged;
    labels.set("soma", tagged(stag));
    labels.set("dend", tagged(dtag));

    arb::decor decor;

    decor.paint("soma"_lab, "hh");
    decor.paint("dend"_lab, "pas");

    decor.set_default(arb::axial_resistivity{100}); // [Ω·cm]

    // Add spike threshold detector at the soma.
    decor.place(arb::mlocation{0,0}, arb::threshold_detector{10});

    // Add a synapse to the mid point of the first dendrite.
    decor.place(arb::mlocation{0, 0.5}, "expsyn");

    // Add additional synapses that will not be connected to anything.
    for (unsigned i=1u; i<params.synapses; ++i) {
        decor.place(arb::mlocation{1, 0.5}, "expsyn");
    }

    // Make a CV between every sample in the sample tree.
    decor.set_default(arb::cv_policy_every_segment());

    arb::cable_cell cell(arb::morphology(tree), labels, decor);
    if (!gid) {
        std::string dicty = "(label-dict (region-def \"soma\"\n"
                            "   (tag 1))\n"
                            " (region-def \"dend\"\n"
                            "   (tag 3)))";
        std::string decory = "(decorations (place\n"
                             "   (location 0 0.5)\n"
                             "   (mechanism \"expsyn\"))\n"
                             " (place\n"
                             "   (location 0 0)\n"
                             "   (threshold-detector 10.000000))\n"
                             " (paint\n"
                             "   (region \"dend\")\n"
                             "   (mechanism \"pas\"))\n"
                             " (paint\n"
                             "   (region \"soma\")\n"
                             "   (mechanism \"hh\"))\n"
                             " (default\n"
                             "   (axial-resistivity 100.000000)))";
        std::string morphy = "(morphology \n"
                             "    (branch 0 -1 \n"
                             "      (segment 0 \n"
                             "        (point -6.300000 0.000000 0.000000 6.300000)\n"
                             "        (point 6.300000 0.000000 0.000000 6.300000)\n"
                             "        1)\n"
                             "      (segment 1 \n"
                             "        (point 6.300000 0.000000 0.000000 0.500000)\n"
                             "        (point 206.300000 0.000000 0.000000 0.200000)\n"
                             "        3)))";
        std::string celly = "(cable-cell \n"
                            "  (label-dict \n"
                            "    (region-def \"soma\" \n"
                            "      (tag 1))\n"
                            "    (region-def \"dend\" \n"
                            "      (join \n"
                            "        (join \n"
                            "          (tag 3)\n"
                            "          (tag 4))\n"
                            "        (tag 42))))\n"
                            "  (decorations \n"
                            "    (place \n"
                            "      (location 0 1)\n"
                            "      (mechanism \"exp2syn\"))\n"
                            "    (paint \n"
                            "      (region \"dend\")\n"
                            "      (mechanism \"pas\"))\n"
                            "    (paint \n"
                            "      (region \"soma\")\n"
                            "      (mechanism \"hh\")))"
                            ")";

        if (auto v = arborio::parse_label_dict(dicty)) {
//            arborio::write_s_expr(std::cout, v.value());
            std::cout << std::endl << std::endl;
        }
        else {
            throw v.error();
        }

        if (auto v = arborio::parse_decor(decory)) {
//            arborio::write_s_expr(std::cout, v.value());
            std::cout << std::endl << std::endl;
        }
        else {
            throw v.error();
        }

        if (auto v = arborio::parse_morphology(morphy)) {
//            arborio::write_s_expr(std::cout, v.value());
            std::cout << std::endl << std::endl;
        }
        else {
            throw v.error();
        }
        if (auto v = arborio::parse_cable_cell(celly)) {
            arborio::write_s_expr(std::cout, v.value());
            std::cout << std::endl << std::endl;
        }
        else {
            throw v.error();
        }
    }

    return cell;
}
