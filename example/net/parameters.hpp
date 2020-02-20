#include <iostream>

#include <array>
#include <cmath>
#include <fstream>
#include <random>

#include <arbor/cable_cell.hpp>
#include <sup/json_params.hpp>

struct sim_params {
    std::string network_file;
    std::string morph_file;
    double temp;
    double v_init;
    double run_time;
    double dt;
    unsigned probe_gid;
};

arb::cable_cell basket_cell() {
    using mech = arb::mechanism_desc;

    arb::sample_tree tree;
    arb::label_dict dict;

    // Add soma.
    tree.append(arb::mnpos, {{0,0,0 ,15.0/2.0}, 1});
    tree.append(0,          {{0,0,20,15.0/2.0}, 1});
    dict.set("soma", arb::reg::tagged(1));

    arb::cable_cell cell(arb::morphology(tree, false), dict);

    // Add denisty mechanisms to soma and set rl and cm
    cell.paint("soma", mech("ichan2").set
            ("gnatbar", 0.12).set
            ("gkfbar",  0.013).set
            ("gl",      0.00018).set
            ("el",      -65));
    cell.paint("soma", mech("ccanl").set
            ("catau",  10).set
            ("caiinf", 5.0e-6));
    cell.paint("soma", mech("borgka").set("gkabar", 0.00015));
    cell.paint("soma", mech("nca").set("gncabar",   0.0008));
    cell.paint("soma", mech("lca").set("glcabar",   0.00));
    cell.paint("soma", mech("gskch").set("gskbar",  0.00000));
    cell.paint("soma", mech("cagk").set("gkbar",    0.0002));
    cell.paint("soma", arb::membrane_capacitance{1.4/100.});
    cell.paint("soma", arb::axial_resistivity{100});

    // Set ion reversal potential method
    cell.default_parameters.reversal_potential_method["nca"] = "ccanlrev";
    cell.default_parameters.reversal_potential_method["lca"] = "ccanlrev";
    cell.default_parameters.reversal_potential_method["tca"] = "ccanlrev";

    // Add a spike detector and soma
    cell.place(arb::mlocation{0,0.5}, mech("exp2syn").set("tau1", 0.1).set("tau2", 1.661328).set("e", 0.0));
    cell.place(arb::mlocation{0,0.5}, mech("exp2syn").set("tau1", 0.1).set("tau2", 16.423438).set("e", 0.0));
    cell.place(arb::mlocation{0,0.5}, mech("exp2syn").set("tau1", 0.1).set("tau2", 16.423438).set("e", 0.0));
    cell.place(arb::mlocation{0,0.5}, arb::threshold_detector{10});

    return cell;
}

arb::cable_cell granule_cell(arb::sample_tree tree, arb::cell_gid_type gid) {
    using namespace arb::reg;
    using namespace arb::ls;
    using mech = arb::mechanism_desc;

    arb::label_dict dict;

    dict.set("soma", tagged(1));
    dict.set("dend", tagged(3));
    dict.set("granule_layer", intersect("dend", z_dist_from_root_gt(0)    , z_dist_from_root_le(6.68)));
    dict.set("inner_layer",   intersect("dend", z_dist_from_root_gt(6.68) , z_dist_from_root_le(20.04)));
    dict.set("middle_layer",  intersect("dend", z_dist_from_root_gt(20.04), z_dist_from_root_le(40.08)));
    dict.set("outer_layer",   intersect("dend", z_dist_from_root_gt(40.08), z_dist_from_root_le(66.9)));


    // Create the cell
    arb::morphology morpho(tree);
    arb::cable_cell cell(morpho, dict);


    // Draw 100 random points on the selected layer
    // Attach a synapse to one of the random locations
    mech exp2syn("exp2syn");
    exp2syn["tau1"] = 0.709067133592;
    exp2syn["tau2"] = 4.79049393295;
    exp2syn["e"] = 0.0;

    cell.place(arb::mlocation{0,0.5}, mech("exp2syn").set("tau1", 0.5).set("tau2", 6.441406).set("e", -75));
    cell.place(uniform("middle_layer",0, 0, gid), mech("exp2syn").set("tau1", 0.5).set("tau2", 1.777344).set("e", 0));
    cell.place(uniform("outer_layer",1, 1, gid), mech("exp2syn").set("tau1", 0.5).set("tau2", 4.109375).set("e", 0));
    cell.place(arb::mlocation{0,0.5}, arb::threshold_detector{10});

    // Add density mechanisms
    cell.default_parameters.discretization = arb::cv_policy_max_extent(5, arb::cv_policy_flag::single_root_cv);
    cell.default_parameters.reversal_potential_method["nca"] = "ccanlrev";
    cell.default_parameters.reversal_potential_method["lca"] = "ccanlrev";
    cell.default_parameters.reversal_potential_method["tca"] = "ccanlrev";

    cell.paint("soma", mech("ichan2").set
            ("gnatbar", 0.120*7.0).set
            ("gkfbar", 0.016*2.25).set
            ("gksbar", 0.006*1.0).set
            ("gl", 0.00004*7.2538).set
            ("el", -73));
    cell.paint("soma", mech("borgka").set("gkabar", 0.001*9.0));
    cell.paint("soma", mech("nca").set("gncabar",   0.001*0.735294118));
    cell.paint("soma", mech("lca").set("glcabar",   0.005*0.5));
    cell.paint("soma", mech("cat").set("gcatbar",   0.000037*2.0));
    cell.paint("soma", mech("gskch").set("gskbar",  0.001*1.0));
    cell.paint("soma", mech("cagk").set("gkbar",    0.0006*0.2));
    cell.paint("soma", mech("ccanl").set("catau",   10).set("caiinf",5.0e-6));
    cell.paint("soma", arb::membrane_capacitance{9.8/100});
    cell.paint("soma", arb::axial_resistivity{410});

    cell.paint("granule_layer", mech("ichan2").set
            ("gnatbar", 0.018*7.0).set
            ("gkfbar", 0.004).set
            ("gksbar", 0.006).set
            ("gl", 0.00004*7.2538).set
            ("el",-73));
    cell.paint("granule_layer", mech("nca").set("gncabar",   0.003*0.735294118));
    cell.paint("granule_layer", mech("lca").set("glcabar",   0.0075));
    cell.paint("granule_layer", mech("cat").set("gcatbar",   0.000075));
    cell.paint("granule_layer", mech("gskch").set("gskbar",  0.0004));
    cell.paint("granule_layer", mech("cagk").set("gkbar",    0.0006*0.2));
    cell.paint("granule_layer", mech("ccanl").set("catau",   10).set("caiinf",5.0e-6));
    cell.paint("granule_layer", arb::membrane_capacitance{9.8/100});
    cell.paint("granule_layer", arb::axial_resistivity{410});

    cell.paint("inner_layer", mech("ichan2").set
            ("gnatbar", 0.013*7.0).set
            ("gkfbar", 0.004).set
            ("gksbar", 0.006).set
            ("gl", 0.000063*7.2538).set
            ("el",-73));
    cell.paint("inner_layer", mech("nca").set("gncabar",   0.001   *0.735294118));
    cell.paint("inner_layer", mech("lca").set("glcabar",   0.0075));
    cell.paint("inner_layer", mech("cat").set("gcatbar",   0.00025));
    cell.paint("inner_layer", mech("gskch").set("gskbar",  0.0002));
    cell.paint("inner_layer", mech("cagk").set("gkbar",    0.001   *0.2));
    cell.paint("inner_layer", mech("ccanl").set("catau",   10).set("caiinf",5.0e-6));
    cell.paint("inner_layer", arb::membrane_capacitance{1.6 * 9.8/100});
    cell.paint("inner_layer", arb::axial_resistivity{410});

    cell.paint("middle_layer", mech("ichan2").set
            ("gnatbar", 0.008*7.0).set
            ("gkfbar", 0.001).set
            ("gksbar", 0.006).set
            ("gl", 0.000063*7.2538).set
            ("el",-73));
    cell.paint("middle_layer", mech("nca").set("gncabar",   0.001*0.735294118));
    cell.paint("middle_layer", mech("lca").set("glcabar",   0.0005));
    cell.paint("middle_layer", mech("cat").set("gcatbar",   0.0005));
    cell.paint("middle_layer", mech("gskch").set("gskbar",  0.0));
    cell.paint("middle_layer", mech("cagk").set("gkbar",    0.0024*0.2));
    cell.paint("middle_layer", mech("ccanl").set("catau",   10).set("caiinf",5.0e-6));
    cell.paint("middle_layer", arb::membrane_capacitance{1.6 * 9.8/100});
    cell.paint("middle_layer", arb::axial_resistivity{410});

    cell.paint("outer_layer", mech("ichan2").set
            ("gnatbar", 0.0*7.0).set
            ("gkfbar", 0.001).set
            ("gksbar", 0.008).set
            ("gl", 0.000063*7.2538).set
            ("el",-73));
    cell.paint("outer_layer", mech("nca").set("gncabar",   0.001*0.735294118));
    cell.paint("outer_layer", mech("lca").set("glcabar",   0.0));
    cell.paint("outer_layer", mech("cat").set("gcatbar",   0.001));
    cell.paint("outer_layer", mech("gskch").set("gskbar",  0.0));
    cell.paint("outer_layer", mech("cagk").set("gkbar",    0.0024*0.2));
    cell.paint("outer_layer", mech("ccanl").set("catau",   10).set("caiinf",5.0e-6));
    cell.paint("outer_layer", arb::membrane_capacitance{1.6 * 9.8/100});
    cell.paint("outer_layer", arb::axial_resistivity{410});

    return cell;

}

sim_params read_params(int argc, char** argv) {
    using sup::param_from_json;
    sim_params p;

    if (argc<2) {
        throw std::runtime_error("No input parameter file provided.");
    }
    if (argc>2) {
        throw std::runtime_error("More than command line one option not permitted.");
    }

    std::string fname = argv[1];
    std::cout << "Loading parameters from file: " << fname << "\n";
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    json << f;

    param_from_json(p.temp, "temp", json);
    param_from_json(p.v_init, "vinit", json);
    param_from_json(p.dt, "dt_arbor", json);
    param_from_json(p.run_time, "run_time", json);
    param_from_json(p.morph_file, "morph_file", json);
    param_from_json(p.network_file, "network_file", json);
    param_from_json(p.probe_gid, "probe_gid", json);

    for (auto it=json.begin(); it!=json.end(); ++it) {
        std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
    }
    std::cout << "\n";
    return p;
}

void write_trace_json(const arb::trace_data<double>& trace) {
    std::string path = "./voltages.json";

    nlohmann::json json;
    json["name"] = "ring demo";
    json["units"] = "mV";
    json["cell"] = "0.0";
    json["probe"] = "0";

    auto& jt = json["data"]["time"];
    auto& jy = json["data"]["voltage"];

    for (const auto& sample: trace) {
        jt.push_back(sample.t);
        jy.push_back(sample.v);
    }

    std::ofstream file(path);
    file << std::setw(1) << json << "\n";
}

arb::sample_tree read_swc(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("unable to open SWC file: "+path);

    return arb::swc_as_sample_tree(arb::parse_swc_file(f));
}

double distance(std::pair<double,double> a, std::pair<double,double> b) {
    return sqrt(pow((a.first-b.first),2) + pow((a.second-b.second),2));
}
