#include <iostream>
#include <fstream>

#include <sup/json_params.hpp>

struct sonata_params {

    std::string nodes_hdf5 = "nodes.h5";
    std::string nodes_csv = "node_types.csv";
    std::string edges_hdf5 = "edges.h5";
    std::string edges_csv = "edge_types.csv";

};

sonata_params read_options(int argc, char** argv) {
    using sup::param_from_json;

    sonata_params params;
    if (argc<4) {
        std::cout << "Using default parameters.\n";
        return params;
    }
    if (argc>4) {
        throw std::runtime_error("More than one command line option not permitted.");
    }

    std::string fname = argv[1];
    std::cout << "Loading parameters from file: " << fname << "\n";
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    json << f;

    param_from_json(params.nodes_hdf5, "nodes-hdf5", json);
    param_from_json(params.nodes_csv, "nodes-csv", json);
    param_from_json(params.edges_hdf5, "edges-hdf5", json);
    param_from_json(params.edges_csv, "edges-csv", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }
    return params;
}