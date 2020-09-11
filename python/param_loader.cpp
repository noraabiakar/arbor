#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <sup/json_params.hpp>

#include "error.hpp"

namespace pyarb {
using sup::find_and_remove_json;

arb::cable_cell_parameter_set load_cell_parameters(nlohmann::json& defaults_json) {
    arb::cable_cell_parameter_set defaults;

    defaults.init_membrane_potential = find_and_remove_json<double>("Vm", defaults_json);
    defaults.membrane_capacitance    = find_and_remove_json<double>("cm", defaults_json);
    defaults.axial_resistivity       = find_and_remove_json<double>("Ra", defaults_json);
    auto temp_c = find_and_remove_json<double>("celsius", defaults_json);
    if (temp_c) {
        defaults.temperature_K = temp_c.value() + 273.15;
    }

    if (auto ions_json = find_and_remove_json<nlohmann::json>("ions", defaults_json)) {
        auto ions_map = ions_json.value().get<std::unordered_map<std::string, nlohmann::json>>();
        for (auto& i: ions_map) {
            auto ion_name = i.first;
            auto ion_json = i.second;

            arb::cable_cell_ion_data ion_data;
            if (auto iconc = find_and_remove_json<double>("internal-concentration", ion_json)) {
                ion_data.init_int_concentration = iconc.value();
            }
            if (auto econc = find_and_remove_json<double>("external-concentration", ion_json)) {
                ion_data.init_ext_concentration = econc.value();
            }
            if (auto rev_pot = find_and_remove_json<double>("reversal-potential", ion_json)) {
                ion_data.init_reversal_potential = rev_pot.value();
            }
            defaults.ion_data.insert({ion_name, ion_data});

            if (auto method = find_and_remove_json<std::string>("method", ion_json)) {
                if (method.value() == "nernst") {
                    defaults.reversal_potential_method.insert({ion_name, "nernst/" + ion_name});
                } else if (method.value() != "constant") {
                    throw pyarb_error("method of ion \"" + ion_name + "\" can only be either constant or nernst");
                }
            }
        }
    }
    return defaults;
}

arb::mechanism_desc load_mechanism_desc(nlohmann::json& mech_json) {
    std::vector<arb::mechanism_desc> mech_vec;

    auto name = find_and_remove_json<std::string>("mechanism", mech_json);
    if (name) {
        auto mech = arb::mechanism_desc(name.value());
        auto params = find_and_remove_json<std::unordered_map<std::string, double>>("parameters", mech_json);
        if (params) {
            for (auto p: params.value()) {
                mech.set(p.first, p.second);
            }
        }
        return mech;
    }
    throw pyarb::pyarb_error("Mechanism not specified");
}

void check_defaults(const arb::cable_cell_parameter_set& defaults) {
    if(!defaults.temperature_K) throw pyarb_error("Default cell values don't include temperature");
    if(!defaults.init_membrane_potential) throw pyarb_error("Default cell values don't include initial membrane potential");
    if(!defaults.axial_resistivity) throw pyarb_error("Default cell values don't include axial_resistivity");
    if(!defaults.membrane_capacitance) throw pyarb_error("Default cell values don't include membrane_capacitance");

    std::vector<std::string> default_ions = {"ca", "na", "k"};

    for (auto ion:default_ions) {
        if(!defaults.ion_data.count(ion)) throw pyarb_error("Default cell values don't include " + ion + " default values");
        if(isnan(defaults.ion_data.at(ion).init_int_concentration))  throw pyarb_error("Default cell values don't include " + ion + "'s initial internal concentration");
        if(isnan(defaults.ion_data.at(ion).init_ext_concentration))  throw pyarb_error("Default cell values don't include " + ion + "'s initial external concentration");
        if(isnan(defaults.ion_data.at(ion).init_reversal_potential)) throw pyarb_error("Default cell values don't include " + ion + "'s initial reversal potential");
    }
}

arb::cable_cell_parameter_set overwrite_cable_parameters(const arb::cable_cell_parameter_set& base, const arb::cable_cell_parameter_set& overwrite) {
    arb::cable_cell_parameter_set merged = base;
    if (auto temp = overwrite.temperature_K) {
        merged.temperature_K = temp;
    }
    if (auto cm = overwrite.membrane_capacitance) {
        merged.membrane_capacitance = cm;
    }
    if (auto ra = overwrite.axial_resistivity) {
        merged.axial_resistivity = ra;
    }
    if (auto vm = overwrite.init_membrane_potential) {
        merged.init_membrane_potential = vm;
    }
    for (auto ion: overwrite.ion_data) {
        auto name = ion.first;
        auto data = ion.second;
        if (!isnan(data.init_reversal_potential)) {
            merged.ion_data[name].init_reversal_potential = data.init_reversal_potential;
        }
        if (!isnan(data.init_ext_concentration)) {
            merged.ion_data[name].init_ext_concentration = data.init_ext_concentration;
        }
        if (!isnan(data.init_int_concentration)) {
            merged.ion_data[name].init_int_concentration = data.init_int_concentration;
        }
    }
    for (auto ion: overwrite.reversal_potential_method) {
        auto name = ion.first;
        auto data = ion.second;
        merged.reversal_potential_method[name] = data;
    }
    return merged;
}

void register_param_loader(pybind11::module& m) {
    m.def("load_cell_default_parameters",
          [](std::string fname) {
              std::ifstream fid{fname};
              if (!fid.good()) {
                  throw pyarb_error(util::pprintf("can't open file '{}'", fname));
              }
              nlohmann::json defaults_json;
              defaults_json << fid;
              auto defaults = load_cell_parameters(defaults_json);
              try {
                  check_defaults(defaults);
              }
              catch (std::exception& e) {
                  throw pyarb_error("error loading parameter from \"" + fname + "\": " + std::string(e.what()));
              }
              return defaults;
          },
          "Load default cell parameters.");

    m.def("load_cell_global_parameters",
          [](std::string fname) {
              std::ifstream fid{fname};
              if (!fid.good()) {
                  throw pyarb_error(util::pprintf("can't open file '{}'", fname));
              }
              nlohmann::json cells_json;
              cells_json << fid;
              auto globals_json = find_and_remove_json<nlohmann::json>("global", cells_json);
              if (globals_json) {
                  return load_cell_parameters(globals_json.value());
              }
              return arb::cable_cell_parameter_set();
          },
          "Load global cell parameters.");

    m.def("load_cell_local_parameter_map",
          [](std::string fname) {
              std::unordered_map<std::string, arb::cable_cell_parameter_set> local_map;

              std::ifstream fid{fname};
              if (!fid.good()) {
                  throw pyarb_error(util::pprintf("can't open file '{}'", fname));
              }
              nlohmann::json cells_json;
              cells_json << fid;

              auto locals_json = find_and_remove_json<std::vector<nlohmann::json>>("local", cells_json);
              if (locals_json) {
                  for (auto l: locals_json.value()) {
                      auto region = find_and_remove_json<std::string>("region", l);
                      if (!region) {
                          throw pyarb_error("Local cell parameters do not include region label (in \"" + fname + "\")");
                      }
                      auto region_params = load_cell_parameters(l);

                      if(!region_params.reversal_potential_method.empty()) {
                          throw pyarb_error("Cannot implement local reversal potential methods (in \"" + fname + "\")");
                      }

                      local_map[region.value()] = region_params;
                  }
              }
              return local_map;
          },
          "Load local cell parameters.");

    m.def("load_cell_mechanism_map",
          [](std::string fname) {
              std::unordered_map<std::string, std::vector<arb::mechanism_desc>> mech_map;

              std::ifstream fid{fname};
              if (!fid.good()) {
                  throw pyarb_error(util::pprintf("can't open file '{}'", fname));
              }
              nlohmann::json cells_json;
              cells_json << fid;

              auto mech_json = find_and_remove_json<std::vector<nlohmann::json>>("mechanisms", cells_json);
              if (mech_json) {
                  for (auto m: mech_json.value()) {
                      auto region = find_and_remove_json<std::string>("region", m);
                      if (!region) {
                          throw pyarb_error("Mechanisms do not include region label (in \"" + fname + "\")");
                      }
                      try {
                          mech_map[region.value()].push_back(load_mechanism_desc(m));
                      }
                      catch (std::exception& e) {
                          throw pyarb_error("error loading mechanism for region " + region.value() + " in file \"" + fname + "\": " + std::string(e.what()));
                      }
                  }
              }
              return mech_map;
          },
          "Load region mechanism descriptions.");

    // arb::cable_cell_parameter_set
    pybind11::class_<arb::cable_cell_parameter_set> cable_cell_parameter_set(m, "cable_cell_parameter_set");

    // map of arb::cable_cell_parameter_set
    pybind11::class_<std::unordered_map<std::string, arb::cable_cell_parameter_set>> region_cable_cell_parameter_set_map(m, "region_cable_cell_parameter_set_map");

    // map of arb::cable_cell_parameter_set
    pybind11::class_<std::unordered_map<std::string, std::vector<arb::mechanism_desc>>> region_mechanism_map(m, "region_mechanism_map");
}

} //namespace pyarb