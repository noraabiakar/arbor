#include <exception>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <tclap/CmdLine.h>

#include "printer/cprinter.hpp"
//#include "printer/cudaprinter.hpp"
#include "printer/infoprinter.hpp"
#include "printer/simd.hpp"

#include "modccutil.hpp"
#include "module.hpp"
#include "parser.hpp"
#include "perfvisitor.hpp"

#include "io/bulkio.hpp"

using std::cout;
using std::cerr;

// Options and option parsing:

int report_error(const std::string& message) {
    cerr << red("error: ") << message << "\n";
    return 1;
}

int report_ice(const std::string& message) {
    cerr << red("internal compiler error: ") << message << "\n"
         << "\nPlease report this error to the modcc developers.\n";
    return 1;
}

enum class targetKind {
    cpu,
    gpu,
};

std::unordered_map<std::string, targetKind> targetKindMap = {
    {"cpu", targetKind::cpu},
    {"gpu", targetKind::gpu},
};

std::unordered_map<std::string, enum simd_spec::simd_abi> simdAbiMap = {
    {"none", simd_spec::none},
    {"avx",  simd_spec::avx},
    {"avx2", simd_spec::avx2},
    {"avx512", simd_spec::avx512},
    {"default", simd_spec::default_abi},
    {"native", simd_spec::native}
};

template <typename Map, typename V>
auto key_by_value(const Map& map, const V& v) -> decltype(map.begin()->first) {
    for (const auto& kv: map) {
        if (kv.second==v) return kv.first;
    }
    throw std::out_of_range("value not found in map");
}

struct Options {
    std::string outprefix;
    std::string modfile;
    std::string modulename;
    bool verbose = true;
    bool analysis = false;
    simd_spec simd = simd_spec::none;
    std::unordered_set<targetKind, enum_hash> targets;
};

// Helper for formatting tabulated output (option reporting).
struct table_prefix { std::string text; };
std::ostream& operator<<(std::ostream& out, const table_prefix& tb) {
    return out << cyan("| "+tb.text) << std::right << std::setw(58-tb.text.size());
};

std::ostream& operator<<(std::ostream& out, simd_spec simd) {
    std::stringstream s;
    s << key_by_value(simdAbiMap, simd.abi);
    if (simd.width!=0) {
        s << '/' << simd.width;
    }
    return out << s.str();
}

std::ostream& operator<<(std::ostream& out, const Options& opt) {
    static const char* noyes[2] = {"no", "yes"};
    static const std::string line_end = cyan(" |") + "\n";
    static const std::string tableline = cyan("."+std::string(60, '-')+".")+"\n";

    std::string targets;
    for (targetKind t: opt.targets) {
        targets += " "+key_by_value(targetKindMap, t);
    }

    return out <<
        tableline <<
        table_prefix{"file"} << opt.modfile << line_end <<
        table_prefix{"output"} << (opt.outprefix.empty()? "-": opt.outprefix) << line_end <<
        table_prefix{"verbose"} << noyes[opt.verbose] << line_end <<
        table_prefix{"targets"} << targets << line_end <<
        table_prefix{"simd"} << opt.simd << line_end <<
        table_prefix{"analysis"} << noyes[opt.analysis] << line_end <<
        tableline;
}

// Constraints for TCLAP arguments that are names for enumertion values.
struct MapConstraint: private std::vector<std::string>, public TCLAP::ValuesConstraint<std::string> {
    template <typename Map>
    MapConstraint(const Map& map):
        std::vector<std::string>(keys(map)),
        TCLAP::ValuesConstraint<std::string>(static_cast<std::vector<std::string>&>(*this)) {}

    template <typename Map>
    static std::vector<std::string> keys(const Map& map) {
        std::vector<std::string> ks;
        for (auto& kv: map) ks.push_back(kv.first);
        return ks;
    }
};

int main(int argc, char **argv) {
    Options opt;

    try {
        TCLAP::CmdLine cmd("modcc code generator for arbor", ' ', "0.1");

        TCLAP::UnlabeledValueArg<std::string>
            fin_arg("input_file", "the name of the .mod file to compile", true, "", "filename", cmd);

        TCLAP::ValueArg<std::string>
            fout_arg("o", "output", "prefix for output file names", false, "", "filename", cmd);

        MapConstraint targets_arg_constraint(targetKindMap);
        TCLAP::MultiArg<std::string>
            target_arg("t", "target", "build module for cpu or gpu back-end", false, &targets_arg_constraint, cmd);

        MapConstraint simd_arg_constraint(simdAbiMap);
        TCLAP::ValueArg<std::string>
            simd_arg("s", "simd", "use SIMD with given ABI", false, "", &simd_arg_constraint, cmd);

        std::vector<unsigned> simd_width_arg_constraint_ = {0u, 1u, 2u, 4u, 8u, 16u, 32u, 64u};
        TCLAP::ValuesConstraint<unsigned> simd_width_arg_constraint(simd_width_arg_constraint_);
        TCLAP::ValueArg<unsigned>
            simd_width_arg("S", "simd-width", "explicit SIMD vector width", false, 0, &simd_width_arg_constraint, cmd);

        TCLAP::SwitchArg verbose_arg("V","verbose","toggle verbose mode", cmd, false);

        TCLAP::SwitchArg analysis_arg("A","analyse","toggle analysis mode", cmd, false);

        TCLAP::ValueArg<std::string>
            module_arg("m", "module", "module name to use (default taken from input .mod file)", false, "", "module", cmd);

        cmd.parse(argc, argv);

        opt.outprefix = fout_arg.getValue();
        opt.modfile = fin_arg.getValue();
        opt.modulename = module_arg.getValue();
        opt.verbose = verbose_arg.getValue();
        opt.analysis = analysis_arg.getValue();

        if (!simd_arg.getValue().empty()) {
            opt.simd = simd_spec(simdAbiMap.at(simd_arg.getValue()), simd_width_arg.getValue());
        }

        for (auto& target: target_arg.getValue()) {
            opt.targets.insert(targetKindMap.at(target));
        }
    }
    catch(TCLAP::ArgException &e) {
        return report_error(e.error()+" for argument "+to_string(e.argId()));
    }

    try {
        auto emit_header = [&opt](const char* h) {
            if (opt.verbose) {
                cout << green("[") << h << green("]") << "\n";
            }
        };

        if (opt.verbose) {
            cout << opt;
        }

        // Load module file and initialize Module object.

        Module m(io::read_all(opt.modfile), opt.modfile);

        if (m.empty()) {
            return report_error("empty file: "+opt.modfile);
        }

        if (!opt.modulename.empty()) {
            m.module_name(opt.modulename);
        }

        // Perform parsing and semantic analysis passes.

        emit_header("parsing");
        Parser p(m, false);
        if (!p.parse()) {
            // Parser::parse() writes its own errors to stderr.
            return 1;
        }

        emit_header("semantic analysis");
        m.semantic();
        if (m.has_warning()) {
            cerr << m.warning_string() << "\n";
        }
        if (m.has_error()) {
            return report_error(m.error_string());
        }

        // Generate backend-specific sources for each backend provided.

        emit_header("code generation");

        // If no output prefix given, use the module name.
        std::string prefix = opt.outprefix.empty()? m.module_name(): opt.outprefix;

        io::write_all(build_info_header(m, "arb"), prefix+".hpp");

        for (targetKind target: opt.targets) {
            std::string outfile = prefix;
            switch (target) {
            case targetKind::gpu:
                // TODO: make cudaprinter work with new internal mechanism API
                outfile += "_gpu";
                {
#if 0
                    CUDAPrinter printer(m);
                    io::write_all(printer.interface_text(), outfile+".hpp");
                    io::write_all(printer.impl_header_text(), outfile+"_impl.hpp");
                    io::write_all(printer.impl_text(), outfile+"_impl.cu");
#else
                    throw std::logic_error("CUDA printer temporariliy disabled");
#endif
                }
                break;
            case targetKind::cpu:
                outfile += "_cpu.cpp";
                io::write_all(emit_cpp_source(m, "arb", opt.simd), outfile);
                break;
            }
        }

        // Optional analysis report.

        if (opt.analysis) {
            cout << green("performance analysis\n");
            for (auto &symbol: m.symbols()) {
                if (auto method = symbol.second->is_api_method()) {
                    cout << white("-------------------------\n");
                    cout << yellow("method " + method->name()) << "\n";
                    cout << white("-------------------------\n");

                    FlopVisitor flops;
                    method->accept(&flops);
                    cout << white("FLOPS\n") << flops.print() << "\n";

                    MemOpVisitor memops;
                    method->accept(&memops);
                    cout << white("MEMOPS\n") << memops.print() << "\n";
                }
            }
        }
    }
    catch(compiler_exception& e) {
        return report_ice(e.what()+std::string(" @ ")+to_string(e.location()));
    }
    catch(std::exception& e) {
        return report_ice(e.what());
    }
    catch(...) {
        return report_ice("");
    }

    return 0;
}
