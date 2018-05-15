#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>

#include "expression.hpp"
#include "io/ostream_wrappers.hpp"
#include "io/prefixbuf.hpp"
#include "printer/cexpr_emit.hpp"
#include "printer/cprinter.hpp"
#include "printer/printerutil.hpp"

using io::indent;
using io::popindent;
using io::quote;


void emit_procedure_proto(std::ostream&, ProcedureExpression*, const std::string& qualified = "");
void emit_simd_procedure_proto(std::ostream&, ProcedureExpression*, const std::string& qualified = "");

void emit_api_body(std::ostream&, APIMethod*);
void emit_simd_api_body(std::ostream&, APIMethod*, moduleKind);

struct cprint {
    Expression* expr_;
    explicit cprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const cprint& w) {
        CPrinter printer(out);
        return w.expr_->accept(&printer), out;
    }
};

struct simdprint {
    Expression* expr_;
    explicit simdprint(Expression* expr): expr_(expr) {}

    friend std::ostream& operator<<(std::ostream& out, const simdprint& w) {
        SimdPrinter printer(out);
        return w.expr_->accept(&printer), out;
    }
};

static std::string ion_state_field(std::string ion_name) {
    return "ion_"+ion_name+"_";
}

static std::string ion_state_index(std::string ion_name) {
    return "ion_"+ion_name+"_index_";
}

std::string emit_cpp_source(const Module& module_, const std::string& ns, simd_spec simd) {
    std::string name = module_.module_name();
    std::string class_name = "mechanism_cpu_"+name;
    auto ns_components = namespace_components(ns);

    NetReceiveExpression* net_receive = find_net_receive(module_);
    APIMethod* init_api = find_api_method(module_, "nrn_init");
    APIMethod* state_api = find_api_method(module_, "nrn_state");
    APIMethod* current_api = find_api_method(module_, "nrn_current");
    APIMethod* write_ions_api = find_api_method(module_, "write_ions");

    bool with_simd = simd.abi!=simd_spec::none;

    // init_api, state_api, current_api methods are mandatory:

    assert_has_scope(init_api, "nrn_init");
    assert_has_scope(state_api, "nrn_state");
    assert_has_scope(current_api, "nrn_current");

    auto vars = local_module_variables(module_);
    auto ion_deps = module_.ion_deps();
    std::string fingerprint = "<placeholder>";

    io::pfxstringstream out;

    out <<
        "#include <cmath>\n"
        "#include <cstddef>\n"
        "#include <memory>\n"
        "#include <" << arb_header_prefix() << "backends/multicore/mechanism.hpp>\n"
        "#include <" << arb_header_prefix() << "math.hpp>\n";

    if (with_simd) {
        out << "#include <" << arb_header_prefix() << "simd/simd.hpp>\n";
    }

    out <<
        "\n" << namespace_declaration_open(ns_components) <<
        "\n"
        "using backend = ::arb::multicore::backend;\n"
        "using base = ::arb::multicore::mechanism;\n"
        "using value_type = base::value_type;\n"
        "using size_type = base::size_type;\n"
        "using index_type = base::index_type;\n"
        "using ::std::abs;\n"
        "using ::std::cos;\n"
        "using ::std::exp;\n"
        "using ::arb::math::exprelr;\n"
        "using ::std::log;\n"
        "using ::arb::math::max;\n"
        "using ::arb::math::min;\n"
        "using ::std::pow;\n"
        "using ::std::sin;\n"
        "\n";

    if (with_simd) {
        out <<
            "namespace S = ::arb::simd;\n"
            "static constexpr unsigned simd_width_ = ";

        if (!simd.width) {
            out << "S::simd_abi::native_width<fvm_value_type>::value;\n";
        }
        else {
            out << simd.width << ";\n";
        }

        std::string abi = "S::simd_abi::";
        switch (simd.abi) {
        case simd_spec::avx:    abi += "avx";    break;
        case simd_spec::avx2:   abi += "avx2";   break;
        case simd_spec::avx512: abi += "avx512"; break;
        case simd_spec::native: abi += "native"; break;
        default:
            abi += "default_abi"; break;
        }

        out <<
            "using simd_value = S::simd<fvm_value_type, simd_width_, " << abi << ">;\n"
            "using simd_index = S::simd<fvm_index_type, simd_width_, " << abi << ">;\n"
            "\n";
    }

    out <<
        "class " << class_name << ": public base {\n"
        "public:\n" << indent <<
        "const mechanism_fingerprint& fingerprint() const override {\n" << indent <<
        "static mechanism_fingerprint hash = " << quote(fingerprint) << ";\n"
        "return hash;\n" << popindent <<
        "}\n"
        "std::string internal_name() const override { return " << quote(name) << "; }\n"
        "mechanismKind kind() const override { return " << module_kind_str(module_) << "; }\n"
        "mechanism_ptr clone() const override { return mechanism_ptr(new " << class_name << "()); }\n"
        "\n"
        "void nrn_init() override;\n"
        "void nrn_state() override;\n"
        "void nrn_current() override;\n"
        "void write_ions() override;\n";

    net_receive && out <<
        "void deliver_events(deliverable_event_stream::state events) override;\n"
        "void net_receive(int i_, value_type weight);\n";

    out <<
        "\n" << popindent <<
        "protected:\n" << indent <<
        "using ionKind = ::arb::ionKind;\n\n"
        "std::size_t object_sizeof() const override { return sizeof(*this); }\n";

    io::separator sep("\n", ",\n");
    if (!vars.scalars.empty()) {
        out <<
            "mechanism_global_table global_table() override {\n" << indent <<
            "return {" << indent;

        for (const auto& scalar: vars.scalars) {
            auto memb = scalar->name();
            out << sep << "{" << quote(memb) << ", &" << memb << "}";
        }
        out << popindent << "\n};\n" << popindent << "}\n";
    }

    if (!vars.arrays.empty()) {
        out <<
            "mechanism_field_table field_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& array: vars.arrays) {
            auto memb = array->name();
            out << sep << "{" << quote(memb) << ", &" << memb << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        out <<
            "mechanism_field_default_table field_default_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& array: vars.arrays) {
            auto memb = array->name();
            auto dflt = array->value();
            if (!std::isnan(dflt)) {
                out << sep << "{" << quote(memb) << ", " << as_c_double(dflt) << "}";
            }
        }
        out << popindent << "\n};" << popindent << "\n}\n";

    }

    if (!ion_deps.empty()) {
        out <<
            "mechanism_ion_state_table ion_state_table() override {\n" << indent <<
            "return {" << indent;

        sep.reset();
        for (const auto& dep: ion_deps) {
            out << sep << "{ionKind::" << dep.name << ", &" << ion_state_field(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";

        sep.reset();
        out << "mechanism_ion_index_table ion_index_table() override {\n" << indent << "return {" << indent;
        for (const auto& dep: ion_deps) {
            out << sep << "{ionKind::" << dep.name << ", &" << ion_state_index(dep.name) << "}";
        }
        out << popindent << "\n};" << popindent << "\n}\n";
    }

    out << popindent << "\n"
        "private:\n" << indent;

    for (const auto& scalar: vars.scalars) {
        out << "value_type " << scalar->name() <<  " = " << as_c_double(scalar->value()) << ";\n";
    }
    for (const auto& array: vars.arrays) {
        out << "value_type* " << array->name() << ";\n";
    }
    for (const auto& dep: ion_deps) {
        out << "ion_state_view " << ion_state_field(dep.name) << ";\n";
        out << "iarray " << ion_state_index(dep.name) << ";\n";
    }

    for (auto proc: normal_procedures(module_)) {
        emit_procedure_proto(out, proc);
        out << ";\n";
        if (with_simd) {
            emit_simd_procedure_proto(out, proc);
            out << ";\n";
        }
    }

    out << popindent <<
        "};\n\n"
        "template <typename B> ::arb::concrete_mech_ptr<B> make_mechanism_" <<name << "();\n"
        "template <> ::arb::concrete_mech_ptr<backend> make_mechanism_" << name << "<backend>() {\n" << indent <<
        "return ::arb::concrete_mech_ptr<backend>(new " << class_name << "());\n" << popindent <<
        "}\n\n";

    // Nrn methods:

    net_receive && out <<
        "void " << class_name << "::deliver_events(deliverable_event_stream::state events) {\n" << indent <<
        "auto ncell = events.n_streams();\n"
        "for (size_type c = 0; c<ncell; ++c) {\n" << indent <<
        "auto begin = events.begin_marked(c);\n"
        "auto end = events.end_marked(c);\n"
        "for (auto p = begin; p<end; ++p) {\n" << indent <<
        "if (p->mech_id==mechanism_id_) net_receive(p->mech_index, p->weight);\n" << popindent <<
        "}\n" << popindent <<
        "}\n" << popindent <<
        "}\n"
        "\n"
        "void " << class_name << "::net_receive(int i_, value_type weight) {\n" << indent <<
        cprint(net_receive->body()) << popindent <<
        "}\n\n";

    auto emit_body = [&](APIMethod *p) {
        if (with_simd) {
            emit_simd_api_body(out, p, module_.kind());
        }
        else {
            emit_api_body(out, p);
        }
    };

    out << "void " << class_name << "::nrn_init() {\n" << indent;
    emit_body(init_api);
    out << popindent << "}\n\n";

    out << "void " << class_name << "::nrn_state() {\n" << indent;
    emit_body(state_api);
    out << popindent << "}\n\n";

    out << "void " << class_name << "::nrn_current() {\n" << indent;
    emit_body(current_api);
    out << popindent << "}\n\n";

    out << "void " << class_name << "::write_ions() {\n" << indent;
    emit_body(write_ions_api);
    out << popindent << "}\n\n";

    // Mechanism procedures

    for (auto proc: normal_procedures(module_)) {
        emit_procedure_proto(out, proc, class_name);
        out <<
            " {\n" << indent <<
            cprint(proc->body()) << popindent <<
            "}\n\n";

        if (with_simd) {
            emit_simd_procedure_proto(out, proc, class_name);
            out <<
                " {\n" << indent <<
                simdprint(proc->body()) << popindent <<
                "}\n\n";
        }
    }

    out << namespace_declaration_close(ns_components);
    return out.str();
}

struct indexed_variable_info {
    std::string data_var;
    std::string index_var;
};

indexed_variable_info decode_indexed_variable(IndexedVariable* sym) {
    std::string data_var, ion_pfx;
    std::string index_var = "constraint_index_";

    if (sym->is_ion()) {
        ion_pfx = "ion_"+to_string(sym->ion_channel())+"_";
        index_var = ion_pfx+"index_";
    }

    switch (sym->data_source()) {
    case sourceKind::voltage:
        data_var="vec_v_";
        break;
    case sourceKind::current:
        data_var="vec_i_";
        break;
    case sourceKind::dt:
        data_var="vec_dt_";
        break;
    case sourceKind::ion_current:
        data_var=ion_pfx+".current_density";
        break;
    case sourceKind::ion_revpot:
        data_var=ion_pfx+".reversal_potential";
        break;
    case sourceKind::ion_iconc:
        data_var=ion_pfx+".internal_concentration";
        break;
    case sourceKind::ion_econc:
        data_var=ion_pfx+".external_concentration";
        break;
    default:
        throw compiler_exception("unrecognized indexed data source", sym->location());
    }

    return {data_var, index_var};
}

// Scalar printing:

void CPrinter::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void CPrinter::visit(LocalVariable* sym) {
    out_ << sym->name();
}

void CPrinter::visit(VariableExpression *sym) {
    out_ << sym->name() << (sym->is_range()? "[i_]": "");
}

void CPrinter::visit(IndexedVariable *sym) {
    indexed_variable_info v = decode_indexed_variable(sym);
    out_ << v.data_var << "[" << v.index_var << "[i_]]";
}

void CPrinter::visit(CallExpression* e) {
    out_ << e->name() << "(i_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}

void CPrinter::visit(BlockExpression* block) {
    // Only include local declarations in outer-most block.
    if (!block->is_nested()) {
        auto locals = pure_locals(block->scope());
        if (!locals.empty()) {
            out_ << "value_type ";
            io::separator sep(", ");
            for (auto local: locals) {
                out_ << sep << local->name();
            }
            out_ << ";\n";
        }
    }

    for (auto& stmt: block->statements()) {
        if (!stmt->is_local_declaration()) {
            stmt->accept(this);
            out_ << (stmt->is_if()? "": ";\n");
        }
    }
}

void emit_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& qualified) {
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name() << "(int i_";
    for (auto& arg: e->args()) {
        out << ", value_type " << arg->is_argument()->name();
    }
    out << ")";
}

void emit_state_read(std::ostream& out, LocalVariable* local) {
    out << "value_type " << cprint(local) << " = ";

    if (local->is_read()) {
        out << cprint(local->external_variable()) << ";\n";
    }
    else {
        out << "0;\n";
    }
}

void emit_state_update(std::ostream& out, Symbol* from, IndexedVariable* external) {
    if (!external->is_write()) return;

    const char* op = external->op()==tok::plus? " += ": " -= ";
    out << cprint(external) << op << from->name() << ";\n";
}

void emit_api_body(std::ostream& out, APIMethod* method) {
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());

    if (!body->statements().empty()) {
        out <<
            "int n_ = width_;\n"
            "for (int i_ = 0; i_ < n_; ++i_) {\n" << indent;

        for (auto& sym: indexed_vars) {
            emit_state_read(out, sym);
        }
        out << cprint(body);

        for (auto& sym: indexed_vars) {
            emit_state_update(out, sym, sym->external_variable());
        }
        out << popindent << "}\n";
    }
}

// SIMD printing:

std::string index_i_name(const std::string& index_var) {
    return index_var+"i_";
}

std::string index_constraint_name(const std::string& index_var) {
    return index_var+"category_";
}


void SimdPrinter::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void SimdPrinter::visit(LocalVariable* sym) {
    out_ << sym->name();
}

void SimdPrinter::visit(VariableExpression *sym) {
    if (sym->is_range()) {
        out_ << "simd_value(" << sym->name() << "+i_)";
    }
    else {
        out_ << sym->name();
    }
}

void SimdPrinter::visit(AssignmentExpression* e) {
    if (!e->lhs() || !e->lhs()->is_identifier() || !e->lhs()->is_identifier()->symbol()) {
        throw compiler_exception("Expect symbol on lhs of assignment: "+e->to_string());
    }

    Symbol* lhs = e->lhs()->is_identifier()->symbol();

    if (lhs->is_variable() && lhs->is_variable()->is_range()) {
        out_ << "simd_value(";
        e->rhs()->accept(this);
        out_ << ").copy_to(" << lhs->name() << "+i_)";
    }
    else {
        out_ << lhs->name() << " = ";
        e->rhs()->accept(this);
    }
}

void SimdPrinter::visit(IndexedVariable *sym) {
    indexed_variable_info v = decode_indexed_variable(sym);
    out_ << "S::indirect(" << v.data_var
         << ", " << index_i_name(v.index_var)
         << ", " << index_constraint_name(v.index_var) << ")";
}

void SimdPrinter::visit(CallExpression* e) {
    out_ << e->name() << "(i_";
    for (auto& arg: e->args()) {
        out_ << ", ";
        arg->accept(this);
    }
    out_ << ")";
}

void SimdPrinter::visit(BlockExpression* block) {
    // Only include local declarations in outer-most block.
    if (!block->is_nested()) {
        auto locals = pure_locals(block->scope());
        if (!locals.empty()) {
            out_ << "simd_value ";
            io::separator sep(", ");
            for (auto local: locals) {
                out_ << sep << local->name();
            }
            out_ << ";\n";
        }
    }

    for (auto& stmt: block->statements()) {
        if (stmt->is_if()) {
            throw compiler_exception("Conditionals not yet supported in SIMD printer: "+stmt->to_string());
        }
        if (!stmt->is_local_declaration()) {
            stmt->accept(this);
            out_ << ";\n";
        }
    }
}

void emit_simd_procedure_proto(std::ostream& out, ProcedureExpression* e, const std::string& qualified) {
    out << "void " << qualified << (qualified.empty()? "": "::") << e->name() << "(int i_";
    for (auto& arg: e->args()) {
        out << ", const simd_value& " << arg->is_argument()->name();
    }
    out << ")";
}

void emit_simd_state_read(std::ostream& out, LocalVariable* local) {
    out << "simd_value " << local->name();

    if (local->is_read()) {
        out << "(" << simdprint(local->external_variable()) << ");\n";
    }
    else {
        out << " = 0;\n";
    }
}

void emit_simd_state_update(std::ostream& out, Symbol* from, IndexedVariable* external) {
    if (!external->is_write()) return;

    const char* op = external->op()==tok::plus? " += ": " -= ";
    out << simdprint(external) << op << from->name() << ";\n";
}


void emit_simd_api_body(std::ostream& out, APIMethod* method, moduleKind module_kind) {
    auto body = method->body();
    auto indexed_vars = indexed_locals(method->scope());

    std::unordered_set<std::string> indices;
    for (auto& sym: indexed_vars) {
        indices.insert(decode_indexed_variable(sym->external_variable()).index_var);
    }

    // Note: expect to make index constraints non-constant for point mechanisms as
    // an optimization in the near future.

    // Another note: (TODO) can't actually use index_constraint::independent
    // for density mechanisms because of collisions in the padded part of
    // the indices. Work-arounds exist, but not yet implemented.


    std::string common_constraint = module_kind==moduleKind::density?
        //"S::index_constraint::independent":
        "S::index_constraint::none":
        "S::index_constraint::none";

    out << "int n_ = width_;\n\n";

    for (auto& index: indices) {
        out << "simd_index " << index_i_name(index) << ";\n";
        out << "index_constraint " << index_constraint_name(index) << ";\n\n";

        out << "int n_" << index_constraint_name(index) << "0 = " << index << ".compartment_sizes[0];\n";
        out << "int n_" << index_constraint_name(index) << "1 = " << index << ".compartment_sizes[1] + n_"
            << index_constraint_name(index) << "0 > n_ ? n_ : " << index << ".compartment_sizes[1] + n_"
            << index_constraint_name(index) << "0;\n";
        out << "int n_" << index_constraint_name(index) << "2 = " << index << ".compartment_sizes[2] + n_"
            << index_constraint_name(index) << "1 > n_ ? n_ : " << index << ".compartment_sizes[2] + n_"
            << index_constraint_name(index) << "1;\n";
        out << "int n_" << index_constraint_name(index) << "3 = " << index << ".compartment_sizes[3] + n_"
            << index_constraint_name(index) << "2 > n_ ? n_ : " << index << ".compartment_sizes[3] + n_"
            << index_constraint_name(index) << "2;\n\n";

    }

    if (!body->statements().empty()) {
        out <<
            "for (int i_ = 0; i_ < n_; i_ += simd_width_) {\n" << indent;

        for (auto& index: indices) {
            out << index_constraint_name(index) << " = i_ < n_" << index_constraint_name(index) << "0 ? index_constraint::contiguous : \n"
                << "\t(i_ < n_" << index_constraint_name(index) << "1 ? index_constraint::independent : \n"
                << "\t(i_ < n_" << index_constraint_name(index) << "2 ? index_constraint::none : \n"
                << "\tindex_constraint::constant));\n";
            out << index_i_name(index) << ".copy_from(" << index << ".full_index_compartments.data()+i_);\n";
        }

        for (auto& sym: indexed_vars) {
            emit_simd_state_read(out, sym);
        }

        out << simdprint(body);

        for (auto& sym: indexed_vars) {
            emit_simd_state_update(out, sym, sym->external_variable());
        }
        out << popindent << "}\n";
    }
}
