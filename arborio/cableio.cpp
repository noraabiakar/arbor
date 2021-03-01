#include <iostream>
#include <numeric>
#include <functional>
#include <sstream>
#include <numeric>

#include <arbor/cable_cell.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/any_visitor.hpp>
#include <arbor/util/pp_util.hpp>

#include <arborio/cableio.hpp>

namespace arborio{
using namespace arb;

// Errors
cableio_parse_error::cableio_parse_error(const std::string& msg, const arb::src_location& loc):
    arb::arbor_exception(msg+" at :"+std::to_string(loc.line)+":"+std::to_string(loc.column))
{}
cableio_unexpected_symbol::cableio_unexpected_symbol(const std::string& sym, const arb::src_location& loc):
    cableio_parse_error("Unexpected symbol "+sym, loc) {}

// Helpers
inline symbol operator "" _symbol(const char* chars, size_t size) {
    return {chars};
}

// Write s-expr

// Helper functions that convert types into s-expressions.
template <typename U, typename V>
s_expr mksexp(const std::pair<U, V>& p) {
    return slist(p.first, p.second);
}

// Forward declarations
s_expr mksexp(const mechanism_desc&);
s_expr mksexp(const msegment&);

struct as_s_expr {
    template <typename T>
    s_expr operator()(const T& v) const {
        return mksexp(v);
    }
};

// Defaultable and paintable
s_expr mksexp(const init_membrane_potential& p) {
    return slist("membrane-potential"_symbol, p.value);
}
s_expr mksexp(const axial_resistivity& r) {
    return slist("axial-resistivity"_symbol, r.value);
}
s_expr mksexp(const temperature_K& t) {
    return slist("temperature-kelvin"_symbol, t.value);
}
s_expr mksexp(const membrane_capacitance& c) {
    return slist("membrane-capacitance"_symbol, c.value);
}
s_expr mksexp(const init_int_concentration& c) {
    return slist("ion-internal-concentration"_symbol, c.ion, c.value);
}
s_expr mksexp(const init_ext_concentration& c) {
    return slist("ion-external-concentration"_symbol, c.ion, c.value);
}
s_expr mksexp(const init_reversal_potential& e) {
    return slist("ion-reversal-potential"_symbol, e.ion, e.value);
}
s_expr mksexp(const initial_ion_data& s) {
    std::vector<s_expr> ion_data;
    if (auto iconc = s.initial.init_int_concentration) {
        ion_data.push_back(slist("ion-internal-concentration"_symbol, s.ion, iconc.value()));
    }
    if (auto econc = s.initial.init_ext_concentration) {
        ion_data.push_back(slist("ion-external-concentration"_symbol, s.ion, econc.value()));
    }
    if (auto revpot = s.initial.init_reversal_potential) {
        ion_data.push_back(slist("ion-reversal-potential"_symbol, s.ion, revpot.value()));
    }
    return slist_range(ion_data);
}

// Defaultable
s_expr mksexp(const ion_reversal_potential_method& e) {
    return slist("ion-reversal-potential-method"_symbol, e.ion, mksexp(e.method));
}
s_expr mksexp(const cv_policy& c) {
    return s_expr();
}

// Paintable
s_expr mksexp(const mechanism_desc& d) {
    std::vector<s_expr> mech;
    mech.push_back(d.name());
    for (const auto& p: d.values()) {
        mech.push_back(slist("param"_symbol, p.first, p.second));
    }
    return s_expr{"mechanism"_symbol, slist_range(mech)};
}

// Placeable
s_expr mksexp(const i_clamp& c) {
    return slist("current-clamp"_symbol, c.amplitude, c.delay, c.duration);
}
s_expr mksexp(const threshold_detector& d) {
    return slist("threshold-detector"_symbol, d.threshold);
}
s_expr mksexp(const gap_junction_site& s) {
    return slist("gap-junction-site"_symbol);
}

// Decor
s_expr mksexp(const decor& d) {
    auto round_trip = [] (auto& x) {
      std::stringstream s;
      s << x;
      return parse_s_expr(s.str());
    };
    s_expr lst = slist();
    for (const auto& p: d.defaults().serialize()) {
        lst = {std::visit([&](auto& x) {return slist("default"_symbol, mksexp(x));}, p), std::move(lst)};
    }
    for (const auto& p: d.paintings()) {
        lst = {std::visit([&](auto& x) {return slist("paint"_symbol, round_trip(p.first), mksexp(x));}, p.second), std::move(lst)};
    }
    for (const auto& p: d.placements()) {
        lst = {std::visit([&](auto& x) {return slist("place"_symbol, round_trip(p.first), mksexp(x));}, p.second), std::move(lst)};
    }
    return {"decorations"_symbol, std::move(lst)};
}

// Label dictionary
s_expr mksexp(const label_dict& dict) {
    auto round_trip = [] (auto& x) {
      std::stringstream s;
      s << x;
      return parse_s_expr(s.str());
    };

    auto defs = slist();
    for (auto& r: dict.regions()) {
        defs = s_expr(slist("region-def"_symbol, r.first, round_trip(r.second)), std::move(defs));
    }
    for (auto& r: dict.locsets()) {
        defs = s_expr(slist("locset-def"_symbol, r.first, round_trip(r.second)), std::move(defs));
    }

    return {"label-dict"_symbol, std::move(defs)};
}

// Morphology
s_expr mksexp(const mpoint& p) {
    return slist("point"_symbol, p.x, p.y, p.z, p.radius);
}
s_expr mksexp(const msegment& seg) {
    return slist("segment"_symbol, (int)seg.id, mksexp(seg.prox), mksexp(seg.dist), seg.tag);
}
s_expr mksexp(const morphology& morph) {
    // s-expression representation of branch i in the morphology
    auto make_branch = [&morph] (int i) {
        std::vector<s_expr> segments;
        for (const auto& s: morph.branch_segments(i)) {
            segments.push_back(mksexp(s));
        }
        return s_expr{"branch"_symbol, {i, {(int)morph.branch_parent(i), slist_range(segments)}}};
    };
    std::vector<s_expr> branches;
    for (msize_t i = 0; i < morph.num_branches(); ++i) {
      branches.push_back(make_branch(i));
    }
    return s_expr{"morphology"_symbol, slist_range(branches)};
}

std::ostream& write_s_expr(std::ostream& o, const label_dict& dict) {
    return o << mksexp(dict);
}
std::ostream& write_s_expr(std::ostream& o, const decor& decorations) {
    return o << mksexp(decorations);
}
std::ostream& write_s_expr(std::ostream& o, const morphology& morphology) {
    return o << mksexp(morphology);
}
std::ostream& write_s_expr(std::ostream& o, const cable_cell& c) {
    return o << s_expr{"cable-cell"_symbol, slist(mksexp(c.morphology()), mksexp(c.labels()), mksexp(c.decorations()))};
}

// Read s-expr

// Anonymous namespace containing helper functions and types
namespace {
// Test whether a value wrapped in std::any can be converted to a target type
template <typename T>
bool match(const std::type_info& info) {
    return info == typeid(T);
}
template <>
bool match<double>(const std::type_info& info) {
    return info == typeid(double) || info == typeid(int);
}

// Convert a value wrapped in a std::any to target type.
template <typename T>
T eval_cast(std::any arg) {
    return std::move(std::any_cast<T&>(arg));
}
template <>
double eval_cast<double>(std::any arg) {
    if (arg.type()==typeid(int)) return std::any_cast<int>(arg);
    return std::any_cast<double>(arg);
}

// std::any to defaultable, paintable and placeable
std::optional<defaultable> any_to_defaultable(const std::any& any) {
    if (match<init_membrane_potential>(any.type())) return std::any_cast<init_membrane_potential>(any);
    if (match<axial_resistivity>(any.type()))       return std::any_cast<axial_resistivity>(any);
    if (match<temperature_K>(any.type()))           return std::any_cast<temperature_K>(any);
    if (match<init_int_concentration>(any.type()))  return std::any_cast<init_int_concentration>(any);
    if (match<init_ext_concentration>(any.type()))  return std::any_cast<init_ext_concentration>(any);
    if (match<init_reversal_potential>(any.type())) return std::any_cast<init_reversal_potential>(any);
    if (match<ion_reversal_potential_method>(any.type())) return std::any_cast<ion_reversal_potential_method>(any);
    return std::nullopt;
}

std::optional<paintable> any_to_paintable(const std::any& any) {
    if (match<init_membrane_potential>(any.type())) return std::any_cast<init_membrane_potential>(any);
    if (match<axial_resistivity>(any.type()))       return std::any_cast<axial_resistivity>(any);
    if (match<temperature_K>(any.type()))           return std::any_cast<temperature_K>(any);
    if (match<init_int_concentration>(any.type()))  return std::any_cast<init_int_concentration>(any);
    if (match<init_ext_concentration>(any.type()))  return std::any_cast<init_ext_concentration>(any);
    if (match<init_reversal_potential>(any.type())) return std::any_cast<init_reversal_potential>(any);
    if (match<mechanism_desc>(any.type())) return std::any_cast<mechanism_desc>(any);
    return std::nullopt;
}

std::optional<placeable> any_to_placeable(const std::any& any) {
    if (match<mechanism_desc>(any.type())) return std::any_cast<mechanism_desc>(any);
    if (match<i_clamp>(any.type()))        return std::any_cast<i_clamp>(any);
    if (match<threshold_detector>(any.type())) return std::any_cast<threshold_detector>(any);
    if (match<gap_junction_site>(any.type()))  return std::any_cast<gap_junction_site>(any);
    return std::nullopt;
}

// Define makers for defaultables, paintables and placeables
#define ARBIO_DEFINE_SINGLE_ARG(name) arb::name make_##name(double val) { return arb::name{val}; }
#define ARBIO_DEFINE_DOUBLE_ARG(name) arb::name make_##name(const std::string& ion, double val) { return arb::name{ion, val};}

ARB_PP_FOREACH(ARBIO_DEFINE_SINGLE_ARG, init_membrane_potential, temperature_K, axial_resistivity, membrane_capacitance, threshold_detector)
ARB_PP_FOREACH(ARBIO_DEFINE_DOUBLE_ARG, init_int_concentration, init_ext_concentration, init_reversal_potential)

arb::i_clamp make_i_clamp(double delay, double duration, double amplitude) {
    return arb::i_clamp(delay, duration, amplitude);
}
arb::gap_junction_site make_gap_junction_site() {
    return arb::gap_junction_site{};
}
arb::ion_reversal_potential_method make_ion_reversal_potential_method(const std::string& ion, const arb::mechanism_desc& mech) {
    return ion_reversal_potential_method{ion, mech};
}
#undef ARBIO_DEFINE_SINGLE_ARG
#undef ARBIO_DEFINE_DOUBLE_ARG


struct evaluator {
    using any_vec = std::vector<std::any>;
    using eval_fn = std::function<std::any(any_vec)>;
    using args_fn = std::function<bool(const any_vec&)>;

    eval_fn eval;
    args_fn match_args;
    const char* message;

    evaluator(eval_fn f, args_fn a, const char* m):
        eval(std::move(f)),
        match_args(std::move(a)),
        message(m)
    {}

    std::any operator()(any_vec args) {
        return eval(std::move(args));
    }
};

// Test whether a list of arguments passed as a std::vector<std::any> can be converted
// to the types in Args.
//
// For example, the following would return true:
//
//  call_match<int, int, string>(vector<any(4), any(12), any(string("hello"))>)
template <typename... Args>
struct call_match {
    template <std::size_t I, typename T, typename Q, typename... Rest>
    bool match_args_impl(const std::vector<std::any>& args) const {
        return match<T>(args[I].type()) && match_args_impl<I+1, Q, Rest...>(args);
    }

    template <std::size_t I, typename T>
    bool match_args_impl(const std::vector<std::any>& args) const {
        return match<T>(args[I].type());
    }

    template <std::size_t I>
    bool match_args_impl(const std::vector<std::any>& args) const {
        return true;
    }

    bool operator()(const std::vector<std::any>& args) const {
        const auto nargs_in = args.size();
        const auto nargs_ex = sizeof...(Args);
        return nargs_in==nargs_ex? match_args_impl<0, Args...>(args): false;
    }
};

// Evaluate a call to a function where the arguments are provided as a std::vector<std::any>.
// The arguments are expanded and converted to the correct types, as specified by Args.
template <typename... Args>
struct call_eval {
    using ftype = std::function<std::any(Args...)>;
    ftype f;
    call_eval(ftype f): f(std::move(f)) {}

    template<std::size_t... I>
    std::any expand_args_then_eval(std::vector<std::any> args, std::index_sequence<I...>) {
        return f(eval_cast<Args>(std::move(args[I]))...);
    }

    std::any operator()(std::vector<std::any> args) {
        return expand_args_then_eval(std::move(args), std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename... Args>
struct make_call {
    evaluator state;

    template <typename F>
    make_call(F&& f, const char* msg="call"):
        state(call_eval<Args...>(std::forward<F>(f)), call_match<Args...>(), msg)
    {}

    operator evaluator() const {
        return state;
    }
};

using param_pair = std::pair<std::string, double>;
struct mech_match {
    bool operator()(const std::vector<std::any>& args) const {
        if (args.front().type() != typeid(std::string)) return false;
        for (auto it = args.begin()+1; it != args.end(); ++it) {
            if(it->type() != typeid(param_pair)) return false;
        }
        return true;
    }
};
struct mech_eval {
    arb::mechanism_desc operator()(std::vector<std::any> args) {
        auto name = std::any_cast<std::string>(args.front());
        arb::mechanism_desc mech(name);
        for (auto it = args.begin()+1; it != args.end(); ++it) {
            auto p = std::any_cast<param_pair>(*it);
            mech.set(p.first, p.second);
        }
        return mech;
    }
};
struct make_mech_call {
    evaluator state;

    make_mech_call(const char* msg="arg_vec"):
        state(mech_eval(), mech_match(), msg)
    {}

    operator evaluator() const {
        return state;
    }
};
} // anonymous namespace

struct nil_tag{};
using eval_map = std::unordered_multimap<std::string, evaluator>;
//threshold_detector
eval_map decor_eval_map {
    // Functions that return defaultables, placeables and paintables
    {"init-membrane-potential",   make_call<double>(make_init_membrane_potential,
                                  "'init-membrane-potential' with 1 argument")},
    {"temperature-kelvin",        make_call<double>(make_temperature_K,
                                  "'temperature-kelvin' with 1 argument")},
    {"axial-resistivity",         make_call<double>(make_axial_resistivity,
                                  "'axial-resistivity' with 1 argument")},
    {"membrane-capacitance",      make_call<double>(make_membrane_capacitance,
                                  "'membrane-capacitance' with 1 argument")},
    {"ion-internal-concentration",make_call<std::string, double>(make_init_int_concentration,
                                  "'ion_internal_concentration' with 2 arguments")},
    {"ion-external-concentration",make_call<std::string, double>(make_init_ext_concentration,
                                  "'ion_external_concentration' with 2 arguments")},
    {"ion-reversal-potential",    make_call<std::string, double>(make_init_reversal_potential,
                                  "'ion_reversal_potential' with 2 arguments")},
    {"current-clamp",      make_call<double, double, double>(make_i_clamp,
                           "'current-clamp' with 3 arguments")},
    {"threshold-detector", make_call<double>(make_threshold_detector,
                           "'threshold-detector' with 1 argument")},
    {"gap-junction-site",  make_call<>(make_gap_junction_site,
                           "'gap-junction-site' with 0 arguments")},
    {"ion-reversal-potential-method", make_call<std::string, arb::mechanism_desc>(make_ion_reversal_potential_method,
                                      "'ion-reversal-potential-method' with 2 arguments")},
    {"mechanism", make_mech_call("'mechanism' with at least one argument")},
};

parse_hopefully<std::any> eval(const s_expr&, const eval_map&);

parse_hopefully<std::vector<std::any>> eval_args(const s_expr& e, const eval_map& map) {
    if (!e) return {std::vector<std::any>{}}; // empty argument list
    std::vector<std::any> args;
    for (auto& h: e) {
        if (auto arg=eval(h, map)) {
            args.push_back(std::move(*arg));
        }
        else {
            return util::unexpected(std::move(arg.error()));
        }
    }
    return args;
}

parse_hopefully<std::any> eval(const s_expr& e, const eval_map& map ) {
    if (e.is_atom()) {
        auto& t = e.atom();
        switch (t.kind) {
        case tok::integer:
            return {std::stoi(t.spelling)};
        case tok::real:
            return {std::stod(t.spelling)};
        case tok::nil:
            return {nil_tag()};
        case tok::string:
            return std::any{std::string(t.spelling)};
            // An arbitrary symbol in a region/locset expression is an error, and is
            // often a result of not quoting a label correctly.
        case tok::symbol:
            return util::unexpected(cableio_unexpected_symbol(e.atom().spelling, location(e)));
        case tok::error:
            return util::unexpected(cableio_parse_error(e.atom().spelling, location(e)));
        default:
            return util::unexpected(cableio_parse_error("Unexpected term "+e.atom().spelling, location(e)));
        }
    }
    if (e.head().is_atom()) {
        // If this is a string, it must be a parameter pair
        if (e.head().atom().kind == tok::string) {
            auto args = eval_args(e.tail(), map);
            if (!args) {
                return util::unexpected(args.error());
            }
            if (args->size() != 1 || (args->front().type() != typeid(int) && args->front().type() != typeid(double))) {
                return util::unexpected(cableio_parse_error("Parameter "+e.head().atom().spelling+" can only have an integer or real value.", location(e)));
            }
            return std::any{param_pair{e.head().atom().spelling, eval_cast<double>(args->front())}};
        };
        // This must be a function evaluation, where head is the function name, and
        // tail is a list of arguments.

        // Evaluate the arguments, and return error state if an error ocurred.
        auto args = eval_args(e.tail(), map);
        if (!args) {
            return util::unexpected(args.error());
        }

        // Find all candidate functions that match the name of the function.
        auto& name = e.head().atom().spelling;
        auto matches = map.equal_range(name);

        // Search for a candidate that matches the argument list.
        for (auto i=matches.first; i!=matches.second; ++i) {
            if (i->second.match_args(*args)) { // found a match: evaluate and return.
                return i->second.eval(*args);
            }
        }
        return util::unexpected(cableio_parse_error("No matches for "+name, location(e)));
    }
    return util::unexpected(cableio_parse_error("expression is neither integer, real expression of the form (op <args>)", location(e)));
}

parse_hopefully<decor> parse_decor(const s_expr& sexp) {
    decor d;
    for (const auto& e: sexp) {
        if (e.head().is_atom()) {
            auto func = e.head().atom().spelling;
            if (func == "place" || func == "paint") {
                if (length(e.tail()) != 2) {
                    return util::unexpected(cableio_parse_error("Expected 2 arguments for " + func, location(e)));
                }
                auto it = e.tail().begin();
                auto label = parse_label_expression(*it);
                auto decoration = eval(*(it + 1), decor_eval_map);
                if (!label) {
                    return util::unexpected(cableio_parse_error(label.error().what(), {}));
                }
                if (!decoration) {
                    return util::unexpected(decoration.error());
                }
                if (func == "place") {
                    auto p = any_to_placeable(decoration.value());
                    if (!match<locset>(label->type())) {
                        return util::unexpected(cableio_parse_error("expected locset as arg for place", {}));
                    }
                    if (!p) {
                        return util::unexpected(cableio_parse_error("expected placeable as arg for place", {}));
                    }
                    d.place(std::any_cast<arb::locset>(label.value()), p.value());
                }
                if (func == "paint") {
                    auto p = any_to_paintable(decoration.value());
                    if (!match<region>(label->type())) {
                        return util::unexpected(cableio_parse_error("expected region as arg for paint", {}));
                    }
                    if (!p) {
                        return util::unexpected(cableio_parse_error("expected paintable as arg for place", {}));
                    }
                    d.paint(std::any_cast<arb::region>(label.value()), p.value());
                }
            }
            else if (func == "default") {
                if (length(e.tail()) != 1) {
                    return util::unexpected(cableio_parse_error("Expected 1 arguments for " + func, location(e)));
                }
                auto decoration = eval(*(e.tail().begin()), decor_eval_map);
                if (!decoration) {
                    return util::unexpected(decoration.error());
                }
                auto p = any_to_defaultable(decoration.value());
                if (!p) {
                    return util::unexpected(cableio_parse_error("expected defaultable as arg for place", {}));
                }
                d.set_default(std::any_cast<defaultable>(p.value()));
            }
            else {
                return util::unexpected(cableio_parse_error("Expected paint, place or default atom", location(e)));
            }
        }
        else {
            return util::unexpected(cableio_parse_error("Expected paint, place or default atom", location(e)));
        }
    }
    std::cout << "whoooo" << std::endl;
    return d;
}
parse_hopefully<decor> parse_decor(const std::string& str) {
    return parse_decor(parse_s_expr(str));
}

parse_hopefully<label_dict> parse_label_dict(const s_expr& sexp) {
    label_dict d;
    for (const auto& e: sexp) {
        if (e.head().is_atom()) {
            auto func = e.head().atom().spelling;
            if (func == "region-def" || func == "locset-def") {
                if (length(e.tail()) != 2) {
                    return util::unexpected(cableio_parse_error("Expected 2 arguments for " + func, location(e)));
                }
                auto it = e.tail().begin();
                if (!it->is_atom() || it->atom().kind != tok::string) {
                    return util::unexpected(cableio_parse_error("Expected string as arg for region-def", {}));
                }
                auto label = it->atom().spelling;
                auto desc = parse_label_expression(*(it+1));
                if (!desc) {
                    return util::unexpected(cableio_parse_error(desc.error().what(), {}));
                }
                if (func == "region-def") {
                    if (!match<region>(desc->type())) {
                        return util::unexpected(cableio_parse_error("expected region as arg for region-def", {}));
                    }
                    d.set(label, eval_cast<region>(desc.value()));
                }
                if (func == "locset-def") {
                    if (!match<locset>(desc->type())) {
                        return util::unexpected(cableio_parse_error("expected locset as arg for locset-def", {}));
                    }
                    d.set(label, eval_cast<locset>(desc.value()));
                }
            }
            else {
                return util::unexpected(cableio_parse_error("Expected region-def or locset-def", location(e)));
            }
        }
        else {
            return util::unexpected(cableio_parse_error("Expected region-def or locset-def", location(e)));
        }
    }
    std::cout << "whoooo" << std::endl;
    return d;
}
parse_hopefully<label_dict> parse_label_dict(const std::string& str) {
    return parse_label_dict(parse_s_expr(str));
}

parse_hopefully<mechanism_desc> parse(const std::string& str) {
    auto s = parse_s_expr(str);
    auto e = eval(s, decor_eval_map);
    if (!e) {
        return util::unexpected(cableio_parse_error(std::string()+e.error().what(), {}));
    }
    if (e->type() == typeid(mechanism_desc)) {
        std::cout << "wooo" << std::endl;
        return std::any_cast<mechanism_desc>(*e);
    }
    return util::unexpected(cableio_parse_error("wtf is this",{}));
}

/*struct label_pair {
    std::string label;
    std::variant<region, locset> desc;
};

parse_hopefully<label_pair> eval_dict(const s_expr& e) {
    if (e.is_atom()) {
        return util::unexpected(format_parse_error("eval_dict expected atom"));
    }
    if (e.head().is_atom()) {
        auto& name = e.head().atom().spelling;
        if (name != "region-def" && name != "locset-def") {
            return util::unexpected(format_parse_error("eval_dict expected region-def or locset-def"));
        }
        auto args = e.tail();
        if (!args.head().is_atom() || args.head().atom().kind != tok::string) {
            return util::unexpected(format_parse_error("eval_dict arg expected string atom"));
        }
        if (args.tail().is_atom()) {
            return util::unexpected(format_parse_error("eval_dict arg did not expect string atom"));
        }
        if (!args.tail().tail().is_atom() || args.tail().tail().atom().kind != tok::nil) {
            return util::unexpected(format_parse_error("eval_dict expected bahh"));
        }
        label_pair p;
        p.label = args.head().atom().spelling;
        if (auto desc = parse_label_expression(args.tail().head())) {
            if (desc->type() == typeid(locset) && name == "locset-def") {
                p.desc = std::any_cast<locset&>(*desc);
                return p;
            }
            if (desc->type() == typeid(region) && name == "region-def") {
                p.desc = std::any_cast<region&>(*desc);
                return p;
            }
        } else {
            throw desc.error();
        }
    }
    return util::unexpected(format_parse_error("expected something else"));
}

parse_hopefully<label_dict> parse_label_dict(const std::string& str) {
    auto s = parse_s_expr(str);
    if (!s.head().is_atom()) {
        throw format_parse_error("Expected atom at head");
    }
    if (s.head().atom().kind != tok::symbol) {
        throw format_parse_error("Expected symbol at head");
    }
    if (s.head().atom().spelling != "label-dict") {
        throw format_parse_error("Expected label-dict symbol at head");
    }
    label_dict d;
    for (const auto& t: s.tail()) {
        if (auto e = eval_dict(t)) {
            if (auto eval = std::get_if<region>(&e->desc)) d.set(e->label, *eval);
            if (auto eval = std::get_if<locset>(&e->desc)) d.set(e->label, *eval);
        }
        else {
            throw e.error();
        }
    }
    return d;
}*/
/*
{    label_dict d;
    for (auto& entry: e) {
        auto it = std::begin(entry);
        std::string kind = get<symbol>(*it);
        ++it;
        if (kind=="locset-def") {
            std::string name = get<std::string>(*it);
            ++it;
            region reg = parse_region_expression(*it);
            d.set(name, std::move(reg));
        }
        else {
            std::string name = get<std::string>(*it);
            ++it;
            region reg = parse_locset_expression(*it);
            d.set(name, std::move(reg));
        }
    }

    return d;
}
*/
} // namespace arborio
