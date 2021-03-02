#include <iostream>
#include <numeric>
#include <functional>
#include <sstream>
#include <numeric>

#include <arbor/cable_cell.hpp>
#include <arbor/s_expr.hpp>

#include <arborio/cableio.hpp>

#include "cable_cell_format.hpp"

namespace arborio {

using namespace arb;

// Errors
cableio_parse_error::cableio_parse_error(const std::string& msg, const arb::src_location& loc):
    arb::arbor_exception(msg+" at :"+std::to_string(loc.line)+":"+std::to_string(loc.column))
{}
cableio_unexpected_symbol::cableio_unexpected_symbol(const std::string& sym, const arb::src_location& loc):
    cableio_parse_error("Unexpected symbol "+sym, loc) {}

// Write s-expr
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

parse_hopefully<decor> parse_decor(const s_expr& sexp) {
    auto dec = parse_decor_expression(sexp);
    if (!dec || (typeid(decor) != dec->type())) {
        return util::unexpected(cableio_parse_error("Expected a decor honey...", {}));
    }
    return std::any_cast<decor>(dec.value());
}
parse_hopefully<decor> parse_decor(const std::string& str) {
    return parse_decor(parse_s_expr(str));
}

parse_hopefully<label_dict> parse_label_dict(const s_expr& sexp) {
    auto dict = parse_label_dict_expression(sexp);
    if (!dict || (typeid(label_dict) != dict->type())) {
        return util::unexpected(cableio_parse_error("Expected a label_dict honey...", {}));
    }
    return std::any_cast<label_dict>(dict.value());
}
parse_hopefully<label_dict> parse_label_dict(const std::string& str) {
    return parse_label_dict(parse_s_expr(str));
}

parse_hopefully<morphology> parse_morphology(const s_expr& sexp) {
    auto morpho = parse_morphology_expression(sexp);
    if (!morpho || (typeid(morphology) != morpho->type())) {
        std::cout << morpho.error().what() << std::endl;
        return util::unexpected(cableio_parse_error("Expected a morphology honey...", {}));
    }
    return std::any_cast<morphology>(morpho.value());
}
parse_hopefully<morphology> parse_morphology(const std::string& str) {
    return parse_morphology(parse_s_expr(str));
}
} // namespace arborio
