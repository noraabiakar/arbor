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
parse_hopefully<cable_cell_component> parse_component(const std::string& s) {
    return parse(parse_s_expr(s));
};

} // namespace arborio
