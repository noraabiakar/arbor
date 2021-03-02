#pragma once

#include <arbor/s_expr.hpp>
#include <arborio/cableio.hpp>

namespace arborio {

inline arb::symbol operator"" _symbol(const char* chars, size_t size) {
    return {chars};
}
struct nil_tag {};

// S-expression makers
arb::s_expr mksexp(const arb::decor& d);
arb::s_expr mksexp(const arb::label_dict& dict);
arb::s_expr mksexp(const arb::morphology& morph);

// S-expression evaluator
parse_hopefully<std::any> parse_decor_expression(const arb::s_expr&);
parse_hopefully<std::any> parse_label_dict_expression(const arb::s_expr&);
parse_hopefully<std::any> parse_morphology_expression(const arb::s_expr&);

} // namespace arborio