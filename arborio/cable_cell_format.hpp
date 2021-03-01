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

template <typename T, std::size_t I=0>
std::optional<T> cast_to_variant(const std::any& a) {
    if constexpr (I<std::variant_size_v<T>) {
        using var_type = std::variant_alternative_t<I, T>;
        return typeid(var_type) == a.type()? std::any_cast<var_type>(a): cast_to_variant<T, I+1>(a);
    }
    return std::nullopt;
}

} // namespace arborio