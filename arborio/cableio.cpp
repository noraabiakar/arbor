#include <iostream>
#include <numeric>
#include <functional>
#include <sstream>
#include <numeric>

#include <arbor/cable_cell.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/any_visitor.hpp>

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
    decor d;
    for (const auto& e: sexp) {
        if (e.head().is_atom()) {
            auto func = e.head().atom().spelling;
            auto tail_ptr = e.tail().begin();
            if (func == "place" || func == "paint") {
                if (length(e.tail()) != 2) {
                    return util::unexpected(cableio_parse_error("Expected 2 arguments for " + func, location(e)));
                }
                auto label = parse_label_expression(*tail_ptr);
                auto decoration = parse_decor_expression(*(tail_ptr + 1));
                if (!label)      return util::unexpected(cableio_parse_error(label.error().what(), {}));
                if (!decoration) return util::unexpected(decoration.error());
                if (func == "place") {
                    if (label->type() != typeid(locset)) {
                        return util::unexpected(cableio_parse_error("expected locset as arg for place", {}));
                    }
                    auto p = cast_to_variant<placeable>(decoration.value());
                    if (!p) {
                        return util::unexpected(cableio_parse_error("expected placeable as arg for place", {}));
                    }
                    d.place(std::any_cast<arb::locset>(label.value()), p.value());
                }
                if (func == "paint") {
                    auto p = cast_to_variant<paintable>(decoration.value());
                    if (label->type() != typeid(region)) {
                        return util::unexpected(cableio_parse_error("expected region as arg for paint", {}));
                    }
                    if (!p) {
                        return util::unexpected(cableio_parse_error("expected paintable as arg for paint", {}));
                    }
                    d.paint(std::any_cast<arb::region>(label.value()), p.value());
                }

            }
            else if (func == "default") {
                if (length(e.tail()) != 1) {
                    return util::unexpected(cableio_parse_error("Expected 1 arguments for " + func, location(e)));
                }
                auto decoration = parse_decor_expression(*(e.tail().begin()));
                if (!decoration) return util::unexpected(decoration.error());
                auto p = cast_to_variant<defaultable>(decoration.value());
                if (!p) {
                    return util::unexpected(cableio_parse_error("expected defaultable as arg for default", {}));
                }
                d.set_default(std::any_cast<defaultable>(p.value()));
            }
            else {
                return util::unexpected(cableio_parse_error("Expected paint, place or default", location(e)));
            }
        }
        else {
            return util::unexpected(cableio_parse_error("Expected atom", location(e)));
        }
    }
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
            auto tail_ptr = e.tail().begin();
            if (func == "region-def" || func == "locset-def") {
                if (length(e.tail()) != 2) {
                    return util::unexpected(cableio_parse_error("Expected 2 arguments for " + func, location(e)));
                }
                if (!tail_ptr->is_atom() || tail_ptr->atom().kind != tok::string) {
                    return util::unexpected(cableio_parse_error("Expected string as arg for region-def", {}));
                }
                auto label = tail_ptr->atom().spelling;
                auto desc = parse_label_expression(*(tail_ptr+1));
                if (!desc)  return util::unexpected(cableio_parse_error(desc.error().what(), {}));
                if (func == "region-def") {
                    if (desc->type() != typeid(region)) {
                        return util::unexpected(cableio_parse_error("expected region as arg for region-def", {}));
                    }
                    d.set(label, std::any_cast<region>(desc.value()));
                }
                if (func == "locset-def") {
                    if (desc->type() != typeid(locset)) {
                        return util::unexpected(cableio_parse_error("expected locset as arg for locset-def", {}));
                    }
                    d.set(label, std::any_cast<locset>(desc.value()));
                }
            }
            else {
                return util::unexpected(cableio_parse_error("Expected region-def or locset-def", location(e)));
            }
        }
        else {
            return util::unexpected(cableio_parse_error("Expected atom", location(e)));
        }
    }
    return d;
}
parse_hopefully<label_dict> parse_label_dict(const std::string& str) {
    return parse_label_dict(parse_s_expr(str));
}

parse_hopefully<mechanism_desc> parse(const std::string& str) {
    auto s = parse_s_expr(str);
    auto e = parse_decor_expression(s);
    if (!e) {
        return util::unexpected(cableio_parse_error(std::string()+e.error().what(), {}));
    }
    if (e->type() == typeid(mechanism_desc)) {
        std::cout << "wooo" << std::endl;
        return std::any_cast<mechanism_desc>(*e);
    }
    return util::unexpected(cableio_parse_error("wtf is this",{}));
}
} // namespace arborio
