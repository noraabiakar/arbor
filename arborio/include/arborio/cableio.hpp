#pragma once

#include <ostream>

#include <arbor/cable_cell.hpp>

namespace arborio {

std::ostream& write_s_expr(std::ostream&, const arb::cable_cell&);
std::ostream& write_s_expr(std::ostream&, const arb::label_dict&);
std::ostream& write_s_expr(std::ostream&, const arb::decor&);
std::ostream& write_s_expr(std::ostream&, const arb::morphology&);

} // namespace arb
