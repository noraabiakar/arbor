#pragma once

#include <iosfwd>
#include <string>

#include "module.hpp"
#include "visitor.hpp"

#include "printer/cexpr_emit.hpp"
#include "printer/simd.hpp"

std::string emit_cpp_source(const Module& m, const std::string& ns, simd_spec simd);

// CPrinter and SimdPrinter visitors exposed in header for testing purposes only.

class CPrinter: public Visitor {
public:
    CPrinter(std::ostream& out): out_(out) {}

    void visit(Expression* e) override {
        throw compiler_exception("CPrinter cannot translate expression "+e->to_string());
    }

    void visit(BlockExpression*) override;
    void visit(CallExpression*) override;
    void visit(IdentifierExpression*) override;
    void visit(VariableExpression*) override;
    void visit(LocalVariable*) override;
    void visit(IndexedVariable*) override;

    // Delegate low-level emits to cexpr_emit:
    void visit(NumberExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(UnaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(BinaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(IfExpression* e) override { cexpr_emit(e, out_, this); }

private:
    std::ostream& out_;
};

class SimdPrinter: public Visitor {
public:
    SimdPrinter(std::ostream& out): out_(out) {}

    void visit(Expression* e) override {
        throw compiler_exception("SimdPrinter cannot translate expression "+e->to_string());
    }

    void set_var_indexed_to(bool is_var_indexed) {
        is_var_indexed_ = is_var_indexed;
    }
    void set_contiguous_to(bool is_contiguous) {
        is_contiguous_ = is_contiguous;
    }
    void set_constant_to(bool is_constant) {
        is_constant_ = is_constant;
    }

    void visit(BlockExpression*) override;
    void visit(CallExpression*) override;
    void visit(IdentifierExpression*) override;
    void visit(VariableExpression*) override;
    void visit(LocalVariable*) override;
    void visit(IndexedVariable*) override;
    void visit(AssignmentExpression*) override;

    void visit(NumberExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(UnaryExpression* e) override { cexpr_emit(e, out_, this); }
    void visit(BinaryExpression* e) override { cexpr_emit(e, out_, this); }

private:
    std::ostream& out_;
    bool is_var_indexed_;
    bool is_contiguous_;
    bool is_constant_;
};
