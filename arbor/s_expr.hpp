#pragma once

#include <cstddef>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "arbor/arbexcept.hpp"

namespace arb {

struct src_location {
    unsigned line = 0;
    unsigned column = 0;

    src_location() = default;

    src_location(unsigned l, unsigned c):
        line(l), column(c)
    {}
};

std::ostream& operator<<(std::ostream& o, const src_location& l);

enum class tok {
    nil,
    real,       // real number
    integer,    // integer
    symbol,     // symbol
    lparen,     // left parenthesis '('
    rparen,     // right parenthesis ')'
    string,     // string, written as "spelling"
    eof,        // end of file/input
    error       // special error state marker
};

std::ostream& operator<<(std::ostream&, const tok&);

struct token {
    src_location loc;
    tok kind;
    std::string spelling;
};

std::ostream& operator<<(std::ostream&, const token&);

inline token nil_token(src_location l={}) {
    return token{l, tok::nil, "()"};
}

struct symbol {
    std::string str;
    operator std::string() const { return str; }
    bool friend operator< (const symbol& lhs, const symbol& rhs) { return lhs.str<rhs.str; }
    bool friend operator==(const symbol& lhs, const symbol& rhs) { return lhs.str==rhs.str; }
};

namespace s_expr_literals {
    inline symbol operator "" _symbol(const char* chars, size_t size) {
        return {chars};
    }
}

struct s_expr {
    template <typename U>
    struct s_pair {
        U head = U();
        U tail = U();
        s_pair(U l, U r): head(std::move(l)), tail(std::move(r)) {}
    };

    // This value_wrapper is used to wrap the shared pointer
    template <typename T>
    struct value_wrapper{
        using state_t = std::unique_ptr<T>;
        state_t state;

        value_wrapper() = default;

        value_wrapper(const T& v):
            state(std::make_unique<T>(v)) {}

        value_wrapper(T&& v):
            state(std::make_unique<T>(std::move(v))) {}

        value_wrapper(const value_wrapper& other):
            state(std::make_unique<T>(other.get())) {}

        value_wrapper& operator=(const value_wrapper& other) {
            state = std::make_unique<T>(other.get());
            return *this;
        }

        value_wrapper(value_wrapper&& other) = default;

        friend std::ostream& operator<<(std::ostream& o, const value_wrapper& w) {
            return o << *w.state;
        }

        operator T() const {
            return *state;
        }

        const T& get() const {
            return *state;
        }

        T& get() {
            return *state;
        }
    };

    template <bool Const>
    class s_expr_iterator_impl {
        public:

        struct sentinel {};

        using value_type = s_expr;
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using pointer   = std::conditional_t<Const, const s_expr*, s_expr*>;
        using reference = std::conditional_t<Const, const s_expr&, s_expr&>;

        s_expr_iterator_impl(reference e):
            inner_(&e)
        {
            // We can't iterate over an atom, unless the atom is
            // nil, which is both an atom and an empty list.
            if (inner_->is_atom() && inner_->atom().kind!=tok::nil) {
                throw std::runtime_error("Attempt to create s_expr_iterator on an atom.");
            }
            if (finished()) inner_ = nullptr;
        }

        s_expr_iterator_impl(const sentinel& e):
            inner_(nullptr)
        {}

        reference operator*() const {
            return inner_->head();
        }

        pointer operator->() const {
            return &inner_->head();
        }

        s_expr_iterator_impl& operator++() {
            advance();
            return *this;
        }

        s_expr_iterator_impl operator++(int) {
            s_expr_iterator_impl cur = *this;
            advance();
            return cur;
        }

        s_expr_iterator_impl operator+(difference_type i) const {
            s_expr_iterator_impl it = *this;
            while (i--) {
                ++it;
            }
            return it;
        }
        bool operator==(const s_expr_iterator_impl& other) const {
            return inner_==other.inner_;
        }
        bool operator!=(const s_expr_iterator_impl& other) const {
            return !(*this==other);
        }
        bool operator==(const sentinel& other) const {
            return !inner_;
        }
        bool operator!=(const sentinel& other) const {
            return !(*this==other);
        }

        reference expression() const {
            return *inner_;
        }

        private:

        bool finished() const {
            return inner_->is_atom() && inner_->atom().kind==tok::nil;
        }

        void advance() {
            if (!inner_) return;
            inner_ = &inner_->tail();
            if (finished()) inner_ = nullptr;
        }

        // Pointer to the current s_expr.
        // Set to nullptr when at the end of the range.
        pointer inner_;
    };

    using iterator       = s_expr_iterator_impl<false>;
    using const_iterator = s_expr_iterator_impl<true>;

    // An s_expr can be one of
    //      1. an atom
    //      2. a pair of s_expr (head and tail)
    // The s_expr uses a util::variant to represent these two possible states,
    // which requires using an incomplete definition of s_expr, requiring
    // with a std::unique_ptr via value_wrapper.

    using pair_type = s_pair<value_wrapper<s_expr>>;
    std::variant<token, pair_type> state = nil_token();

    s_expr(const s_expr& s): state(s.state) {}
    s_expr() = default;
    s_expr(token t): state(std::move(t)) {}
    s_expr(s_expr l, s_expr r):
        state(pair_type(std::move(l), std::move(r)))
    {}

    s_expr(std::string s):
        s_expr(token{{0,0}, tok::string, std::move(s)}) {}
    s_expr(const char* s):
        s_expr(token{{0,0}, tok::string, s}) {}
    s_expr(double x):
        s_expr(token{{0,0}, tok::real, std::to_string(x)}) {}
    s_expr(int x):
        s_expr(token{{0,0}, tok::integer, std::to_string(x)}) {}
    s_expr(symbol s):
        s_expr(token{{0,0}, tok::symbol, s}) {}

    bool is_atom() const;

    const token& atom() const;

    operator bool() const;

    const s_expr& head() const;
    const s_expr& tail() const;
    s_expr& head();
    s_expr& tail();

    iterator       begin()        { return {*this}; }
    iterator       end()          { return iterator::sentinel{}; }
    const_iterator begin()  const { return {*this}; }
    const_iterator end()    const { return const_iterator::sentinel{}; }
    const_iterator cbegin() const { return {*this}; }
    const_iterator cend()   const { return const_iterator::sentinel{}; }

    friend std::ostream& operator<<(std::ostream& o, const s_expr& x);
};

struct bad_s_expr_get: arbor_exception {
    bad_s_expr_get(const std::string& msg):
        arbor_exception("bad_s_expr_get: "+msg)
    {}
};

template <typename T>
T get(const s_expr&) {
    throw bad_s_expr_get("no cast to type possible");
}

template <>
double get<double>(const s_expr& e);

// Helper function for programmatically building lists
//
//   slist(1, 2, "hello world", "banjax@cat/3"_symbol);
//
// Would produce the following s-expression:
//
//   (1 2 "hello world" banjax@cat/3)
//
// And can be nested:
//
//   slist(1, slist(2, 3), 4, 5 );
//
// Produces:
//
//   (1 (2 3) 4 5)

template <typename T>
s_expr slist(T v) {
    return {v, {}};
}

template <typename T, typename... Args>
s_expr slist(T v, Args... args) {
    return {v, slist(args...)};
}

inline s_expr slist() {
    return {};
}

template <typename I, typename S>
s_expr slist_range(I b, S e) {
    return b==e ? s_expr{}
                : s_expr{*b, slist_range(++b,e)};
}

template <typename Range>
s_expr slist_range(const Range& range) {
    return slist_range(std::begin(range), std::end(range));
}

std::size_t length(const s_expr& l);
src_location location(const s_expr& l);

s_expr parse_s_expr(const std::string& line);

} // namespace arb

