#pragma once

// Base class (via CRTP) for concrete implementation
// classes, with default implementations based on
// copy_to/copy_from.
//
// Also provides simd_detail::simd_traits type map.
//
// Maths functions are implemented in terms of
// arithmetic primitives or lane-wise invocation of
// std::math functions; specialized implementations
// should be provided by the concrete implementation
// where it is more efficient:
//
// Function | Default implemention by
// ----------------------------------
// min      | negate, cmp_gt, ifelse
// max      | negate, cmp_gt, ifelse
// abs      | negate, max
// sin      | lane-wise std::sin
// cos      | lane-wise std::cos
// exp      | lane-wise std::exp
// log      | lane-wise std::log
// pow      | lane-wise std::pow
// expm1    | lane-wise std::expm1
// exprelr  | expm1, div, add, cmp_eq, ifelse
//
// 'exprelr' is the function x ↦ x/(exp(x)-1).

#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>

// Derived class I must at minimum provide:
//
// * specialization of simd_traits.
//
// * implementations (static) for copy_to, copy_from:
//
//     void I::copy_to(const vector_type&, scalar_type*)
//     vector_type I::copy_from(const scalar_type*)
//
//     void I::mask_copy_to(const vector_type&, scalar_type*)
//     vector_type I::mask_copy_from(const bool*)
//
// * implementations (static) for mask element get/set:
//
//     bool I::mask_element(const vector_type& v, int i);
//     void I::mask_set_element(vector_type& v, int i, bool x);

namespace arb {
namespace simd {
namespace simd_detail {

// The simd_traits class provides the mapping between a concrete SIMD
// implementation I and its associated classes. This must be specialized
// for each concrete implementation.

template <typename I>
struct simd_traits {
    static constexpr unsigned width = 0;
    using scalar_type = void;
    using vector_type = void;
    using mask_impl = void;
};

template <typename I>
struct implbase {
    constexpr static unsigned width = simd_traits<I>::width;
    using scalar_type = typename simd_traits<I>::scalar_type;
    using vector_type = typename simd_traits<I>::vector_type;

    using mask_impl = typename simd_traits<I>::mask_impl;
    using mask_type = typename simd_traits<mask_impl>::vector_type;

    using store = scalar_type[width];
    using mask_store = bool[width];

    static vector_type broadcast(scalar_type x) {
        store a;
        std::fill(std::begin(a), std::end(a), x);
        return I::copy_from(a);
    }

    static scalar_type element(const vector_type& v, int i) {
        store a;
        I::copy_to(v, a);
        return a[i];
    }

    static void set_element(vector_type& v, int i, scalar_type x) {
        store a;
        I::copy_to(v, a);
        a[i] = x;
        v = I::copy_from(a);
    }

    static void copy_to_masked(const vector_type& v, scalar_type* p, const mask_type& mask) {
        store a;
        I::copy_to(v, a);

        mask_store m;
        mask_impl::mask_copy_to(mask, m);
        for (unsigned i = 0; i<width; ++i) {
            if (m[i]) p[i] = a[i];
        }
    }

    static vector_type copy_from_masked(const scalar_type* p, const mask_type& mask) {
        store a;

        mask_store m;
        mask_impl::mask_copy_to(mask, m);
        for (unsigned i = 0; i<width; ++i) {
            if (m[i]) a[i] = p[i];
        }
        return I::copy_from(a);
    }

    static vector_type copy_from_masked(const vector_type& v, const scalar_type* p, const mask_type& mask) {
        store a;
        I::copy_to(v, a);

        mask_store m;
        mask_impl::mask_copy_to(mask, m);
        for (unsigned i = 0; i<width; ++i) {
            if (m[i]) a[i] = p[i];
        }
        return I::copy_from(a);
    }

    static vector_type negate(const vector_type& u) {
        store a, r;
        I::copy_to(u, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = -a[i];
        }
        return I::copy_from(r);
    }

    static vector_type add(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]+b[i];
        }
        return I::copy_from(r);
    }

    static vector_type mul(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]*b[i];
        }
        return I::copy_from(r);
    }

    static vector_type sub(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]-b[i];
        }
        return I::copy_from(r);
    }

    static vector_type div(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]/b[i];
        }
        return I::copy_from(r);
    }

    static vector_type fma(const vector_type& u, const vector_type& v, const vector_type& w) {
        store a, b, c, r;
        I::copy_to(u, a);
        I::copy_to(v, b);
        I::copy_to(w, c);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::fma(a[i], b[i], c[i]);
        }
        return I::copy_from(r);
    }

    static mask_type cmp_eq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]==b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_neq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]!=b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_gt(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]>b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_geq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]>=b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_lt(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]<b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static mask_type cmp_leq(const vector_type& u, const vector_type& v) {
        store a, b;
        mask_store r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i]<=b[i];
        }
        return mask_impl::mask_copy_from(r);
    }

    static vector_type logical_not(const vector_type& u) {
        store a, r;
        I::copy_to(u, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = !a[i];
        }
        return I::copy_from(r);
    }

    static vector_type logical_and(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i] && b[i];
        }
        return I::copy_from(r);
    }

    static vector_type logical_or(const vector_type& u, const vector_type& v) {
        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = a[i] || b[i];
        }
        return I::copy_from(r);
    }

    static vector_type ifelse(const mask_type& mask, const vector_type& u, const vector_type& v) {
        mask_store m;
        mask_impl::mask_copy_to(mask, m);

        store a, b, r;
        I::copy_to(u, a);
        I::copy_to(v, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = m[i]? a[i]: b[i];
        }
        return I::copy_from(r);
    }

    static vector_type mask_broadcast(bool v) {
        mask_store m;
        std::fill(std::begin(m), std::end(m), v);
        return I::mask_copy_from(m);
    }

    static vector_type mask_unpack(unsigned long long k) {
        mask_store m;
        for (unsigned i = 0; i<width; ++i) {
            m[i] = k&(1ull<<i);
        }
        return I::mask_copy_from(m);
    }

    template <typename ImplIndex>
    static vector_type gather(ImplIndex, const scalar_type* p, const typename ImplIndex::vector_type& index) {
        typename ImplIndex::scalar_type o[width];
        ImplIndex::copy_to(index, o);

        store a;
        for (unsigned i = 0; i<width; ++i) {
            a[i] = p[o[i]];
        }
        return I::copy_from(a);
    }

    template <typename ImplIndex>
    static vector_type gather(ImplIndex, const vector_type& s, const scalar_type* p, const typename ImplIndex::vector_type& index, const mask_type& mask) {
        mask_store m;
        mask_impl::mask_copy_to(mask, m);

        typename ImplIndex::scalar_type o[width];
        ImplIndex::copy_to(index, o);

        store a;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            if (m[i]) { a[i] = p[o[i]]; }
        }
        return I::copy_from(a);
    }

    template <typename ImplIndex>
    static void scatter(ImplIndex, const vector_type& s, scalar_type* p, const typename ImplIndex::vector_type& index) {
        typename ImplIndex::scalar_type o[width];
        ImplIndex::copy_to(index, o);

        store a;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            p[o[i]] = a[i];
        }
    }

    template <typename ImplIndex>
    static void scatter(ImplIndex, const vector_type& s, scalar_type* p, const typename ImplIndex::vector_type& index, const mask_type& mask) {
        mask_store m;
        mask_impl::mask_copy_to(mask, m);

        typename ImplIndex::scalar_type o[width];
        ImplIndex::copy_to(index, o);

        store a;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            if (m[i]) { p[o[i]] = a[i]; }
        }
    }

    // Maths

    static vector_type abs(const vector_type& u) {
        store a;
        I::copy_to(u, a);

        for (unsigned i = 0; i<width; ++i) {
            a[i] = std::abs(a[i]);
        }
        return I::copy_from(a);
    }

    static vector_type min(const vector_type& s, const vector_type& t) {
        return ifelse(cmp_gt(t, s), s, t);
    }

    static vector_type max(const vector_type& s, const vector_type& t) {
        return ifelse(cmp_gt(t, s), t, s);
    }

    static vector_type sin(const vector_type& s) {
        store a, r;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::sin(a[i]);
        }
        return I::copy_from(r);
    }

    static vector_type cos(const vector_type& s) {
        store a, r;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::cos(a[i]);
        }
        return I::copy_from(r);
    }

    static vector_type exp(const vector_type& s) {
        store a, r;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::exp(a[i]);
        }
        return I::copy_from(r);
    }

    static vector_type expm1(const vector_type& s) {
        store a, r;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::expm1(a[i]);
        }
        return I::copy_from(r);
    }

    static vector_type log(const vector_type& s) {
        store a, r;
        I::copy_to(s, a);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::log(a[i]);
        }
        return I::copy_from(r);
    }

    static vector_type exprelr(const vector_type& s) {
        vector_type ones = I::broadcast(1);
        return ifelse(cmp_eq(ones, add(ones, s)), ones, div(s, expm1(s)));
    }

    static vector_type pow(const vector_type& s, const vector_type &t) {
        store a, b, r;
        I::copy_to(s, a);
        I::copy_to(t, b);

        for (unsigned i = 0; i<width; ++i) {
            r[i] = std::pow(a[i], b[i]);
        }
        return I::copy_from(r);
    }
};

} // namespace simd_detail
} // namespace simd
} // namespace arb
