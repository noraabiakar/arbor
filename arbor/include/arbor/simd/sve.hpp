#pragma once

// SVE SIMD intrinsics implementation.

#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>
#include <array>
#include <cmath>
#include <cstdint>

#include <arbor/simd/approx.hpp>
#include <arbor/simd/implbase.hpp>

namespace arb {
namespace simd {
namespace detail {

struct sve_double8;
struct sve_int8;
struct sve_mask8;

template <>
struct simd_traits<sve_mask8> {
    static constexpr unsigned width = 8;
    using scalar_type = bool;
    using vector_type = std::array<uint8_t, width>;
    using mask_impl = sve_mask8;
};

template <>
struct simd_traits<sve_double8> {
    static constexpr unsigned width = 8;
    using scalar_type = double;
    using vector_type = std::array<double, width>;
    using mask_impl = sve_mask8;
};

template <>
struct simd_traits<sve_int8> {
    static constexpr unsigned width = 8;
    using scalar_type = int32_t;
    using vector_type = std::array<int32_t, width>;
    using mask_impl = sve_mask8;
};

struct sve_mask8: implbase<sve_mask8> {
    using implbase<sve_mask8>::gather;
    using implbase<sve_mask8>::scatter;
    using implbase<sve_mask8>::cast_from;

    using bool_arr = std::array<uint8_t, 8>;

    static void copy_to(const bool_arr& k, bool* b) {
        std::copy(std::begin(k), std::end(k), b);
    }

    static bool_arr copy_from(const bool* p) {
        return std::copy(b, b+a.size(), std::begin(a));
    }

    static bool_arr broadcast(bool b) {
        bool_arr a;
        std::fill(std::begin(a), std::end(a), b);
        return a;
    }

    static bool_arr logical_not(const bool_arr& k) {
        bool_arr a;
        copy_to_sve(svnot_b_z(svptrue_b64(), copy_from_sve(k), a);
        return a;
    }

    static bool_arr logical_and(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(svand_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr logical_or(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(svorr_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    // Arithmetic operations not necessarily appropriate for
    // packed bit mask, but implemented for completeness/testing,
    // with Z modulo 2 semantics:
    //     a + b   is equivalent to   a ^ b
    //     a * b                      a & b
    //     a / b                      a
    //     a - b                      a ^ b
    //     -a                         a
    //     max(a, b)                  a | b
    //     min(a, b)                  a & b

    static bool_arr negate(const bool_arr& a) {
        return a;
    }

    static bool_arr add(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(sveor_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr sub(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(sveor_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr mul(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(svand_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr div(const bool_arr& a, const bool_arr& b) {
        return a;
    }

    static bool_arr fma(const bool_arr& a, const bool_arr& b, const bool_arr& c) {
        return add(mul(a, b), c);
    }

    static bool_arr max(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(svorr_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr min(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(svand_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    // Comparison operators are also taken as operating on Z modulo 2,
    // with 1 > 0:
    //
    //     a > b    is equivalent to  a & ~b
    //     a >= b                     a | ~b,  ~(~a & b)
    //     a < b                      ~a & b
    //     a <= b                     ~a | b,  ~(a & ~b)
    //     a == b                     ~(a ^ b)
    //     a != b                     a ^ b

    static bool_arr cmp_eq(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(svnot_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr cmp_neq(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(sveor_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr cmp_lt(const bool_arr& a, const bool_arr& b) {
        bool_arr r;
        copy_to_sve(svbic_b_z(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr cmp_gt(const bool_arr& a, const bool_arr& b) {
        return cmp_lt(b, a);
    }

    static bool_arr cmp_geq(const bool_arr& a, const bool_arr& b) {
        return logical_not(cmp_lt(a, b));
    }

    static bool_arr cmp_leq(const bool_arr& a, const bool_arr& b) {
        return logical_not(cmp_gt(a, b));
    }

    static bool_arr ifelse(const bool_arr& m, const bool_arr& u, const bool_arr& v) {
        bool_arr r;
        copy_to_sve(svsel_b(svptrue_b64(), copy_from_sve(a), copy_from_sve(b)), r);
        return r;
    }

    static bool_arr mask_broadcast(bool b) {
        return broadcast(b);
    }

    static bool mask_element(const bool_arr& u, int i) {
        return element(u, i);
    }

    static void mask_set_element(bool_arr& u, int i, bool b) {
        set_element(u, i, b);
    }

    static void mask_copy_to(const bool_arr& m, bool* y) {
        copy_to(m, y);
    }

    static bool_arr mask_copy_from(const bool* y) {
        return copy_from(y);
    }

private:
    static void copy_to_sve(const svbool_t& k, bool_arr& b) {
        svuint64_t a = svdup_u64_z(k, 1);
        svst1b_u64(svptrue_b64(), b.data(), a);
    }

    static svbool_t copy_from_sve(const bool_arr& p) {
        svuint64_t a = svld1ub_u64(svptrue_b64(), p.data());
        svuint64_t ones = svdup_n_u64(1);
        return svcmpeq_u64(svptrue_b64(), a, ones);
    }
};

struct sve_int8: implbase<sve_int8> {
    // Use default implementations for:
    //     element, set_element.

    using implbase<sve_int8>::gather;
    using implbase<sve_int8>::scatter;
    using implbase<sve_int8>::cast_from;

    using int32 = std::int32_t;

    using bool_arr = std::array<uint8_t, 8>;
    using int32_arr = std::array<int32, 8>;

    static void copy_to(const int32_arr& v, int32* p) {
        std::copy(std::begin(v), std::end(v), p);
    }

    static void copy_to_masked(const int32_arr& v, int32* p, const bool_arr& mask) {
        sve_mask = sve_mask8::copy_from_sve(mask);
        sve_v = copy_from_sve(v);
        svst1w_s64(sve_mask, p, sve_v);
    }

    static int32_arr copy_from(const int32* p) {
        int32_arr r;
        std::copy(p, p+r.size(), std::begin(r));
        return r;
    }

    static int32_arr copy_from_masked(const int32* p, const bool_arr& mask) {
        sve_mask = sve_mask8::copy_from_sve(mask);
        int32_arr r;
        copy_to_sve(svld1sw_s64(sve_mask, p), r);
        return r;
    }

    static int32_arr copy_from_masked(const int32_arr& v, const int32* p, const bool_arr& mask) {
        sve_mask = sve_mask8::copy_from_sve(mask);
        sve_v = copy_from_sve(v);
        int32_arr r;
        copy_to_sve(svsel_s64(mask, svld1sw_s64(sve_mask, p), sve_v), r);
        return r;
    }

    /*static svint64_t broadcast(int32 v) {
        return svreinterpret_s64_s32(svdup_n_s32(v));
    }

        return svld1sw_s64(mask, p);
    }

    static int32_arr copy_from_masked(const int32_arr& v, const int32* p, const bool_arr& mask) {
        return svsel_s64(mask, svld1sw_s64(mask, p), v);
    }

    /*static svint64_t broadcast(int32 v) {
        return svreinterpret_s64_s32(svdup_n_s32(v));
    }

    static int element0(const svint64_t& a) {
        return svlasta_s64(svptrue_b64(), a);
    }

    static svint64_t negate(const svint64_t& a) {
        return svneg_s64_z(svptrue_b64(), a);
    }

    static svint64_t add(const svint64_t& a, const svint64_t& b) {
        return svadd_s64_z(svptrue_b64(), a, b);
    }

    static svint64_t sub(const svint64_t& a, const svint64_t& b) {
        return svsub_s64_m(svptrue_b64(), a, b);
    }

    static svint64_t mul(const svint64_t& a, const svint64_t& b) {
        //May overflow
        return svmul_s64_z(svptrue_b64(), a, b);
    }

    static svint64_t div(const svint64_t& a, const svint64_t& b) {
        return svdiv_s64_z(svptrue_b64(), a, b);
    }

    static svint64_t fma(const svint64_t& a, const svint64_t& b, const svint64_t& c) {
        return add(mul(a, b), c);
    }

    static svbool_t cmp_eq(const svint64_t& a, const svint64_t& b) {
        return svcmpeq_s64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_neq(const svint64_t& a, const svint64_t& b) {
        return svcmpne_s64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_gt(const svint64_t& a, const svint64_t& b) {
        return svcmpgt_s64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_geq(const svint64_t& a, const svint64_t& b) {
        return svcmpge_s64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_lt(const svint64_t& a, const svint64_t& b) {
        return svcmplt_s64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_leq(const svint64_t& a, const svint64_t& b) {
        return svcmple_s64(svptrue_b64(), a, b);
    }

    static svint64_t ifelse(const svbool_t& m, const svint64_t& u, const svint64_t& v) {
        return svsel_s64(m, u, v);
    }

    static svint64_t max(const svint64_t& a, const svint64_t& b) {
        return svmax_s64_x(svptrue_b64(), a, b);
    }

    static svint64_t min(const svint64_t& a, const svint64_t& b) {
        return svmin_s64_x(svptrue_b64(), a, b);
    }

    static svint64_t abs(const svint64_t& a) {
        return svabs_s64_z(svptrue_b64(), a);
    }

    static int reduce_add(const svint64_t& a) {
        return svaddv_s64(svptrue_b64(), a);
    }

    static svint64_t gather(tag<sve_int8>, const int32* p, const svint64_t& index) {
        return svld1sw_gather_s64index_s64(svptrue_b64(), p, index);
    }

    static svint64_t gather(tag<sve_int8>, svint64_t a, const int32* p, const svint64_t& index, const svbool_t& mask) {
        return svsel_s64(mask, svld1sw_gather_s64index_s64(mask, p, index), a);
    }

    static void scatter(tag<sve_int8>, const svint64_t& s, int32* p, const svint64_t& index) {
        svst1w_scatter_s64index_s64(svptrue_b64(), p, index, s);
    }

    static void scatter(tag<sve_int8>, const svint64_t& s, int32* p, const svint64_t& index, const svbool_t& mask) {
        svst1w_scatter_s64index_s64(mask, p, index, s);
    }*/

private:
    static void copy_to_sve(const svint64_t& v, int32_arr&p) {
        svst1w_s64(svptrue_b64(), p.data(), v);
    }

    static svint64_t copy_from_sve(const int32_arr& p) {
        return svld1sw_s64(svptrue_b64(), p.data());
    }
};

struct sve_double8: implbase<sve_double8> {
    // Use default implementations for:
    //     element, set_element.

    using implbase<sve_double8>::gather;
    using implbase<sve_double8>::scatter;
    using implbase<sve_double8>::cast_from;

    using bool_arr = std::array<uint8_t, 8>;
    using int32_arr = std::array<int32, 8>;
    using double_arr = std::array<double, 8>;

    static void copy_to(const double_arr& v, double* p) {
        std::copy(std::begin(v), std::end(v), p);
    }

    static void copy_to_masked(const double_arr& v, double* p, const svbool_t& mask) {
        sve_mask = sve_mask8::copy_from_sve(mask);
        sve_v = copy_from_sve(v);
        svst1_f64(sve_mask, p, sve_v);
    }

    static double_arr copy_from(const double* p) {
        double_arr r;
        std::copy(p, p+r.size(), std::begin(r));
        return r;
    }

    static double_arr copy_from_masked(const double* p, const svbool_t& mask) {
        sve_mask = sve_mask8::copy_from_sve(mask);
        double_arr r;
        copy_to_sve(svld1_f64(sve_mask, p),r);
        return r;
    }

    static double_arr copy_from_masked(const double_arr& v, const double* p, const svbool_t& mask) {
        sve_mask = sve_mask8::copy_from_sve(mask);
        sve_v = copy_from_sve(v);
        double_arr r;
        copy_to_sve(svsel_f64(mask, svld1_f64(mask, p), v), r);
        return r;
    }

    /*static svfloat64_t broadcast(double v) {
        return svdup_n_f64(v);
    }

    static double element0(const svfloat64_t& a) {
        return svlasta_f64(svptrue_b64(), a);
    }

    static svfloat64_t negate(const svfloat64_t& a) {
        return svneg_f64_z(svptrue_b64(), a);
    }

    static svfloat64_t add(const svfloat64_t& a, const svfloat64_t& b) {
        return svadd_f64_z(svptrue_b64(), a, b);
    }

    static svfloat64_t sub(const svfloat64_t& a, const svfloat64_t& b) {
        return svsub_f64_m(svptrue_b64(), a, b);
    }

    static svfloat64_t mul(const svfloat64_t& a, const svfloat64_t& b) {
        return svmul_f64_z(svptrue_b64(), a, b);
    }

    static svfloat64_t div(const svfloat64_t& a, const svfloat64_t& b) {
        return svdiv_f64_z(svptrue_b64(), a, b);
    }

    static svfloat64_t fma(const svfloat64_t& a, const svfloat64_t& b, const svfloat64_t& c) {
        return svmad_f64_z(svptrue_b64(), a, b, c);
    }

    static svbool_t cmp_eq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpeq_f64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_neq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpne_f64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_gt(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpgt_f64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_geq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpge_f64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_lt(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmplt_f64(svptrue_b64(), a, b);
    }

    static svbool_t cmp_leq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmple_f64(svptrue_b64(), a, b);
    }

    static svfloat64_t ifelse(const svbool_t& m, const svfloat64_t& u, const svfloat64_t& v) {
        return svsel_f64(m, u, v);
    }

    static svfloat64_t max(const svfloat64_t& a, const svfloat64_t& b) {
        return svmax_f64_x(svptrue_b64(), a, b);
    }

    static svfloat64_t min(const svfloat64_t& a, const svfloat64_t& b) {
        return svmin_f64_x(svptrue_b64(), a, b);
    }

    static svfloat64_t abs(const svfloat64_t& x) {
        return svabs_f64_x(svptrue_b64(), x);
    }

    static double reduce_add(const svfloat64_t& a) {
        return svaddv_f64(svptrue_b64(), a);
    }

    static svfloat64_t gather(tag<sve_int8>, const double* p, const svint64_t& index) {
        return svld1_gather_s64index_f64(svptrue_b64(), p, index);
    }

    static svfloat64_t gather(tag<sve_int8>, svfloat64_t a, const double* p, const svint64_t& index, const svbool_t& mask) {
        return svsel_f64(mask, svld1_gather_s64index_f64(mask, p, index), a);
    }

    static void scatter(tag<sve_int8>, const svfloat64_t& s, double* p, const svint64_t& index) {
        svst1_scatter_s64index_f64(svptrue_b64(), p, index, s);
    }

    static void scatter(tag<sve_int8>, const svfloat64_t& s, double* p, const svint64_t& index, const svbool_t& mask) {
        svst1_scatter_s64index_f64(mask, p, index, s);
    }

    // Refer to avx/avx2 code for details of the exponential and log
    // implementations.

    static  svfloat64_t exp(const svfloat64_t& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));

        // Compute n and g.

        auto n = svrintz_f64_z(svptrue_b64(), add(mul(broadcast(ln2inv), x), broadcast(0.5)));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).
        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = fma(broadcast(2), div(odd, sub(even, odd)), broadcast(1));

        // Scale by 2^n, propogating NANs.

        auto result = svscale_f64_z(svptrue_b64(), expg, svcvt_s64_f64_z(svptrue_b64(), n));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(0),
                   result));
    }

    static  svfloat64_t expm1(const svfloat64_t& x) {
        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(expm1_minarg));

        auto half = broadcast(0.5);
        auto one = broadcast(1.);

        auto nnz = cmp_gt(abs(x), half);
        svfloat64_t svrinta_f64_z(svbool_t pg, svfloat64_t op);
        auto n = svrinta_f64_z(nnz, mul(broadcast(ln2inv), x));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) -1 = 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expgm1 = div(mul(broadcast(2), odd), sub(even, odd));

        // For small x (n zero), bypass scaling step to avoid underflow.
        // Otherwise, compute result 2^n * expgm1 + (2^n-1) by:
        //     result = 2 * ( 2^(n-1)*expgm1 + (2^(n-1)+0.5) )
        // to avoid overflow when n=1024.

        auto nm1 = svcvt_s64_f64_z(svptrue_b64(), sub(n, one));

        auto result =
            svscale_f64_z(svptrue_b64(),
                add(sub(svscale_f64_z(svptrue_b64(),one, nm1), half),
                    svscale_f64_z(svptrue_b64(),expgm1, nm1)),
                svcvt_s64_f64_z(svptrue_b64(), one));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-1),
            ifelse(nnz, result, expgm1)));
    }

    static svfloat64_t log(const svfloat64_t& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        is_small = sve_mask8::logical_and(is_small, cmp_geq(x, broadcast(0)));

        svfloat64_t g = svcvt_f64_s64_z(svptrue_b64(), logb_normal(x));
        svfloat64_t u = fraction_normal(x);

        svfloat64_t one = broadcast(1.);
        svfloat64_t half = broadcast(0.5);
        auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
        g = ifelse(gtsqrt2, add(g, one), g);
        u = ifelse(gtsqrt2, mul(u, half), u);

        auto z = sub(u, one);
        auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
        auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

        auto z2 = mul(z, z);
        auto z3 = mul(z2, z);

        auto r = div(mul(z3, pz), qz);
        r = fma(g,  broadcast(ln2C4), r);
        r = fms(z2, half, r);
        r = sub(z, r);
        r = fma(g,  broadcast(ln2C3), r);

        // r is alrady NaN if x is NaN or negative, otherwise
        // return  +inf if x is +inf, or -inf if zero or (positive) denormal.

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-HUGE_VAL),
                r));
    }

protected:
    // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite and normal.
    static svint64_t logb_normal(const svfloat64_t& x) {
        svuint64_t xw    = svunpkhi_u64(svreinterpret_u32_f64(x));
        svuint64_t emask = svunpkhi_u64(svdup_n_u32(0x7ff00000));
        svuint64_t ebiased = svlsr_n_u64_z(svptrue_b64(), svand_u64_z(svptrue_b64(), xw, emask), 20);

        return svsub_s64_z(svptrue_b64(), svreinterpret_s64_u64(ebiased), svunpkhi_s64(svdup_n_s32(1023)));
    }

    static svfloat64_t fraction_normal(const svfloat64_t& x) {
        svuint64_t emask = svdup_n_u64(0x800fffffffffffff);
        svuint64_t bias =  svdup_n_u64(0x3ff0000000000000);
        return svreinterpret_f64_u64(
            svorr_u64_z(svptrue_b64(), bias, svand_u64_z(svptrue_b64(), emask, svreinterpret_u64_f64(x))));
    }

    static inline svfloat64_t horner1(svfloat64_t x, double a0) {
        return add(x, broadcast(a0));
    }

    static inline svfloat64_t horner(svfloat64_t x, double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static svfloat64_t horner(svfloat64_t x, double a0, T... tail) {
        return fma(x, horner(x, tail...), broadcast(a0));
    }

    template <typename... T>
    static svfloat64_t horner1(svfloat64_t x, double a0, T... tail) {
        return fma(x, horner1(x, tail...), broadcast(a0));
    }

    static svfloat64_t fms(const svfloat64_t& a, const svfloat64_t& b, const svfloat64_t& c) {
        return svnmsb_f64_z(svptrue_b64(), a, b, c);
    }*/

private:
    static void copy_to_sve(const svfloat64_t& v, double_arr& p) {
        svst1_f64(svptrue_b64(), p.data(), v);
    }

    static svfloat64_t copy_from_sve(const double_arr& p) {
        return svld1_f64(svptrue_b64(), p.data());
    }
};


}  // namespace detail

namespace simd_abi {
template <typename T, unsigned N>
struct sve;

template <>
struct sve<double, 8> {
    using type = detail::sve_double8;
};
template <>
struct sve<int, 8> {
    using type = detail::sve_int8;
};

}  // namespace simd_abi

}  // namespace simd
}  // namespace arb

#endif  // def __ARM_FEATURE_SVE
