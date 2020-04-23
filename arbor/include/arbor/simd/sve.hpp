#pragma once

// SVE SIMD intrinsics implementation.

#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>
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
    using vector_type = svbool_t;
    using mask_impl = sve_mak8;
};

template <>
struct simd_traits<sve_double8> {
    static constexpr unsigned width = 8;
    using scalar_type = double;
    using vector_type = svfloat64_t;
    using mask_impl = sve_mask8;
};

template <>
struct simd_traits<sve_int8> {
    static constexpr unsigned width = 8;
    using scalar_type = int32_t;
    using vector_type = svint64_t;
    using mask_impl = sve_mask8;
};

struct sve_mask8: implbase<sve_mask8> {
    using implbase<sve_mask8>::gather;
    using implbase<sve_mask8>::scatter;
    using implbase<sve_mask8>::cast_from;

    static svbool_t broadcast(bool b) {
        return svdup_b64(-b);
    }

    static void copy_to(const svbool_t& k, bool* b) {
        svuint64_t a = svdup_u64_z(k, 1);
        svst1b_u64(true_pred, reinterpret_cast<uint8_t*>(b), a);
    }

    static svbool_t copy_from(const bool* p) {
        svuint64_t a = svld1ub_u64(true_pred, reinterpret_cast<const uint8_t*>(p));
        svuint64_t ones = svdup_n_u64(1);
        return svcmpeq_u64(true_pred, a, ones);
    }

    static svbool_t logical_not(const svbool_t& k) {
        return svnot_b_z(true_pred, k);
    }

    static svbool_t logical_and(const svbool_t& a, const svbool_t& b) {
        return svand_b_z(true_pred, a, b);
    }

    static svbool_t logical_or(const svbool_t& a, const svbool_t& b) {
        return svorr_b_z(true_pred, a, b);
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

    static svbool_t negate(const svbool_t& a) {
        return a;
    }

    static svbool_t add(const svbool_t& a, const svbool_t& b) {
        return sveor_b_z(true_pred, a, b);
    }

    static svbool_t sub(const svbool_t& a, const svbool_t& b) {
        return sveor_b_z(true_pred, a, b);
    }

    static svbool_t mul(const svbool_t& a, const svbool_t& b) {
        return svand_b_z(true_pred, a, b);
    }

    static svbool_t div(const svbool_t& a, const svbool_t& b) {
        return a;
    }

    static svbool_t fma(const svbool_t& a, const svbool_t& b, const svbool_t& c) {
        return add(mul(a, b), c);
    }

    static svbool_t max(const svbool_t& a, const svbool_t& b) {
        return svorr_b_z(true_pred, a, b);
    }

    static svbool_t min(const svbool_t& a, const svbool_t& b) {
        return svand_b_z(true_pred, a, b);
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

    static svbool_t cmp_eq(const svbool_t& a, const svbool_t& b) {
        return svnot_b_z(true_pred, sveor_b_z(true_pred, a, b));
    }

    static svbool_t cmp_neq(const svbool_t& a, const svbool_t& b) {
        return sveor_b_z(true_pred, a, b);
    }

    static svbool_t cmp_lt(const svbool_t& a, const svbool_t& b) {
        return svbic_b_z(true_pred, b, a);
    }

    static svbool_t cmp_gt(const svbool_t& a, const svbool_t& b) {
        return cmp_lt(b, a);
    }

    static svbool_t cmp_geq(const svbool_t& a, const svbool_t& b) {
        return logical_not(cmp_lt(a, b));
    }

    static svbool_t cmp_leq(const svbool_t& a, const svbool_t& b) {
        return logical_not(cmp_gt(a, b));
    }

    static svbool_t ifelse(const svbool_t& m, const svbool_t& u, const svbool_t& v) {
        return svsel_b(m, u, v);
    }

    static svbool_t mask_broadcast(bool b) {
        return broadcast(b);
    }

    static bool mask_element(const svbool_t& u, int i) {
        return element(u, i);
    }

    static void mask_set_element(svbool_t& u, int i, bool b) {
        set_element(u, i, b);
    }

    static void mask_copy_to(const svbool_t& m, bool* y) {
        copy_to(m, y);
    }

    static svbool_t mask_copy_from(const bool* y) {
        return copy_from(y);
    }

private:
    svbool_t true_pred = svptrue_b64();
    svbool_t false_pred = svpfalse_b64();
};

struct sve_int8: implbase<sve_int8> {
    // Use default implementations for:
    //     element, set_element.

    using implbase<sve_int8>::gather;
    using implbase<sve_int8>::scatter;
    using implbase<sve_int8>::cast_from;

    using int32 = std::int32_t;

    static svint64_t broadcast(int32 v) {
        return svreinterpret_s64_s32(svdup_n_s32(v));
    }

    static void copy_to(const svint64_t& v, int32* p) {
        svst1w_s64(true_pred, p, v);
    }

    static void copy_to_masked(const svint64_t& v, int32* p, const svbool_t& mask) {
        svst1w_s64(mask, p, v);
    }

    static svint64_t copy_from(const int32* p) {
        return svld1sw_s64(true_pred, p);
    }

    static svint64_t copy_from_masked(const int32* p, const svbool_t& mask) {
        return svld1sw_s64(mask, p);
    }

    static svint64_t copy_from_masked(const svint64_t& v, const int32* p, const svbool_t& mask) {
        return svsel_s64(mask, svld1sw_s64(mask, p), v);
    }

    static int element0(const svint64_t& a) {
        return svlasta_s64(true_pred, a);
    }

    static svint64_t negate(const svint64_t& a) {
        return svneg_s64_z(true_pred, a);
    }

    static svint64_t add(const svint64_t& a, const svint64_t& b) {
        return svadd_s64_z(true_pred, a, b);
    }

    static svint64_t sub(const svint64_t& a, const svint64_t& b) {
        return svsub_s64_m(true_pred, a, b);
    }

    static svint64_t mul(const svint64_t& a, const svint64_t& b) {
        //May overflow
        return svmul_s64_z(true_pred, a, b);
    }

    static svint64_t div(const svint64_t& a, const svint64_t& b) {
        return svdiv_s64_z(true_pred, a, b);
    }

    static svint64_t fma(const svint64_t& a, const svint64_t& b, const svint64_t& c) {
        return add(mul(a, b), c);
    }

    static svbool_t cmp_eq(const svint64_t& a, const svint64_t& b) {
        return svcmpeq_s64(true_pred, a, b);
    }

    static svbool_t cmp_neq(const svint64_t& a, const svint64_t& b) {
        return svcmpne_s64(true_pred, a, b);
    }

    static svbool_t cmp_gt(const svint64_t& a, const svint64_t& b) {
        return svcmpgt_s64(true_pred, a, b);
    }

    static svbool_t cmp_geq(const svint64_t& a, const svint64_t& b) {
        return svcmpge_s64(true_pred, a, b);
    }

    static svbool_t cmp_lt(const svint64_t& a, const svint64_t& b) {
        return svcmplt_s64(true_pred, a, b);
    }

    static svbool_t cmp_leq(const svint64_t& a, const svint64_t& b) {
        return svcmple_s64(true_pred, a, b);
    }

    static svint64_t ifelse(const svbool_t& m, const svint64_t& u, const svint64_t& v) {
        return svsel_s64(m, u, v);
    }

    static svint64_t max(const svint64_t& a, const svint64_t& b) {
        return svmax_s64_x(true_pred, a, b);
    }

    static svint64_t min(const svint64_t& a, const svint64_t& b) {
        return svmin_s64_x(true_pred, a, b);
    }

    static svint64_t abs(const svint64_t& a) {
        return svabs_s64_z(true_pred, a);
    }

    static int reduce_add(const svint64_t& a) {
        return svaddv_s64(true_pred, a);
    }

    static svint64_t gather(tag<sve_int8>, const int32* p, const svint64_t& index) {
        return svld1sw_gather_s64index_s64(true_pred, p, index);
    }

    static svint64_t gather(tag<sve_int8>, svint64_t a, const int32* p, const svint64_t& index, const svbool_t& mask) {
        return svsel_s64(mask, svld1sw_gather_s64index_s64(mask, p, index), a);
    }

    static void scatter(tag<sve_int8>, const svint64_t& s, int32* p, const svint64_t& index) {
        svst1w_scatter_s64index_s64(true_pred, p, index, s);
    }

    static void scatter(tag<sve_int8>, const svint64_t& s, int32* p, const svint64_t& index, const svbool_t& mask) {
        svst1s_scatter_s64index_s64(mask, p, index, s);
    }

private:
    svbool_t true_pred = svptrue_b64();
    svbool_t false_pred = svpfalse_b64();
};

struct sve_double8: implbase<sve_double8> {
    // Use default implementations for:
    //     element, set_element.

    using implbase<sve_double8>::gather;
    using implbase<sve_double8>::scatter;
    using implbase<sve_double8>::cast_from;

    static svfloat64_t broadcast(double v) {
        return svdup_n_f64(v);
    }

    static void copy_to(const svfloat64_t& v, double* p) {
        svst1w_f64(true_pred, p, v);
    }

    static void copy_to_masked(const svfloat64_t& v, double* p, const svbool_t& mask) {
        svst1w_f64(mask, p, v);
    }

    static svfloat64_t copy_from(const double* p) {
        return svld1sw_f64(true_pred, p);
    }

    static svfloat64_t copy_from_masked(const double* p, const svbool_t& mask) {
        return svld1sw_f64(mask, p);
    }

    static svfloat64_t copy_from_masked(const svfloat64_t& v, const double* p, const svbool_t& mask) {
        return svsel_f64(mask, svld1sw_f64(mask, p), v);
    }

    static double element0(const svfloat64_t& a) {
        return svlasta_f64(true_pred, a);
    }

    static svfloat64_t negate(const svfloat64_t& a) {
        return svneg_f64_z(true_pred, a);
    }

    static svfloat64_t add(const svfloat64_t& a, const svfloat64_t& b) {
        return svadd_f64_z(true_pred, a, b);
    }

    static svfloat64_t sub(const svfloat64_t& a, const svfloat64_t& b) {
        return svsub_f64_m(true_pred, a, b);
    }

    static svfloat64_t mul(const svfloat64_t& a, const svfloat64_t& b) {
        return svmul_f64_z(true_pred, a, b);
    }

    static svfloat64_t div(const svfloat64_t& a, const svfloat64_t& b) {
        return svdiv_f64_z(true_pred, a, b);
    }

    static svfloat64_t fma(const svfloat64_t& a, const svfloat64_t& b, const svfloat64_t& c) {
        return svmad_f64_z(true_pred, a, b, c);
    }

    static svbool_t cmp_eq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpeq_f64(true_pred, a, b);
    }

    static svbool_t cmp_neq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpne_f64(true_pred, a, b);
    }

    static svbool_t cmp_gt(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpgt_f64(true_pred, a, b);
    }

    static svbool_t cmp_geq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmpge_f64(true_pred, a, b);
    }

    static svbool_t cmp_lt(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmplt_f64(true_pred, a, b);
    }

    static svbool_t cmp_leq(const svfloat64_t& a, const svfloat64_t& b) {
        return svcmple_f64(true_pred, a, b);
    }

    static svfloat64_t ifelse(const svbool_t& m, const svfloat64_t& u, const svfloat64_t& v) {
        return svsel_f64(m, u, v);
    }

    static svfloat64_t max(const svfloat64_t& a, const svfloat64_t& b) {
        return svmax_s64_x(true_pred, a, b);
    }

    static svfloat64_t min(const svfloat64_t& a, const svfloat64_t& b) {
        return svmin_s64_x(true_pred, a, b);
    }

    static svfloat64_t abs(const svfloat64_t& x) {
        return svabs_s64_x(true_pred, a, b);
    }

    static double reduce_add(const svfloat64_t& a) {
        return svaddv_f64(true_pred, a);
    }

    static svfloat64_t gather(tag<sve_int8>, const double* p, const svint64_t& index) {
        return svld1_gather_s64index_f64(true_pred, p, index);
    }

    static svfloat64_t gather(tag<sve_int8>, svfloat64_t a, const double* p, const svint64_t& index, const svbool_t& mask) {
        return svsel_f64(mask, svld1_gather_s64index_f64(mask, p, index), a);
    }

    static void scatter(tag<sve_int8>, const svfloat64_t& s, double* p, const svint64_t& index) {
        svst1_scatter_s64index_f64(true_pred, p, index, s);
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

        auto n = svcvt_s64_f64_z(true_pred, add(mul(broadcast(ln2inv), x), broadcast(0.5))));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).
        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = fma(broadcast(2), div(odd, sub(even, odd)), broadcast(1));

        // Scale by 2^n, propogating NANs.

        auto result = svscale_f64_z(true_pred, expg, n);

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
        svfloat64_t svrinta_f64_z(svbool_t pg, svfloat64_t op)
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

        auto nm1 = sub(n, one);

        auto result =
            svscale_f64_z(true_pred,
                add(sub(svscale_f64_z(true_pred,one, nm1), half),
                    svscale_f64_z(true_pred,expgm1, nm1)),
                one);

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

        svfloat64_t g = svcvt_f64_s64_z(true_pred, logb_normal(x));
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
#endif

protected:
    // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite and normal.
    static svint64_t logb_normal(const svfloat64_t& x) {
        svuint64_t xw    = svunpkhi_u64(svreinterpret_u32_f64(x));
        svuint64_t emask = svunpkhi_u64(svdup_n_u32(0x7ff00000));
        svuint64_t ebiased = svlsr_n_u64_z(true_pred, svand_u64_z(true_pred, xw, emask), 20);

        return svsub_s64_z(true_pred, svreinterpret_s64_u64(ebiased), svunpkhi_s64(svdup_n_s32(1023)));
    }

    static svfloat64_t fraction_normal(const svfloat64_t& x) {
        svuint64_t emask = svdup_n_u64(0x800fffffffffffff);
        svuint64_t bias =  svdup_n_u64(0x3ff0000000000000);
        return svreinterpretq_f64_u64(
            svorr_u64_z(true_pred, bias, svand_u64_z(true_pred, emask, vreinterpretq_u64_f64(x))));
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
        return svnmsb_f64_z(true_pred, a, b, c);
    }

private:
    svbool_t true_pred = svptrue_b64();
    svbool_t false_pred = svpfalse_b64();
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
