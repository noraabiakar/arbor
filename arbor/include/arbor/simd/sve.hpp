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
    using vector_type = svint32_t;
    using mask_impl = sve_mask8;
};

struct avx512_mask8: implbase<avx512_mask8> {
    using implbase<sve_mask8>::gather;
    using implbase<sve_mask8>::scatter;
    using implbase<sve_mask8>::cast_from;

    static svbool_t broadcast(bool b) {
        return svdup_b64(-b);
    }

    static void copy_to(const svbool_t& k, bool* b) {
        uint64_t c[8];
        svuint64_t a = svdup_u64_z(k, 1);
        svst1(true_pred, c, a);
        for (unsigned i =0; i< 8; i++) {
	        b[i] = (bool)c[i];
        }
    }

    static svbool_t copy_from(const bool* p) {
        // copy p into vector of uint64 a
        uint64_t mask[8];
        for (unsigned i =0; i< 8; i++) {
            mask[i] = (uint64_t)p[i];
        }
        svuint64_t a = svld1(true_pred, mask);

        // Create vector of ones
        svuint64_t ones = svdup_n_u64(1);

        return svcmpeq_u64(true_pred, a, ones);

    }

    // Note: fall back to implbase implementations of copy_to_masked and copy_from_masked;
    // could be improved with the use of AVX512BW instructions on supported platforms.

    static svbool_t logical_not(const svbool_t& k) {
        return _mm512_knot(k);
    }

    static svbool_t logical_and(const svbool_t& a, const svbool_t& b) {
        return _mm512_kand(a, b);
    }

    static svbool_t logical_or(const svbool_t& a, const svbool_t& b) {
        return _mm512_kor(a, b);
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
        return _mm512_kxor(a, b);
    }

    static svbool_t sub(const svbool_t& a, const svbool_t& b) {
        return _mm512_kxor(a, b);
    }

    static svbool_t mul(const svbool_t& a, const svbool_t& b) {
        return _mm512_kand(a, b);
    }

    static svbool_t div(const svbool_t& a, const svbool_t& b) {
        return a;
    }

    static svbool_t fma(const svbool_t& a, const svbool_t& b, const svbool_t& c) {
        return add(mul(a, b), c);
    }

    static svbool_t max(const svbool_t& a, const svbool_t& b) {
        return _mm512_kor(a, b);
    }

    static svbool_t min(const svbool_t& a, const svbool_t& b) {
        return _mm512_kand(a, b);
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
        return _mm512_kxnor(a, b);
    }

    static svbool_t cmp_neq(const svbool_t& a, const svbool_t& b) {
        return _mm512_kxor(a, b);
    }

    static svbool_t cmp_lt(const svbool_t& a, const svbool_t& b) {
        return _mm512_kandn(a, b);
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
        return _mm512_kor(_mm512_kandn(m, u), _mm512_kand(m, v));
    }

    static bool element(const svbool_t& k, int i) {
        return _mm512_mask2int(k)&(1<<i);
    }

    static void set_element(svbool_t& k, int i, bool b) {
        int n = _mm512_mask2int(k);
        k = _mm512_int2mask((n&~(1<<i))|(b<<i));
    }

    static svbool_t mask_broadcast(bool b) {
        return broadcast(b);
    }

    static svbool_t mask_unpack(unsigned long long p) {
        return _mm512_int2mask(p);
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
};


struct sve_int8 : implbase<sve_int8> {
    // Use default implementations for:
    //     element, set_element, div.

    using int32 = std::int32_t;

    //static svint32_t broadcast(int32 v) { return vdup_n_s32(v); }

    static void copy_to(const svint32_t& v, int32* p) { svst1(true_pred, p, v); }

    /*static void copy_to_masked(const svint32_t& v, int32* p,
                               const svint32_t& mask) {
        svint32_t r = vld1_s32(p);
        r = vbsl_s32(vreinterpret_u32_s32(mask), v, r);
        vst1_s32(p, r);
    }*/

    static svint32_t copy_from(const int32* p) { return svld1(true_pred, p); }

    /*static svint32_t copy_from_masked(const int32* p, const svint32_t& mask) {
        svint32_t a = {};
        svint32_t r = vld1_s32(p);
        a = vbsl_s32(vreinterpret_u32_s32(mask), r, a);
        return a;
    }

    static svint32_t copy_from_masked(const svint32_t& v, const int32* p,
                                      const svint32_t& mask) {
        svint32_t a;
        svint32_t r = vld1_s32(p);
        a = vbsl_s32(vreinterpret_u32_s32(mask), r, v);
        return a;
    }

    static svint32_t negate(const svint32_t& a) { return vneg_s32(a); }

    static svint32_t add(const svint32_t& a, const svint32_t& b) {
        return vadd_s32(a, b);
    }

    static svint32_t sub(const svint32_t& a, const svint32_t& b) {
        return vsub_s32(a, b);
    }

    static svint32_t mul(const svint32_t& a, const svint32_t& b) {
        return vmul_s32(a, b);
    }

    static svint32_t logical_not(const svint32_t& a) { return vmvn_s32(a); }

    static svint32_t logical_and(const svint32_t& a, const svint32_t& b) {
        return vand_s32(a, b);
    }

    static svint32_t logical_or(const svint32_t& a, const svint32_t& b) {
        return vorr_s32(a, b);
    }

    static svint32_t cmp_eq(const svint32_t& a, const svint32_t& b) {
        return vreinterpret_s32_u32(vceq_s32(a, b));
    }

    static svint32_t cmp_neq(const svint32_t& a, const svint32_t& b) {
        return logical_not(cmp_eq(a, b));
    }

    static svint32_t cmp_gt(const svint32_t& a, const svint32_t& b) {
        return vreinterpret_s32_u32(vcgt_s32(a, b));
    }

    static svint32_t cmp_geq(const svint32_t& a, const svint32_t& b) {
        return vreinterpret_s32_u32(vcge_s32(a, b));
    }

    static svint32_t cmp_lt(const svint32_t& a, const svint32_t& b) {
        return vreinterpret_s32_u32(vclt_s32(a, b));
    }

    static svint32_t cmp_leq(const svint32_t& a, const svint32_t& b) {
        return vreinterpret_s32_u32(vcle_s32(a, b));
    }

    static svint32_t ifelse(const svint32_t& m, const svint32_t& u,
                            const svint32_t& v) {
        return vbsl_s32(vreinterpret_u32_s32(m), u, v);
    }

    static svint32_t mask_broadcast(bool b) {
        return vreinterpret_s32_u32(vdup_n_u32(-(int32)b));
    }*/

    static bool mask_element(const svint32_t& u, int i) {
        svint32_t dup_vec = svdup_lane_s32(u, i);
        return static_cast<bool> svlasta_s32(true_vec, dup_vec);
    }

    /*static svint32_t mask_unpack(unsigned long long k) {
        // Only care about bottom two bits of k.
        uint8x8_t b = vdup_n_u8((char)k);
        uint8x8_t bl = vorr_u8(b, vdup_n_u8(0xfe));
        uint8x8_t bu = vorr_u8(b, vdup_n_u8(0xfd));
        uint8x16_t blu = vcombine_u8(bl, bu);

        uint8x16_t ones = vdupq_n_u8(0xff);
        uint64x2_t r =
            vceqq_u64(vreinterpretq_u64_u8(ones), vreinterpretq_u64_u8(blu));

        return vreinterpret_s32_u32(vmovn_u64(r));
    }*/

    static void mask_set_element(svint32_t& u, int i, bool b) {
        char data[64];
        vst1_s32((int32*)data, u);
        ((int32*)data)[i] = -(int32)b;
        u = vld1_s32((int32*)data);
    }

    static void mask_copy_to(const svint32_t& m, bool* y) {
        // Negate (convert 0xffffffff to 0x00000001) and move low bytes to
        // bottom 2 bytes.

        int64x1_t ml = vreinterpret_s64_s32(vneg_s32(m));
        int64x1_t mh = vshr_n_s64(ml, 24);
        ml = vorr_s64(ml, mh);
        std::memcpy(y, &ml, 2);
    }

    static svint32_t mask_copy_from(const bool* w) {
        // Move bytes:
        //   rl: byte 0 to byte 0, byte 1 to byte 8, zero elsewhere.
        //
        // Subtract from zero to translate
        // 0x0000000000000001 to 0xffffffffffffffff.

        int8_t a[16] = {0};
        std::memcpy(&a, w, 2);
        int8x8x2_t t = vld2_s8(a);  // intervleaved load
        int64x1_t rl = vreinterpret_s64_s8(t.val[0]);
        int64x1_t rh = vshl_n_s64(vreinterpret_s64_s8(t.val[1]), 32);
        int64x1_t rc = vadd_s64(rl, rh);
        return vneg_s32(vreinterpret_s32_s64(rc));
    }

    /*static svint32_t max(const svint32_t& a, const svint32_t& b) {
        return vmax_s32(a, b);
    }

    static svint32_t min(const svint32_t& a, const svint32_t& b) {
        return vmin_s32(a, b);
    }

    static svint32_t abs(const svint32_t& x) { return vabs_s32(x); }*/

private:
    svbool_t true_pred = svptrue_b64();
    svbool_t false_pred = svpfalse_b64();
};

struct sve_double8 : implbase<sve_double8> {
    // Use default implementations for:
    //     element, set_element.

    using int64 = std::int64_t;

    //static float64x2_t broadcast(double v) { return vdupq_n_f64(v); }

    static void copy_to(const float64x2_t& v, double* p) { vst1q_f64(p, v); }

    /*static void copy_to_masked(const float64x2_t& v, double* p,
                               const float64x2_t& mask) {
        float64x2_t r = vld1q_f64(p);
        r = vbslq_f64(vreinterpretq_u64_f64(mask), v, r);
        vst1q_f64(p, r);
    }*/

    static float64x2_t copy_from(const double* p) { return vld1q_f64(p); }

    /*static float64x2_t copy_from_masked(const double* p,
                                        const float64x2_t& mask) {
        float64x2_t a = {};
        float64x2_t r = vld1q_f64(p);
        a = vbslq_f64(vreinterpretq_u64_f64(mask), r, a);
        return a;
    }

    static float64x2_t copy_from_masked(const float64x2_t& v, const double* p,
                                        const float64x2_t& mask) {
        float64x2_t a;
        float64x2_t r = vld1q_f64(p);
        a = vbslq_f64(vreinterpretq_u64_f64(mask), r, v);
        return a;
    }

    static float64x2_t negate(const float64x2_t& a) { return vnegq_f64(a); }

    static float64x2_t add(const float64x2_t& a, const float64x2_t& b) {
        return vaddq_f64(a, b);
    }

    static float64x2_t sub(const float64x2_t& a, const float64x2_t& b) {
        return vsubq_f64(a, b);
    }

    static float64x2_t mul(const float64x2_t& a, const float64x2_t& b) {
        return vmulq_f64(a, b);
    }

    static float64x2_t div(const float64x2_t& a, const float64x2_t& b) {
        return vdivq_f64(a, b);
    }

    static float64x2_t logical_not(const float64x2_t& a) {
        return vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(a)));
    }

    static float64x2_t logical_and(const float64x2_t& a, const float64x2_t& b) {
        return vreinterpretq_f64_u64(
            vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
    }

    static float64x2_t logical_or(const float64x2_t& a, const float64x2_t& b) {
        return vreinterpretq_f64_u64(
            vorrq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
    }

    static float64x2_t cmp_eq(const float64x2_t& a, const float64x2_t& b) {
        return vreinterpretq_f64_u64(vceqq_f64(a, b));
    }

    static float64x2_t cmp_neq(const float64x2_t& a, const float64x2_t& b) {
        return logical_not(cmp_eq(a, b));
    }

    static float64x2_t cmp_gt(const float64x2_t& a, const float64x2_t& b) {
        return vreinterpretq_f64_u64(vcgtq_f64(a, b));
    }

    static float64x2_t cmp_geq(const float64x2_t& a, const float64x2_t& b) {
        return vreinterpretq_f64_u64(vcgeq_f64(a, b));
    }

    static float64x2_t cmp_lt(const float64x2_t& a, const float64x2_t& b) {
        return vreinterpretq_f64_u64(vcltq_f64(a, b));
    }

    static float64x2_t cmp_leq(const float64x2_t& a, const float64x2_t& b) {
        return vreinterpretq_f64_u64(vcleq_f64(a, b));
    }

    static float64x2_t ifelse(const float64x2_t& m, const float64x2_t& u,
                              const float64x2_t& v) {
        return vbslq_f64(vreinterpretq_u64_f64(m), u, v);
    }

    static float64x2_t mask_broadcast(bool b) {
        return vreinterpretq_f64_u64(vdupq_n_u64(-(int64)b));
    }*/

    static bool mask_element(const float64x2_t& u, int i) {
        return static_cast<bool>(element(u, i));
    }

    /*static float64x2_t mask_unpack(unsigned long long k) {
        // Only care about bottom two bits of k.
        uint8x8_t b = vdup_n_u8((char)k);
        uint8x8_t bl = vorr_u8(b, vdup_n_u8(0xfe));
        uint8x8_t bu = vorr_u8(b, vdup_n_u8(0xfd));
        uint8x16_t blu = vcombine_u8(bl, bu);

        uint8x16_t ones = vdupq_n_u8(0xff);
        uint64x2_t r =
            vceqq_u64(vreinterpretq_u64_u8(ones), vreinterpretq_u64_u8(blu));

        return vreinterpretq_f64_u64(r);
    }*/

    static void mask_set_element(float64x2_t& u, int i, bool b) {
        char data[256];
        vst1q_f64((double*)data, u);
        ((int64*)data)[i] = -(int64)b;
        u = vld1q_f64((double*)data);
    }

    static void mask_copy_to(const float64x2_t& m, bool* y) {
        // Negate (convert 0xffffffff to 0x00000001) and move low bytes to
        // bottom 2 bytes.

        int8x16_t mc = vnegq_s8(vreinterpretq_s8_f64(m));
        int8x8_t mh = vget_high_s8(mc);
        mh = vand_s8(mh, vreinterpret_s8_s64(vdup_n_s64(0x000000000000ff00)));
        int8x8_t ml = vget_low_s8(mc);
        ml = vand_s8(ml, vreinterpret_s8_s64(vdup_n_s64(0x00000000000000ff)));
        mh = vadd_s8(mh, ml);
        std::memcpy(y, &mh, 2);
    }

    static float64x2_t mask_copy_from(const bool* w) {
        // Move bytes:
        //   rl: byte 0 to byte 0, byte 1 to byte 8, zero elsewhere.
        //
        // Subtract from zero to translate
        // 0x0000000000000001 to 0xffffffffffffffff.

        int8_t a[16] = {0};
        std::memcpy(&a, w, 2);
        int8x8x2_t t = vld2_s8(a);  // intervleaved load
        int64x2_t r = vreinterpretq_s64_s8(vcombine_s8((t.val[0]), (t.val[1])));
        int64x2_t r2 = (vnegq_s64(r));
        return vreinterpretq_f64_s64(r2);
    }

    /*static float64x2_t max(const float64x2_t& a, const float64x2_t& b) {
        return vmaxnmq_f64(a, b);
    }

    static float64x2_t min(const float64x2_t& a, const float64x2_t& b) {
        return vminnmq_f64(a, b);
    }

    static float64x2_t abs(const float64x2_t& x) { return vabsq_f64(x); }*/

    // Exponential is calculated as follows:
    //
    //     e^x = e^g · 2^n,
    //
    // where g in [-0.5, 0.5) and n is an integer. 2^n can be
    // calculated via bit manipulation or specialized scaling intrinsics,
    // whereas e^g is approximated using the order-6 rational
    // approximation:
    //
    //     e^g = R(g)/R(-g)
    //
    // with R(x) split into even and odd terms:
    //
    //     R(x) = Q(x^2) + x·P(x^2)
    //
    // so that the ratio can be computed as:
    //
    //     e^g = 1 + 2·g·P(g^2) / (Q(g^2)-g·P(g^2)).
    //
    // Note that the coefficients for R are close to but not the same as those
    // from the 6,6 Padé approximant to the exponential.
    //
    // The exponents n and g are calculated by:
    //
    //     n = floor(x/ln(2) + 0.5)
    //     g = x - n·ln(2)
    //
    // so that x = g + n·ln(2). We have:
    //
    //     |g| = |x - n·ln(2)|
    //         = |x - x + α·ln(2)|
    //
    // for some fraction |α| ≤ 0.5, and thus |g| ≤ 0.5ln(2) ≈ 0.347.
    //
    // Tne subtraction x - n·ln(2) is performed in two parts, with
    // ln(2) = C1 + C2, in order to compensate for the possible loss of
    // precision
    // attributable to catastrophic rounding. C1 comprises the first
    // 32-bits of mantissa, C2 the remainder.

    /*static float64x2_t exp(const float64x2_t& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));
        auto is_not_nan = cmp_eq(x, x);

        // Compute n and g.

        // floor: round toward negative infinity
        auto n = vcvtmq_s64_f64(add(mul(broadcast(ln2inv), x), broadcast(0.5)));

        auto g = sub(x, mul(vcvtq_f64_s64(n), broadcast(ln2C1)));
        g = sub(g, mul(vcvtq_f64_s64(n), broadcast(ln2C2)));

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg =
            add(broadcast(1), mul(broadcast(2), div(odd, sub(even, odd))));

        // Finally, compute product with 2^n.
        // Note: can only achieve full range using the ldexp implementation,
        // rather than multiplying by 2^n directly.

        auto result = ldexp_positive(expg, vmovn_s64(n));

        return ifelse(is_large, broadcast(HUGE_VAL),
                      ifelse(is_small, broadcast(0),
                             ifelse(is_not_nan, result, broadcast(NAN))));
    }*/

    // Use same rational polynomial expansion as for exp(x), without
    // the unit term.
    //
    // For |x|<=0.5, take n to be zero. Otherwise, set n as above,
    // and scale the answer by:
    //     expm1(x) = 2^n * expm1(g) + (2^n - 1).

    /*static float64x2_t expm1(const float64x2_t& x) {
        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(expm1_minarg));
        auto is_not_nan = cmp_eq(x, x);

        auto half = broadcast(0.5);
        auto one = broadcast(1.);
        auto two = add(one, one);

        auto nzero = cmp_leq(abs(x), half);
        auto n = vcvtmq_s64_f64(add(mul(broadcast(ln2inv), x), half));

        auto p = ifelse(nzero, zero(), vcvtq_f64_s64(n));

        auto g = sub(x, mul(p, broadcast(ln2C1)));
        g = sub(g, mul(p, broadcast(ln2C2)));

        auto gg = mul(g, g);

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Note: multiply by two, *then* divide: avoids a subnormal
        // intermediate that will get truncated to zero with default
        // icpc options.
        auto expgm1 = div(mul(two, odd), sub(even, odd));

        // For small x (n zero), bypass scaling step to avoid underflow.
        // Otherwise, compute result 2^n * expgm1 + (2^n-1) by:
        //     result = 2 * ( 2^(n-1)*expgm1 + (2^(n-1)+0.5) )
        // to avoid overflow when n=1024.

        auto nm1 = vmovn_s64(vcvtmq_s64_f64(sub(vcvtq_f64_s64(n), one)));

        auto scaled =
            mul(add(sub(exp2int(nm1), half), ldexp_normal(expgm1, nm1)), two);

        return ifelse(is_large, broadcast(HUGE_VAL),
                      ifelse(is_small, broadcast(-1),
                             ifelse(is_not_nan, ifelse(nzero, expgm1, scaled), broadcast(NAN))));
    }*/

    // Natural logarithm:
    //
    // Decompose x = 2^g * u such that g is an integer and
    // u is in the interval [sqrt(2)/2, sqrt(2)].
    //
    // Then ln(x) is computed as R(u-1) + g*ln(2) where
    // R(z) is a rational polynomial approximating ln(z+1)
    // for small z:
    //
    //     R(z) = z - z^2/2 + z^3 * P(z)/Q(z)
    //
    // where P and Q are degree 5 polynomials, Q monic.
    //
    // In order to avoid cancellation error, ln(2) is represented
    // as C3 + C4, with the C4 correction term approx. -2.1e-4.
    // The summation order for R(z)+2^g is:
    //
    //     z^3*P(z)/Q(z) + g*C4 - z^2/2 + z + g*C3

    /*static float64x2_t log(const float64x2_t& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        auto is_domainerr = cmp_lt(x, broadcast(0));

        auto is_nan = logical_not(cmp_eq(x, x));
        is_domainerr = logical_or(is_nan, is_domainerr);

        float64x2_t g = vcvt_f64_f32(vcvt_f32_s32(logb_normal(x)));
        float64x2_t u = fraction_normal(x);

        float64x2_t one = broadcast(1.);
        float64x2_t half = broadcast(0.5);
        auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
        g = ifelse(gtsqrt2, add(g, one), g);
        u = ifelse(gtsqrt2, mul(u, half), u);

        auto z = sub(u, one);
        auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
        auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

        auto z2 = mul(z, z);
        auto z3 = mul(z2, z);

        auto r = div(mul(z3, pz), qz);
        r = add(r, mul(g, broadcast(ln2C4)));
        r = sub(r, mul(z2, half));
        r = add(r, z);
        r = add(r, mul(g, broadcast(ln2C3)));

        // Return NaN if x is NaN or negarive, +inf if x is +inf,
        // or -inf if zero or (positive) denormal.

        return ifelse(is_domainerr, broadcast(NAN),
                      ifelse(is_large, broadcast(HUGE_VAL),
                             ifelse(is_small, broadcast(-HUGE_VAL), r)));
    }*/

    /*protected:
    static float64x2_t zero() { return vdupq_n_f64(0); }

    static svuint32_t hi_32b(float64x2_t x) {
        svuint32_t xwh = vget_high_u32(vreinterpretq_u32_f64(x));
        svuint32_t xwl = vget_low_u32(vreinterpretq_u32_f64(x));

        uint64x1_t xh = vand_u64(vreinterpret_u64_u32(xwh),
                                 vcreate_u64(0xffffffff00000000));
        uint64x1_t xl = vshr_n_u64(vreinterpret_u64_u32(xwl), 32);
        return vreinterpret_u32_u64(vorr_u64(xh, xl));
    }

    // horner(x, a0, ..., an) computes the degree n polynomial A(x) with
    // coefficients
    // a0, ..., an by a0+x·(a1+x·(a2+...+x·an)...).

    static inline float64x2_t horner(float64x2_t x, double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static float64x2_t horner(float64x2_t x, double a0, T... tail) {
        return add(mul(x, horner(x, tail...)), broadcast(a0));
    }

    // horner1(x, a0, ..., an) computes the degree n+1 monic polynomial A(x)
    // with coefficients
    // a0, ..., an, 1 by by a0+x·(a1+x·(a2+...+x·(an+x)...).

    static inline float64x2_t horner1(float64x2_t x, double a0) {
        return add(x, broadcast(a0));
    }

    template <typename... T>
    static float64x2_t horner1(float64x2_t x, double a0, T... tail) {
        return add(mul(x, horner1(x, tail...)), broadcast(a0));
    }

    // Compute 2.0^n.
    static float64x2_t exp2int(svint32_t n) {
        int64x2_t nlong = vshlq_s64(vmovl_s32(n), vdupq_n_s64(52));
        nlong = vaddq_s64(nlong, vshlq_s64(vdupq_n_s64(1023), vdupq_n_s64(52)));
        return vreinterpretq_f64_s64(nlong);
    }

    // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite
    // and normal.
    static svint32_t logb_normal(const float64x2_t& x) {
        svuint32_t xw = hi_32b(x);
        svuint32_t emask = vdup_n_u32(0x7ff00000);
        svuint32_t ebiased = vshr_n_u32(vand_u32(xw, emask), 20);

        return vsub_s32(vreinterpret_s32_u32(ebiased), vdup_n_s32(1023));
    }

    static float64x2_t fraction_normal(const float64x2_t& x) {
        // 0x800fffffffffffff (intrinsic takes signed parameter)
        uint64x2_t emask = vdupq_n_u64(-0x7ff0000000000001);
        uint64x2_t bias = vdupq_n_u64(0x3ff0000000000000);
        return vreinterpretq_f64_u64(
            vorrq_u64(bias, vandq_u64(emask, vreinterpretq_u64_f64(x))));
    }

    // Compute 2^n·x when both x and 2^n·x are normal, finite and strictly
    // positive doubles.
    static float64x2_t ldexp_positive(float64x2_t x, svint32_t n) {
        int64x2_t nlong = vmovl_s32(n);
        nlong = vshlq_s64(nlong, vdupq_n_s64(52));
        int64x2_t r = vaddq_s64(nlong, vreinterpretq_s64_f64(x));

        return vreinterpretq_f64_s64(r);
    }

    // Compute 2^n·x when both x and 2^n·x are normal and finite.
    static float64x2_t ldexp_normal(float64x2_t x, svint32_t n) {
        int64x2_t smask = vdupq_n_s64(0x7fffffffffffffffll);
        int64x2_t not_smask =
            vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(smask)));
        int64x2_t sbits = vandq_s64(not_smask, vreinterpretq_s64_f64(x));

        int64x2_t nlong = vmovl_s32(n);
        nlong = vshlq_s64(nlong, vdupq_n_s64(52));
        int64x2_t sum = vaddq_s64(nlong, vreinterpretq_s64_f64(x));

        auto nzans =
            vreinterpretq_f64_s64(vorrq_s64(vandq_s64(sum, smask), sbits));
        return ifelse(cmp_eq(x, zero()), zero(), nzans);
    }*/
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
