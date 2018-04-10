#pragma once

// NEON SIMD intrinsics implementation.

#if defined(__ARM_NEON__) || defined(__aarch64__)
#include <cmath>
#include <cstdint>
#include <arm_neon.h>

#include <simd/approx.hpp>
#include <simd/implbase.hpp>

namespace arb {
namespace simd {
namespace simd_detail {

struct neon_double2;

template <>
struct simd_traits<neon_double2> {
    static constexpr unsigned width = 2;
    using scalar_type = double;
    using vector_type = float64x2_t;
    using mask_impl = neon_double2; //int64x2_t?
};

struct neon_double2: implbase<neon_double2> {
    // Use default implementations for:
    //     element, set_element, fma.

    using int64 = std::int64_t;

    static float64x2_t broadcast(double v) {
        return vdupq_n_f64(v);
    }

    static void copy_to(const float64x2_t& v, double* p) {
		vst1q_f64(p, v);
    }/**essental**/

    static void copy_to_masked(const float64x2_t& v, double* p, const float64x2_t& mask) {
		
        float64x2_t r = vld1q_f64(p);
		r = vbslq_f64(vreinterpretq_u64_f64(mask), v, r);
		vst1q_f64(p, r);
	}

    static float64x2_t copy_from(const double* p) {
        return vld1q_f64(p);
    }/**essential**/

    static float64x2_t copy_from_masked(const double* p, const float64x2_t& mask) {
		float64x2_t a;
        float64x2_t r = vld1q_f64(p);
		a = vbslq_f64(vreinterpretq_u64_f64(mask), r, a);
		return a;
	}

    static float64x2_t copy_from_masked(const float64x2_t& v, const double* p, const float64x2_t& mask) {
		float64x2_t a = (*v);
        float64x2_t r = vld1q_f64(p);
		a = vbslq_f64(vreinterpretq_u64_f64(mask), r, a);
		return a;
	}

    static float64x2_t negate(const float64x2_t& a) {
		return vnegq_f64(a);
    }

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
		return vreinterpretq_f64_u64(vmvnq_u64(vreinterpretq_u64_f64(a)));
	}

    static float64x2_t logical_and(const float64x2_t& a, const float64x2_t& b) {
		return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a)));
	}

    static float64x2_t logical_or(const float64x2_t& a, const float64x2_t& b) {
		return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(a)));
	}

    static float64x2_t cmp_eq(const float64x2_t& a, const float64x2_t& b) {
		return vceqq_f64 (a, b);
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

    static float64x2_t ifelse(const float64x2_t& m, const float64x2_t& u, const float64x2_t& v) {
		a = vbslq_f64(vreinterpretq_u64_f64(mask), u, v);
		return a;
	}

    static float64x2_t mask_broadcast(bool b) {
		return vreinterpretq_f64_u64(vdupq_n_u64(-(int64)b));	
	}

    static bool mask_element(const float64x2_t& u, int i) {
        return static_cast<bool>(element(u, i));
    }/**essential**/

    static float64x2_t mask_unpack(unsigned long long k) {
        // Only care about bottom two bits of k.
		uint8x8_t b = vdup_n_u8((char)k); 
		unit8x8_t bl = vorr_u8(b, vdup_n_u8(0xfe));  
		unit8x8_t bu = vorr_u8(b, vdup_n_u8(0xfd));  
		uint8x16_t blu = vcombine_s8(bl, bu);

		uint8x16_t ones = vdupq_n_u8(0xff); 
		uint64x2_t r = vceqq_u64(vreinterpretq_u64_u8(ones), vreinterpretq_u64_u8(blu)); 
		
        return vreinterpretq_f64_u64(r);
    }

    static void mask_set_element(float64x2_t& u, int i, bool b) {
        char data[256];
		vst1q_f64((double*)data, u);
        ((int64*)data)[i] = -(int64)b;
		u = vld1q_f64((double*)data);
    }/**essential**/

    static void mask_copy_to(const float64x2_t& m, bool* y) {
        // Negate (convert 0xffffffff to 0x00000001) and move low bytes to
        // bottom 2 bytes.

        int8x16_t mc = vnegq_s8(vreinterpretq_s8_f64(m)); 
		int8x8_t mh = vget_high_s8(mc);
		mh = vand_s8(mh, vreinterpret_s8_s64(vdup_n_s64(0x000000000000ff00)));
		int8x8_t ml = vget_low_s8(mc);
		ml = vand_s8(mh, vreinterpret_s8_s64(vdup_n_s64(0x00000000000000ff)));
		mh = vadd_s8(mh, ml); 
        std::memcpy(y, &mh, 2);
    }/**essential**/

    static float64x2_t mask_copy_from(const bool* w) {
        // Move bytes:
        //   rl: byte 0 to byte 0, byte 1 to byte 8, zero elsewhere.
        //
        // Subtract from zero to translate
        // 0x0000000000000001 to 0xffffffffffffffff.

		int8x8x2_t t = vld2_s8(w); //intervleaved load
		int64x2_t r = vreinterpretq_s64_s8(vcombine_s8 ((t.val[0]), (t.val[1])));
		return vreinterpretq_f64_s64(vnegq_s64(r));
		
    }/**essential**/

    static float64x2_t max(const float64x2_t& a, const float64x2_t& b) {
        return _mm256_max_pd(a, b);
    }

    static float64x2_t min(const float64x2_t& a, const float64x2_t& b) {
        return _mm256_min_pd(a, b);
    }

    static float64x2_t abs(const float64x2_t& x) {
        __m256i m = _mm256_set1_epi64x(0x7fffffffffffffffll);
        return _mm256_and_pd(x, _mm256_castsi256_pd(m));
    }

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
    // ln(2) = C1 + C2, in order to compensate for the possible loss of precision
    // attributable to catastrophic rounding. C1 comprises the first
    // 32-bits of mantissa, C2 the remainder.

    static  float64x2_t exp(const float64x2_t& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));
        auto is_nan = _mm256_cmp_pd(x, x, cmp_unord_q);

        // Compute n and g.

        auto n = _mm256_floor_pd(add(mul(broadcast(ln2inv), x), broadcast(0.5)));

        auto g = sub(x, mul(n, broadcast(ln2C1)));
        g = sub(g, mul(n, broadcast(ln2C2)));

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = add(broadcast(1), mul(broadcast(2),
            div(odd, sub(even, odd))));

        // Finally, compute product with 2^n.
        // Note: can only achieve full range using the ldexp implementation,
        // rather than multiplying by 2^n directly.

        auto result = ldexp_positive(expg, _mm256_cvtpd_epi32(n));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(0),
            ifelse(is_nan, broadcast(NAN),
                   result)));
    }

    // Use same rational polynomial expansion as for exp(x), without
    // the unit term.
    //
    // For |x|<=0.5, take n to be zero. Otherwise, set n as above,
    // and scale the answer by:
    //     expm1(x) = 2^n * expm1(g) + (2^n - 1).

    static  float64x2_t expm1(const float64x2_t& x) {
        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(expm1_minarg));
        auto is_nan = _mm256_cmp_pd(x, x, cmp_unord_q);

        auto half = broadcast(0.5);
        auto one = broadcast(1.);
        auto two = add(one, one);

        auto nzero = cmp_leq(abs(x), half);
        auto n = _mm256_floor_pd(add(mul(broadcast(ln2inv), x), half));
        n = ifelse(nzero, zero(), n);

        auto g = sub(x, mul(n, broadcast(ln2C1)));
        g = sub(g, mul(n, broadcast(ln2C2)));

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

        auto nm1 = _mm256_cvtpd_epi32(sub(n, one));
        auto scaled = mul(add(sub(exp2int(nm1), half), ldexp_normal(expgm1, nm1)), two);

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-1),
            ifelse(is_nan, broadcast(NAN),
            ifelse(nzero, expgm1,
                   scaled))));
    }

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

    static float64x2_t log(const float64x2_t& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        auto is_domainerr = _mm256_cmp_pd(x, broadcast(0), cmp_nge_uq);

        float64x2_t g = _mm256_cvtepi32_pd(logb_normal(x));
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
        r = add(r, mul(g,  broadcast(ln2C4)));
        r = sub(r, mul(z2, half));
        r = add(r, z);
        r = add(r, mul(g,  broadcast(ln2C3)));

        // Return NaN if x is NaN or negarive, +inf if x is +inf,
        // or -inf if zero or (positive) denormal.

        return
            ifelse(is_domainerr, broadcast(NAN),
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-HUGE_VAL),
                r)));
    }

protected:
    static float64x2_t zero() {
        return vdupq_n_f64(0);
    }

    static __m128i hi_epi32(__m256i x) {
        __m128i xl = _mm256_castsi256_si128(x);
        __m128i xh = _mm256_extractf128_si256(x, 1);
        return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xl), _mm_castsi128_ps(xh), 0xddu));
    }

    static __m128i lo_epi32(__m256i x) {
        __m128i xl = _mm256_castsi256_si128(x);
        __m128i xh = _mm256_extractf128_si256(x, 1);
        return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xl), _mm_castsi128_ps(xh), 0x88u));
    }

    static __m256i combine_m128i(__m128i hi, __m128i lo) {
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
    }

    // horner(x, a0, ..., an) computes the degree n polynomial A(x) with coefficients
    // a0, ..., an by a0+x·(a1+x·(a2+...+x·an)...).

    static inline float64x2_t horner(float64x2_t x, double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static float64x2_t horner(float64x2_t x, double a0, T... tail) {
        return add(mul(x, horner(x, tail...)), broadcast(a0));
    }

    // horner1(x, a0, ..., an) computes the degree n+1 monic polynomial A(x) with coefficients
    // a0, ..., an, 1 by by a0+x·(a1+x·(a2+...+x·(an+x)...).

    static inline float64x2_t horner1(float64x2_t x, double a0) {
        return add(x, broadcast(a0));
    }

    template <typename... T>
    static float64x2_t horner1(float64x2_t x, double a0, T... tail) {
        return add(mul(x, horner1(x, tail...)), broadcast(a0));
    }

    // Compute 2.0^n.
    static float64x2_t exp2int(__m128i n) {
        n = _mm_slli_epi32(n, 20);
        n = _mm_add_epi32(n, _mm_set1_epi32(1023<<20));

        auto nl = _mm_shuffle_epi32(n, 0x50);
        auto nh = _mm_shuffle_epi32(n, 0xfa);
        __m256i nhnl = combine_m128i(nh, nl);

        return _mm256_castps_pd(
            _mm256_blend_ps(_mm256_set1_ps(0),
            _mm256_castsi256_ps(nhnl),0xaa));
    }

    // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite and normal.
    static __m128i logb_normal(const float64x2_t& x) {
        __m128i xw = hi_epi32(_mm256_castpd_si256(x));
        __m128i emask = _mm_set1_epi32(0x7ff00000);
        __m128i ebiased = _mm_srli_epi32(_mm_and_si128(xw, emask), 20);

        return _mm_sub_epi32(ebiased, _mm_set1_epi32(1023));
    }

    static float64x2_t fraction_normal(const float64x2_t& x) {
        // 0x800fffffffffffff (intrinsic takes signed parameter)
        float64x2_t emask = _mm256_castsi256_pd(_mm256_set1_epi64x(-0x7ff0000000000001));
        float64x2_t bias = _mm256_castsi256_pd(_mm256_set1_epi64x(0x3ff0000000000000));
        return _mm256_or_pd(bias, _mm256_and_pd(emask, x));
    }

    // Compute 2^n·x when both x and 2^n·x are normal, finite and strictly positive doubles.
    static float64x2_t ldexp_positive(float64x2_t x, __m128i n) {
        n = _mm_slli_epi32(n, 20);
        auto zero = _mm_set1_epi32(0);
        auto nl = _mm_unpacklo_epi32(zero, n);
        auto nh = _mm_unpackhi_epi32(zero, n);

        __m128d xl = _mm256_castpd256_pd128(x);
        __m128d xh = _mm256_extractf128_pd(x, 1);

        __m128i suml = _mm_add_epi64(nl, _mm_castpd_si128(xl));
        __m128i sumh = _mm_add_epi64(nh, _mm_castpd_si128(xh));
        __m256i sumhl = combine_m128i(sumh, suml);

        return _mm256_castsi256_pd(sumhl);
    }

    // Compute 2^n·x when both x and 2^n·x are normal and finite.
    static float64x2_t ldexp_normal(float64x2_t x, __m128i n) {
        float64x2_t smask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffll));
        float64x2_t sbits = _mm256_andnot_pd(smask, x);

        n = _mm_slli_epi32(n, 20);
        auto zi = _mm_set1_epi32(0);
        auto nl = _mm_unpacklo_epi32(zi, n);
        auto nh = _mm_unpackhi_epi32(zi, n);

        __m128d xl = _mm256_castpd256_pd128(x);
        __m128d xh = _mm256_extractf128_pd(x, 1);

        __m128i suml = _mm_add_epi64(nl, _mm_castpd_si128(xl));
        __m128i sumh = _mm_add_epi64(nh, _mm_castpd_si128(xh));
        __m256i sumhl = combine_m128i(sumh, suml);

        auto nzans = _mm256_or_pd(_mm256_and_pd(_mm256_castsi256_pd(sumhl), smask), sbits);
        return ifelse(cmp_eq(x, zero()), zero(), nzans);
    }
	
};

} // namespace simd_detail

namespace simd_abi {
    template <typename T, unsigned N> struct neon;

    template <> struct neon<double, 2> { using type = simd_detail::neon_double2; };

#endif
} // namespace simd_abi

} // namespace simd
} // namespace arb

#endif // def __ARM_NEON__
