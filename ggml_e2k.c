#include <stdint.h>
#include <immintrin.h>
#define inline inline __attribute__((__always_inline__))
#define E2K_ALIGNED 1

#ifdef __e2k__
#define _mm_shuffle2_epi8(a, b, c) \
	((__m128i)__builtin_e2k_qppermb((__v2di)(b), (__v2di)(a), (__v2di)(c)))
#else
#define _mm_shuffle2_epi8(a, b, c) \
	_mm_blendv_epi8(_mm_shuffle_epi8(a, c), _mm_shuffle_epi8(b, c), \
			_mm_slli_epi16(c, 3))
#endif

#define QK4_0 32
typedef struct {
	float d; // delta
	uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK8_0 32
typedef struct {
	float d; // delta
	int8_t qs[QK8_0]; // quants
} block_q8_0;

static inline
__m128i e2k_dot_q4_0_q8_0_core(__m128i bx, __m128i by0, __m128i by1) {
	__m128i c15 = _mm_set1_epi8(15), c8 = _mm_set1_epi8(8);
	__m128i bx0 = _mm_and_si128(bx, c15);
	__m128i bx1 = _mm_and_si128(_mm_srli_epi16(bx, 4), c15);
#ifdef __AVXVNNIINT8__
	bx0 = _mm_sub_epi8(bx0, bias);
	bx1 = _mm_sub_epi8(bx1, bias);
	__m128i dot = _mm_setzero_si128();
	dot = _mm_dpbssds_epi32(dot, bx0, by0);
	dot = _mm_dpbssds_epi32(dot, bx1, by1);
	return dot;
#else
	__m128i sy0 = _mm_maddubs_epi16(c8, by0);
	__m128i sy1 = _mm_maddubs_epi16(c8, by1);
	__m128i dot0 = _mm_maddubs_epi16(bx0, by0);
	__m128i dot1 = _mm_maddubs_epi16(bx1, by1);
	__m128i dot = _mm_add_epi16(dot0, dot1);
	return _mm_sub_epi16(dot, _mm_add_epi16(sy0, sy1));
#endif
}

#if E2K_ALIGNED && __LCC__ >= 126
__attribute__((optimize("-faligned")))
#endif
void ggml_vec_dot_q4_0_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
	const int nb = n / QK8_0;
	const block_q4_0 * restrict x = vx;
	const block_q8_0 * restrict y = vy;
	int i;
	__m128 acc = _mm_setzero_ps();

#pragma loop count(1000)
	for (i = 0; i < nb - 3; i += 4, x += 4, y += 4) {
#if !E2K_ALIGNED
		__m128i bx0 = _mm_loadu_si128((__m128i*)x[0].qs);
		__m128i bx1 = _mm_loadu_si128((__m128i*)x[1].qs);
		__m128i bx2 = _mm_loadu_si128((__m128i*)x[2].qs);
		__m128i bx3 = _mm_loadu_si128((__m128i*)x[3].qs);
		__m128i by0l = _mm_loadu_si128((__m128i*)y[0].qs);
		__m128i by0h = _mm_loadu_si128((__m128i*)y[0].qs + 1);
		__m128i by1l = _mm_loadu_si128((__m128i*)y[1].qs);
		__m128i by1h = _mm_loadu_si128((__m128i*)y[1].qs + 1);
		__m128i by2l = _mm_loadu_si128((__m128i*)y[2].qs);
		__m128i by2h = _mm_loadu_si128((__m128i*)y[2].qs + 1);
		__m128i by3l = _mm_loadu_si128((__m128i*)y[3].qs);
		__m128i by3h = _mm_loadu_si128((__m128i*)y[3].qs + 1);

		uint64_t xla = *(uint32_t*)&x[0].d | (uint64_t)*(uint32_t*)&x[1].d << 32;
		uint64_t xlb = *(uint32_t*)&x[2].d | (uint64_t)*(uint32_t*)&x[3].d << 32;
		uint64_t yla = *(uint32_t*)&y[0].d | (uint64_t)*(uint32_t*)&y[1].d << 32;
		uint64_t ylb = *(uint32_t*)&y[2].d | (uint64_t)*(uint32_t*)&y[3].d << 32;
#else
		__m128i xl0 = _mm_load_si128((__m128i*)x);
		__m128i xl1 = _mm_load_si128((__m128i*)x + 1);
		__m128i xl2 = _mm_load_si128((__m128i*)x + 2);
		__m128i xl3 = _mm_load_si128((__m128i*)x + 3);
		__m128i bx3 = _mm_load_si128((__m128i*)x + 4);

		__m128i yl0 = _mm_load_si128((__m128i*)y);
		__m128i yl1 = _mm_load_si128((__m128i*)y + 1);
		__m128i yl2 = _mm_load_si128((__m128i*)y + 2);
		__m128i yl3 = _mm_load_si128((__m128i*)y + 3);
		__m128i yl4 = _mm_load_si128((__m128i*)y + 4);
		__m128i yl5 = _mm_load_si128((__m128i*)y + 5);
		__m128i yl6 = _mm_load_si128((__m128i*)y + 6);
		__m128i by3l = _mm_load_si128((__m128i*)y + 7);
		__m128i by3h = _mm_load_si128((__m128i*)y + 8);

		__m128i bx0 = _mm_alignr_epi8(xl1, xl0, 4);
		__m128i bx1 = _mm_alignr_epi8(xl2, xl1, 8);
		__m128i bx2 = _mm_alignr_epi8(xl3, xl2, 12);
		__m128i by0l = _mm_alignr_epi8(yl1, yl0, 4);
		__m128i by0h = _mm_alignr_epi8(yl2, yl1, 4);
		__m128i by1l = _mm_alignr_epi8(yl3, yl2, 8);
		__m128i by1h = _mm_alignr_epi8(yl4, yl3, 8);
		__m128i by2l = _mm_alignr_epi8(yl5, yl4, 12);
		__m128i by2h = _mm_alignr_epi8(yl6, yl5, 12);

		uint64_t xla = _mm_cvtsi128_si64(_mm_shuffle2_epi8(xl0, xl1,
				_mm_set_epi64x(0x8080808080808080, 0x1716151403020100)));
		uint64_t xlb = _mm_cvtsi128_si64(_mm_shuffle2_epi8(xl2, xl3,
				_mm_set_epi64x(0x8080808080808080, 0x1f1e1d1c0b0a0908)));

		uint64_t yla = _mm_cvtsi128_si64(_mm_shuffle2_epi8(yl0, yl2,
				_mm_set_epi64x(0x8080808080808080, 0x1716151403020100)));
		uint64_t ylb = _mm_cvtsi128_si64(_mm_shuffle2_epi8(yl4, yl6,
				_mm_set_epi64x(0x8080808080808080, 0x1f1e1d1c0b0a0908)));
#endif

		__m128i xy0 = e2k_dot_q4_0_q8_0_core(bx0, by0l, by0h);
		__m128i xy1 = e2k_dot_q4_0_q8_0_core(bx1, by1l, by1h);
		__m128i xy2 = e2k_dot_q4_0_q8_0_core(bx2, by2l, by2h);
		__m128i xy3 = e2k_dot_q4_0_q8_0_core(bx3, by3l, by3h);
#ifdef __AVXVNNIINT8__
		xy0 = _mm_packs_epi32(xy0, xy1);
		xy2 = _mm_packs_epi32(xy2, xy3);
#else
		xy0 = _mm_hadd_epi16(xy0, xy1);
		xy2 = _mm_hadd_epi16(xy2, xy3);
#endif
		xy0 = _mm_hadd_epi16(xy0, xy2);
		xy0 = _mm_madd_epi16(xy0, _mm_set1_epi16(1));
		__m128 fxy = _mm_cvtepi32_ps(xy0);

		__m128 xl = _mm_castsi128_ps(_mm_set_epi64x(xlb, xla));
		__m128 yl = _mm_castsi128_ps(_mm_set_epi64x(ylb, yla));
		acc = _mm_fmadd_ps(_mm_mul_ps(xl, yl), fxy, acc);
	}

	if (i < nb) {
#if !E2K_ALIGNED
		__m128i bx0 = _mm_loadu_si128((__m128i*)x[0].qs);
		__m128i bx1 = _mm_loadu_si128((__m128i*)x[1].qs);

		__m128i by0l = _mm_loadu_si128((__m128i*)y[0].qs);
		__m128i by0h = _mm_loadu_si128((__m128i*)y[0].qs + 1);
		__m128i by1l = _mm_loadu_si128((__m128i*)y[1].qs);
		__m128i by1h = _mm_loadu_si128((__m128i*)y[1].qs + 1);

		uint64_t xla = *(uint32_t*)&x[0].d | (uint64_t)*(uint32_t*)&x[1].d << 32;
		uint64_t yla = *(uint32_t*)&y[0].d | (uint64_t)*(uint32_t*)&y[1].d << 32;
#else
		__m128i xl0 = _mm_load_si128((__m128i*)x);
		__m128i xl1 = _mm_load_si128((__m128i*)x + 1);
		__m128i xl2 = _mm_load_si128((__m128i*)x + 2);

		__m128i yl0 = _mm_load_si128((__m128i*)y);
		__m128i yl1 = _mm_load_si128((__m128i*)y + 1);
		__m128i yl2 = _mm_load_si128((__m128i*)y + 2);
		__m128i yl3 = _mm_load_si128((__m128i*)y + 3);
		__m128i yl4 = _mm_load_si128((__m128i*)y + 4);

		__m128i bx0 = _mm_alignr_epi8(xl1, xl0, 4);
		__m128i bx1 = _mm_alignr_epi8(xl2, xl1, 8);
		__m128i by0l = _mm_alignr_epi8(yl1, yl0, 4);
		__m128i by0h = _mm_alignr_epi8(yl2, yl1, 4);
		__m128i by1l = _mm_alignr_epi8(yl3, yl2, 8);
		__m128i by1h = _mm_alignr_epi8(yl4, yl3, 8);

		uint64_t xla = _mm_cvtsi128_si64(_mm_shuffle2_epi8(xl0, xl1,
				_mm_set_epi64x(0x8080808080808080, 0x1716151403020100)));
		uint64_t yla = _mm_cvtsi128_si64(_mm_shuffle2_epi8(yl0, yl2,
				_mm_set_epi64x(0x8080808080808080, 0x1716151403020100)));
#endif
		__m128i xy0 = e2k_dot_q4_0_q8_0_core(bx0, by0l, by0h);
		__m128i xy1 = e2k_dot_q4_0_q8_0_core(bx1, by1l, by1h);
#ifdef __AVXVNNIINT8__
		xy0 = _mm_packs_epi32(xy0, xy1);
#else
		xy0 = _mm_hadd_epi16(xy0, xy1);
#endif
		xy0 = _mm_hadd_epi16(xy0, _mm_setzero_si128());
		xy0 = _mm_madd_epi16(xy0, _mm_set1_epi16(1));
		__m128 fxy = _mm_cvtepi32_ps(xy0);
		__m128 xl = _mm_castsi128_ps(_mm_set_epi64x(0, xla));
		__m128 yl = _mm_castsi128_ps(_mm_set_epi64x(0, yla));
		acc = _mm_fmadd_ps(_mm_mul_ps(xl, yl), fxy, acc);
	}
	acc = _mm_hadd_ps(acc, acc);
	acc = _mm_hadd_ps(acc, acc);
	*s = _mm_cvtss_f32(acc);
}
