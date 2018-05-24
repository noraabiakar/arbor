#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <benchmark/benchmark.h>

#define WIDTH 4

void vectorAcc_2g_1s_store_r(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m128i vec_ind = _mm_loadu_si128((const __m128i*)(ind + i));
            __m256d vec_b = _mm256_i32gather_pd(b, vec_ind, 8); 

            __m256d vec_a = _mm256_i32gather_pd(a, vec_ind, 8);

            __m256d vec_r = _mm256_add_pd(vec_a, vec_b);

            double r[WIDTH]; 
            _mm256_storeu_pd(r, vec_r); 

            for(int j = 0; j < WIDTH; j++) {
               a[ind[i + j]] = r[j];
            }
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(ind);
}

void vectorAcc_2g_1s_store_r_i(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m128i vec_ind = _mm_loadu_si128((const __m128i*)(ind + i));
            __m256d vec_b = _mm256_i32gather_pd(b, vec_ind, 8); 

            __m256d vec_a = _mm256_i32gather_pd(a, vec_ind, 8);

            __m256d vec_r = _mm256_add_pd(vec_a, vec_b);

            double r[WIDTH]; 
            _mm256_storeu_pd(r, vec_r); 

            int id[WIDTH]; 
            _mm_storeu_si128((__m128i*)id, vec_ind); 

            for(int j = 0; j < WIDTH; j++) {
               a[id[j]] = r[j];
            }
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(ind);
}

void vectorAcc_2g_1s_memcpy_r_i(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m128i vec_ind = _mm_loadu_si128((const __m128i*)(ind + i));
            __m256d vec_b = _mm256_i32gather_pd(b, vec_ind, 8); 

            __m256d vec_a = _mm256_i32gather_pd(a, vec_ind, 8);

            __m256d vec_r = _mm256_add_pd(vec_a, vec_b);

            //int* id = reinterpret_cast<int*>(&vec_ind);
	    //double* r = reinterpret_cast<double*>(&vec_r);
            int id[WIDTH];
            double r[WIDTH];

            memcpy(id, &vec_ind, sizeof(__m128i));
            memcpy(r, &vec_r, sizeof(__m256d));

            for(int j = 0; j < WIDTH; j++) {
               a[id[j]] = r[j];
            }
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(ind);
}

void autoAcc_2g_1s(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i++) {
            a[ind[i]] += b[ind[i]]; 
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(ind);
}

void vectorAcc_1g(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m128i vec_ind = _mm_loadu_si128((const __m128i*)(ind + i));
            __m256d vec_b = _mm256_i32gather_pd(b, vec_ind, 8); 

            __m256d vec_a = _mm256_loadu_pd(a + i);

            __m256d vec_r = _mm256_add_pd(vec_a, vec_b);

            _mm256_storeu_pd(a + i, vec_r);
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(ind);
}

void autoAcc_1g(benchmark::State& state) {

    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i++) {
            a[i] += b[ind[i]]; 
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(ind);
}

void vectorAcc_1s(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b, *__restrict__ c; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    c = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        c[i] = 2.7 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m256d vec_b = _mm256_loadu_pd(b + i);
            __m256d vec_c = _mm256_loadu_pd(c + i);
            __m256d vec_r = _mm256_add_pd(vec_b, vec_c);
            double* r= reinterpret_cast<double*>(&vec_r);
            for(int j = 0; j < WIDTH; j++) {
               a[ind[i + j]] = r[j];
            }
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(c);
    free(ind);
}

void autoAcc_1s(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b, *__restrict__ c; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    c = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        c[i] = 2.7 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i ++) {
            a[ind[i]] = b[i] + c[i]; 
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(c);
    free(ind);
}

void vectorAcc_1g_1s(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b, *__restrict__ c; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    c = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        c[i] = 2.7 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m128i vec_ind = _mm_loadu_si128((const __m128i*)(ind + i));
            __m256d vec_b = _mm256_i32gather_pd(b, vec_ind, 8); 

            __m256d vec_c = _mm256_loadu_pd(c + i);

            __m256d vec_r = _mm256_add_pd(vec_b, vec_c);

            double* r= reinterpret_cast<double*>(&vec_r);
            int* id= reinterpret_cast<int*>(&vec_ind);
            for(int j = 0; j < WIDTH; j++) {
               a[id[j]] = r[j];
            }
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(c);
    free(ind);
}

void vectorAcc_1g_1s_split(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b, *__restrict__ c, *__restrict__ d; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    c = (double *)malloc(sizeof(double) * size);
    d = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        c[i] = 2.7 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m128i vec_ind = _mm_loadu_si128((const __m128i*)(ind + i));
            __m256d vec_b = _mm256_i32gather_pd(b, vec_ind, 8); 

            __m256d vec_c = _mm256_loadu_pd(c + i);

            __m256d vec_r = _mm256_add_pd(vec_b, vec_c);

            _mm256_storeu_pd(d + i, vec_r);
        }

        for(unsigned i = 0; i < size; i ++) {
               a[ind[i]] = d[i];
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(c);
    free(d);
    free(ind);
}

void vectorAcc_1g_1s_unroll_g(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b, *__restrict__ c; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    c = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        c[i] = 2.7 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i += WIDTH) {
            __m128i vec_ind = _mm_loadu_si128((const __m128i*)(ind + i));
            int* id= reinterpret_cast<int*>(&vec_ind);

            double p[WIDTH]; 
            for(int j = 0; j < WIDTH; j++) {
               p[j] = b[id[j]];
            }

            __m256d vec_b = _mm256_loadu_pd(p); 

            __m256d vec_c = _mm256_loadu_pd(c + i);

            __m256d vec_r = _mm256_add_pd(vec_b, vec_c);

            double* r= reinterpret_cast<double*>(&vec_r);
            for(int j = 0; j < WIDTH; j++) {
               a[id[j]] = r[j];
            }
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(c);
    free(ind);
}

void autoAcc_1g_1s(benchmark::State& state) {
    const unsigned size = state.range(0);

    double *__restrict__ a, *__restrict__ b, *__restrict__ c; 
    int *__restrict__ ind; 

    a = (double *)malloc(sizeof(double) * size);
    b = (double *)malloc(sizeof(double) * size);
    c = (double *)malloc(sizeof(double) * size);
    ind = (int *)malloc(sizeof(int) * size);

    for(unsigned i = 0; i < size; i++) {
        a[i] = 1.5 * (i + 1); 
        b[i] = 1.4 * (i + 1); 
        c[i] = 2.7 * (i + 1); 
        ind[i] = i;
    }

    while (state.KeepRunning()) {
        for(unsigned i = 0; i < size; i++) {
            a[ind[i]] = b[ind[i]] + c[i]; 
        }
        //benchmark::ClobberMemory();
    }
    free(a);
    free(b);
    free(c);
    free(ind);
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto size: {100, 1000, 10000, 100000, 1000000, 10000000}) {
        b->Args({size});
    }
}

BENCHMARK(autoAcc_2g_1s)->Apply(run_custom_arguments);
BENCHMARK(vectorAcc_2g_1s_store_r)->Apply(run_custom_arguments);
//BENCHMARK(vectorAcc_2g_1s_store_r_i)->Apply(run_custom_arguments);
BENCHMARK(vectorAcc_2g_1s_memcpy_r_i)->Apply(run_custom_arguments);
BENCHMARK(autoAcc_1g)->Apply(run_custom_arguments);
BENCHMARK(vectorAcc_1g)->Apply(run_custom_arguments);
BENCHMARK(autoAcc_1s)->Apply(run_custom_arguments);
BENCHMARK(vectorAcc_1s)->Apply(run_custom_arguments);
BENCHMARK(autoAcc_1g_1s)->Apply(run_custom_arguments);
BENCHMARK(vectorAcc_1g_1s)->Apply(run_custom_arguments);
BENCHMARK(vectorAcc_1g_1s_split)->Apply(run_custom_arguments);
BENCHMARK(vectorAcc_1g_1s_unroll_g)->Apply(run_custom_arguments);
BENCHMARK_MAIN();
