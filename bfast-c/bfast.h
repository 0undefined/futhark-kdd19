#ifndef BFAST_H
#define BFAST_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

#define R_TILE   32
#define T_TILE   16

typedef float real;

typedef struct HNsSigmaStruct {
    int32_t h;
    int32_t ns;
    real    sigma;
} HNnsSigma;

typedef struct DatasetStruct {
    int32_t  M;
    int32_t  N;
    int32_t  trend;
    int32_t  k;
    int32_t  n;
    real     freq;
    real     hfrac;
    real     lam;
    int32_t* mappingindices;
    real*    image;
} Dataset;

real logplus (const real x);    
void mkBound(const int32_t N, const int32_t n, const real lam, int32_t* mappingindices, real* BOUND);
real* mkX(const uint32_t N, const uint32_t k2p2, const real f, int32_t* mappingindices, real* X, real* Xt);
void batchMMM(const int32_t M, const int32_t N, const int32_t n, 
              const int32_t K, const real* Y, real* X, real* Xsqr);

void batchMinv(const int32_t K, real* Xsqr, real* Xinv1, real* Xinv2);
void mm1(const int32_t N, int32_t n, const int32_t K, real* Xh, real* Yh, real* beta0);
void mm2(const int32_t K, real* Xinv, real* beta0, real* beta);
void mm3(const int32_t N, const int32_t K, real* Xt, real* beta, real* y_preds);
int32_t filterKer(const int32_t N, real* Y, real* y_preds, int32_t* val_ind);
HNnsSigma sgmRedomap2Ker(const int32_t n, const int32_t K, const real hfrac, real* Yh, real* y_error); 

void runBfastMulticore(Dataset data, real* means, int32_t* fst_breaks);

void freeResources(Dataset input) {
    free(input.mappingindices);
    free(input.image);
}

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

int32_t min(int32_t a, int32_t b) {
    return (a < b) ? a : b;
}

Dataset buildDataset(const int32_t M, const int32_t N, const int32_t n, const real f_nan) {
    Dataset res;
    const int32_t Npad = ((N + T_TILE - 1) / T_TILE) * T_TILE;
    res.M = M; res.N = N; res.n = n; res.trend = 1;
    res.k = 3; res.freq = 12.0; //for peru 365.0f32 for sahara
    res.hfrac = 0.25; res.lam = 1.736126;
    
    res.mappingindices = (int32_t*)malloc(N*sizeof(int32_t));
    res.image = (real*)malloc(M*Npad*sizeof(real));

    // filling in mapping indices with [1..N]
    for(int32_t k=0; k<N; k++) {
        res.mappingindices[k] = k+1;
    }

    // filling in the image with contrived data
    srand(246);
    for(int32_t i=0; i<M; i++) {
        real r01 = rand() / (real)RAND_MAX;
        int32_t b0 = r01 * (N-n-2) + 1;
        int32_t brk= b0 + n; 
        for(int32_t j = 0; j < brk; j++) {
            real r01 = rand() / (real)RAND_MAX;
            if(r01 < f_nan) {
                res.image[i*N + j] = NAN;
            } else {
                real r01 = rand() / (real)RAND_MAX;
                real x   = r01 * 4000 + 4000;
                res.image[i*N + j] = x;
            }
        }
        for(int32_t j = brk; j < N; j++) {
            real r01 = rand() / (real)RAND_MAX;
            if(r01 < f_nan) {
                res.image[i*N + j] = NAN;
            } else {
                real r01 = rand() / (real)RAND_MAX;
                real x   = r01 * 5000;
                res.image[i*N + j] = x;
            }
        }
    }
    return res;
}

#endif // BFAST_H
