#define RUNS_CPU 10

#include "bfast.h"

inline real logplus (const real x) {
    if ( x > exp(1) ) return log(x); else return 1;
}

inline
void mkBound(const int32_t N, const int32_t n, const real lam, int32_t* mappingindices, real* BOUND) {
    for(int32_t q=0; q<N-n; q++) {
        int32_t t = n+1+q;
        int32_t time = mappingindices[t-1];
        real tmp = logplus ( ((real)time) / mappingindices[N-1] );
        BOUND[q] = lam * sqrt(tmp);
    }
}

/**
 *  mappingindices : [N]int32_t
 *  X  : [k2p2][N]real
 *  Xt : [N][k2p2]real
 *  The results is (X, Xt)
 */
inline
real* mkX(const uint32_t N, const uint32_t k2p2, const real f, int32_t* mappingindices, real* X, real* Xt) {
    // compute X and transpose of X
    // does not make sense to parallelize or to optimize
    // because this takes negligible runtime (N < 1000, k2p2 = 8)
    for(uint32_t j = 0, offset=0; j < N; j++, offset+=k2p2) {
        real ind = (real)mappingindices[j];
        Xt[j*k2p2 + 0] = 1.0; X[0*N + j] = 1.0;
        Xt[j*k2p2 + 1] = ind; X[1*N + j] = ind;
        for(uint32_t i=2; i<k2p2; i++) {
            real angle = 2.0 * PI * (i/2) * ind / f;
            real v = (i & 1) ? cos(angle) : sin(angle);
            Xt[j*k2p2 + i] = v;  X[i*N + j] = v;
        }
    }
}

/**
 * batched matrix matrix multiplication
 * let Xsqr = map (matmul_filt Xh Xth) Yh
 */
inline void batchMMM(const int32_t M, const int32_t N, const int32_t n, 
                     const int32_t K, const real* Y, real* X, real* Xsqr) {
    for(int32_t jj=0; jj < n; jj+=T_TILE) {
        for(int32_t k1=0; k1 < K; k1++) {
            for(int32_t k2=0; k2 < K; k2++) {
                real vs[T_TILE];
                #pragma unroll
                for(int32_t j=0; j < T_TILE; j++) {
                    real x1 = X[k1*N + jj+j];
                    real x2 = X[k2*N + jj+j];
                    real v  = Y[jj+j];
                    vs[j]   = 0.0;
                    if( (jj+j < N) && (!isnan(v)) ) {
                        vs[j] = x1 * x2;
                    }
                }
                real acc = 0.0;
                #pragma unroll
                for(int32_t j=0; j < T_TILE; j++) {
                    acc += vs[j];
                }
                Xsqr[k1*K + k2] = acc;
            }
        }
    }
}

/**
 * batched matrix inversion
 * let Xinv = map mat_inv Xsqr
 * Result is in Xsqr! 
 */
inline void batchMinv(const int32_t K, real* Xsqr, real* Xinv11, real* Xinv22) {
    real *Xinv1 = Xinv11, *Xinv2 = Xinv22;
    const int32_t K2 = 2*K;
    // init step
    for(int32_t i=0; i<K; i++) {
        for(int32_t j=0; j<K2; j++) {
            Xinv1[i*K2 + j] = (j < K) ? Xsqr[i*K + j] : (j==K+i) ? 1.0 : 0.0;
        }
    }
    // Gauss-Jordan
    for(int32_t i=0; i<K; i++) {
        real v1 = Xinv1[i];
        if (v1 != 0) {
            for(int32_t k=0; k<K-1; k++) {
                for(int32_t j=0; j<K2; j++) {
                    real x = Xinv1[j] / v1;
                    Xinv2[k*K2+j] = (k < K-1) ? (Xinv1[(k+1)*K2+j] - x*Xinv1[(k+1)*K2+i]) : x;
                }
            }
            int32_t k = K-1;
            for(int32_t j=0; j<K2; j++) {
                Xinv2[k*K2+j] = Xinv1[j] / v1;
            }
            // switch pointers
            {
                real* tmp = Xinv1;
                Xinv1 = Xinv2;
                Xinv2 = tmp;
            }
        }
    }
    // finally, write back only the second half to Xsqr;
    // init step
    for(int32_t i=0; i<K; i++) {
        for(int32_t j=0; j<K; j++) {
            Xsqr[i*K + j] = Xinv1[i*K2 + j + K];
        }
    }
}

/**
 * Xh : logically [K,n]real, but the layout is [K,N]real
 * Yh : [n]real
 * let beta0  = matvecmul_row_filt Xh Yh   -- [2k+2]
 */ 
inline 
void mm1(const int32_t N, int32_t n, const int32_t K, real* Xh, real* Yh, real* beta0) {
    for(int32_t i=0; i<K; i++) {
        real* Xh_row = Xh + i*N;
        real acc = 0.0;
        for(int32_t j=0; j<n; j++) {
            real y = Yh[j];
            if (!isnan(y)) {
                acc += Xh_row[j] * y;
            }
        }
        beta0[i] = acc;
    }
}

/**
 * Xinv  : [K,K]real
 * beta0 : [K]real
 * let beta = matvecmul_row Xinv beta0 --[2k+2]
 */ 
inline 
void mm2(const int32_t K, real* Xinv, real* beta0, real* beta) {
    for(int32_t i=0; i<K; i++) {
        real* Xinv_row = Xinv + i*K;
        real acc = 0.0;
        for(int32_t j=0; j<K; j++) {
            acc += Xinv_row[j] * beta0[j];
        }
        beta[i] = acc;
    }
}

/**
 * Xt   : [N][K]real
 * beta : [K]real
 * let y_preds= matvecmul_row Xt beta -- : [N]real
 */ 
inline
void mm3(const int32_t N, const int32_t K, real* Xt, real* beta, real* y_preds) {
    for(int32_t i=0; i<N; i++) {
        real* Xt_row = Xt + i*K;
        real acc = 0.0;
        for(int32_t j=0; j<K; j++) {
            acc += Xt_row[j] * beta[j];
        }
        y_preds[i] = acc;
    }
}

/**
 * Y       : [N]real
 * y_preds : [N]real
 * val_inds: [N]int32_t
 * let (Nss, y_errors, val_indss) = map2 (...filterPadWithKeys...)
 */
inline 
int32_t filterKer(const int32_t N, real* Y, real* y_preds, int32_t* val_ind) {
    int32_t count = 0;
    for(int32_t i=0; i<N; i++) {
        real y  = Y[i];
        if(!isnan(y)) {
            real yp = y_preds[i];
            real d  = y - yp;
            y_preds[count] = d;
            val_ind[count] = i;
            count++;
        }
    }
    return count;
}

/**
 * Y       : [n]real
 * y_error : [n]real
 * let (hs, nss, sigmas) = map2 (... redomap o redomap ...)
 */
inline
HNnsSigma sgmRedomap2Ker(const int32_t n, const int32_t K, const real hfrac, real* Yh, real* y_error) {
    int32_t ns = 0, h = 0;
    real sigma = 0.0;
    for(int32_t i=0; i<n; i++) {
        if (!isnan(Yh[i])) ns++;
    }
    for(int32_t i=0; i<ns; i++) {
        real v = y_error[i];
        sigma += v*v;
    }
    sigma = sqrt( sigma / (ns-K));
    h = (int) hfrac * ns;
    HNnsSigma res;
    res.h = h; res.ns = ns; res.sigma = sigma;
    return res;
}

#define CHUNK 64
void runBfastMulticore(Dataset data, real* means, int32_t* fst_breaks) {
    const int32_t M = data.M;
    const int32_t N = data.N;
    const int32_t n = data.n;
    const int32_t K = 2*data.k + 2;
    const int32_t Ksq = K * K;
    const int32_t Npad = ((N + T_TILE - 1) / T_TILE) * T_TILE;

    real* mem1 = (real*)malloc( (2*K*Npad + (N-n) )*sizeof(real) );
    real* X  = mem1;           // [K,N]
    real* Xt = X + K*Npad;     // [N,K]
    real* BOUND = Xt + K*Npad; // [N-n]

    // 1. compute BOUND
    mkBound(N, n, data.lam, data.mappingindices, BOUND);

    // 2. compute X and its transpose Xt
    mkX(N, K, data.freq, data.mappingindices, X, Xt);

    #pragma omp parallel for
    for(int32_t ii=0; ii<M; ii+=CHUNK) {
        real* mem2  = (real*)malloc( (Ksq + 4*Ksq + 2*K + N + N-n)*sizeof(real) + N*sizeof(int32_t) );

        real* Xsqr    = mem2;           // [Ksq]
        real* Xinv1   = Xsqr  + Ksq;    // [2*Ksq]
        real* Xinv2   = Xinv1 + 2*Ksq;  // [2*Ksq]
        real* beta0   = Xinv2 + 2*Ksq;  // [K]
        real* beta    = beta0 + K;      // [k]
        real* y_error = beta  + K;      // [N]
        real* MO      = y_error + N;    // [N-n]
        int32_t* val_ind = (int32_t*)(MO + N - n); // [N]

        for(int32_t i=ii; i<min(ii+CHUNK, M); i++) {
            real* Y = data.image + i*N;

            batchMMM (M, N, n, K, Y, X, Xsqr);
            batchMinv(K, Xsqr, Xinv1, Xinv2); // the result is in Xsqr
            mm1(N, n, K, X, Y, beta0);        // the result is in beta0
            mm2(K, Xsqr, beta0, beta);        // the result is in beta
            mm3(N, K, Xt, beta, y_error);     // the result is in y_error

            int32_t Ns = filterKer(N, Y, y_error, val_ind); // result in y_error (y_error) and val_ind
            HNnsSigma hnssig = sgmRedomap2Ker(n, K, data.hfrac, Y, y_error);
            int32_t ns = hnssig.ns;
            int32_t h  = hnssig.h;
            real sigma = hnssig.sigma;

            real MO_fst = 0.0;
            for(int32_t j=0; j<h; j++) {
                MO_fst += y_error[j + ns - h + 1];
            }
            
            real MO_acc = MO_fst;
            for(int32_t j=0; j < Ns-ns; j++) {
                MO_acc += (-y_error[ns-h+j] + y_error[ns+j]);
                MO[j] = MO_acc;
            }

            real mean = 0.0;
            int32_t is_break = 0;
            int32_t fst_break = -1;
            for(int32_t j=0; j < Ns-ns; j++) {
                real b  = BOUND[j];
                real mo = MO[j];
                mo = mo / (sigma * sqrt((real)ns));
                
                if(is_break == 0) {
                    if( (!isnan(mo)) && (fabs(mo) > b) ) {
                        is_break = 1;
                        fst_break = j;
                    }
                }
                mean += mo;
            }

            if(is_break) {
                fst_break = (fst_break < Ns - ns) ? (val_ind[fst_break+ns] - n) : -1;
            }
            if ( (ns <= 5) || (Ns-ns <= 5) ) {
                fst_break = -2;
            }

            fst_breaks[i] = fst_break;
            means[i] = mean;
        }
      
        free(mem2);
    }

    free(mem1);
}

int main(int argc, char** argv) {
    Dataset input;

    if(argc != 5) {
        printf("Bfast expects 4 arguments:\n");
        printf("(1) the number of pixels in the image (M : int)\n");
        printf("(2) the length of the time series for each pixel (N : int)\n");
        printf("(3) the length of the time series used for training (n : int, n < N)\n");
        printf("(4) the fraction of NaN values (nanfreq : real, nanfreq < 1.0)\n");
        printf("This program generates a contrived dataset from this values");
        printf(" and runs Bfast in parallel using OpenMP; only the outer loop is parallelized.\n");
        exit(1);
    }

    { // 1. build the dataset
        int32_t M = atoi(argv[1]);
        int32_t N = atoi(argv[2]);
        int32_t n = atoi(argv[3]);
        real f_nan = atof(argv[4]);

        fprintf(stderr, "Input parameters are: M=%d, N=%d, n=%d, f_nan=%.2f\n", M, N, n, f_nan);

        input = buildDataset(M, N, n, f_nan);
    }
    
    { // 2. run CPU-parallel and report average runtime across RUNS_CPU runs
        int64_t elapsed, aft, bef = get_wall_time();

        real* means = (real*)malloc(input.M*sizeof(real));
        int32_t* fst_breaks = (int32_t*)malloc(input.M*sizeof(int32_t));

        for (int32_t i = 0; i < RUNS_CPU; i++) {
            runBfastMulticore(input, means, fst_breaks);
        }

        aft = get_wall_time();
        elapsed = aft - bef;
        printf("%ldμs\n", elapsed/RUNS_CPU);
    }

    freeResources(input);
} 
