/* sgejsv_mex.c — MATLAB MEX wrapper for LAPACK SGEJSV (preconditioned Jacobi SVD)
 *
 * Call from MATLAB:
 *   [U, S, V, sva, work, iwork, info] = sgejsv_mex(A, joba, jobu, jobv, jobr, jobt, jobp, lwork)
 *
 * Inputs
 *   A     : (single or double, real) MxN with M >= N (required)
 *           - If double is provided, it is cast to single internally.
 *   joba  : char in {'C','E','F','G','A','R'}  (required)  [accuracy/conditioning option]
 *   jobu  : char in {'U','F','W','N'}          (required)  [left vectors: N cols, full M, workspace, none]
 *   jobv  : char in {'V','J','W','N'}          (required)  [right vectors: V/J (accum), workspace, none]
 *   jobr  : char in {'N','R'}                  (required)  [range restriction / allow killing tiny cols]
 *   jobt  : char in {'N','T'}                  (required)  [may transpose if square & entropy test says so]
 *   jobp  : char in {'N','P'}                  (required)  [structured perturbation of denormals]
 *   lwork : (optional) positive int. If omitted or <=0, a safe default is used
 *
 * Outputs (all SINGLE precision, matching SGEJSV)
 *   U     : as per JOBU (MxN if 'U'; MxM if 'F'; minimal dummy if 'N'; used as workspace if 'W')  [out]
 *   S     : NxN diagonal matrix with scaled singular values: (work(1)/work(2))*sva   [convenience]
 *   V     : as per JOBV (NxN if 'V' or 'J' or 'W'; minimal dummy if 'N')             [out]
 *   sva   : length-N vector SVA returned by SGEJSV (see scaling notes)               [out]
 *   work  : single workspace returned by SGEJSV (length max(7, LWORK) we used)      [out]
 *   iwork : int32 workspace returned by SGEJSV (length max(3, M+3*N))               [out]
 *   info  : int32 (Fortran INFO)                                                    [out]
 *
 * Key spec points (LAPACK 3.12.x):
 *   - Prototype: SGEJSV(JOBA,JOBU,JOBV,JOBR,JOBT,JOBP,M,N,A,LDA,SVA,U,LDU,V,LDV,WORK,LWORK,IWORK,INFO)
 *   - WORK on exit: WORK(1)=uscal2*scalem, WORK(2)=uscal1. Final sigmas are (WORK(1)/WORK(2))*SVA.
 *
 * Build (Linux/macOS, typical):
 *   mex -R2018a sgejsv_mex.c -lmwlapack -lmwblas
 */

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <ctype.h>
#include <math.h>

/* Fortran symbol name mangling (gfortran appends underscore on Unix-like systems) */
#if !defined(_WIN32)
#define sgejsv sgejsv_
#endif

/* Fortran prototype */
extern void sgejsv(char* JOBA, char* JOBU, char* JOBV, char* JOBR, char* JOBT, char* JOBP,
                   int* M, int* N, float* A, int* LDA,
                   float* SVA,
                   float* U, int* LDU,
                   float* V, int* LDV,
                   float* WORK, int* LWORK,
                   int* IWORK,
                   int* INFO);

/* Helper: read a single-character option */
static char read_opt_char(const mxArray* arg, const char* name, const char* allowed){
    if (!mxIsChar(arg)) mexErrMsgIdAndTxt("sgejsv_mex:argtype", "%s must be a char array.", name);
    char buf[8]; buf[0]='\0';
    if (mxGetString(arg, buf, sizeof(buf)) != 0 || buf[0]=='\0')
        mexErrMsgIdAndTxt("sgejsv_mex:argval", "Failed to read %s.", name);
    char c = (char)toupper((unsigned char)buf[0]);
    if (strchr(allowed, c) == NULL)
        mexErrMsgIdAndTxt("sgejsv_mex:argval", "%s = '%c' not in {%s}.", name, c, allowed);
    return c;
}

/* Make a 1x1 dummy single array for unused U/V if needed */
static mxArray* make_dummy_single(){
    mxArray* arr = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
    *(float*)mxGetData(arr) = 0.0f;
    return arr;
}

/* Convert/copy A to a new single matrix (column-major), as SGEJSV overwrites A */
static mxArray* to_single_copy(const mxArray* Ain){
    mwSize M = mxGetM(Ain);
    mwSize N = mxGetN(Ain);
    mxArray* Aout = mxCreateNumericMatrix(M, N, mxSINGLE_CLASS, mxREAL);
    float* dst = (float*)mxGetData(Aout);

    if (mxIsSingle(Ain)) {
        const float* src = (const float*)mxGetData(Ain);
        memcpy(dst, src, (size_t)M*(size_t)N*sizeof(float));
        return Aout;
    }

    /* must be double */
    const double* src = mxGetPr(Ain);
    size_t n = (size_t)M*(size_t)N;
    for (size_t i = 0; i < n; ++i) dst[i] = (float)src[i];
    return Aout;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    /* ---- Parse inputs ---- */
    if (nrhs < 7 || nrhs > 8)
        mexErrMsgIdAndTxt("sgejsv_mex:arity",
                          "Usage: [U,S,V,sva,work,iwork,info] = sgejsv_mex(A,joba,jobu,jobv,jobr,jobt,jobp[,lwork])");

    const mxArray* Ain = prhs[0];
    if (mxIsComplex(Ain) || !(mxIsSingle(Ain) || mxIsDouble(Ain)))
        mexErrMsgIdAndTxt("sgejsv_mex:type","A must be real single or real double.");

    mwSize Mmw = mxGetM(Ain), Nmw = mxGetN(Ain);
    if (Mmw < Nmw)
        mexErrMsgIdAndTxt("sgejsv_mex:shape","Require M >= N.");

    char JOBA = read_opt_char(prhs[1], "joba", "CEFGAR");
    char JOBU = read_opt_char(prhs[2], "jobu", "UFWN");
    char JOBV = read_opt_char(prhs[3], "jobv", "VJWN");
    char JOBR = read_opt_char(prhs[4], "jobr", "NR");
    char JOBT = read_opt_char(prhs[5], "jobt", "TN");
    char JOBP = read_opt_char(prhs[6], "jobp", "PN");

    long lwork_in = 0;
    if (nrhs == 8) {
        if (!mxIsNumeric(prhs[7]) || mxIsComplex(prhs[7]) || mxGetNumberOfElements(prhs[7]) != 1)
            mexErrMsgIdAndTxt("sgejsv_mex:lwork","lwork must be a real scalar.");
        lwork_in = (long) mxGetScalar(prhs[7]);
    }

    /* ---- Dimensions (Fortran int = 32-bit) ---- */
    int M = (int) Mmw;
    int N = (int) Nmw;
    int LDA = (int) Mmw;

    /* ---- Allocate outputs according to JOBU/JOBV ---- */
    mxArray *Uarr = NULL, *Varr = NULL;

    int UC = (JOBU=='F') ? M : N;
    int LDU = ( (JOBU=='U') || (JOBU=='F') || (JOBU=='W') ) ? M : 1;
    if (JOBU=='N' && JOBT=='T') {
        /* Provide workspace if transposition is chosen internally; allocate MxN to be safe */
        LDU = M; UC = N;
    }
    if (LDU>0 && UC>0 && JOBU!='N') Uarr = mxCreateNumericMatrix((mwSize)LDU, (mwSize)UC, mxSINGLE_CLASS, mxREAL);
    else if (JOBU=='N' && JOBT!='T') Uarr = make_dummy_single();
    else Uarr = mxCreateNumericMatrix((mwSize)LDU, (mwSize)UC, mxSINGLE_CLASS, mxREAL);

    int LDV = ( (JOBV=='V') || (JOBV=='J') || (JOBV=='W') ) ? N : 1;
    if (JOBV=='N' && JOBT=='T') {
        /* Provide workspace if needed per docs; allocate NxN to be safe */
        LDV = N;
    }
    if (LDV>0 && N>0 && JOBV!='N') Varr = mxCreateNumericMatrix((mwSize)LDV, (mwSize)N, mxSINGLE_CLASS, mxREAL);
    else if (JOBV=='N' && JOBT!='T') Varr = make_dummy_single();
    else Varr = mxCreateNumericMatrix((mwSize)LDV, (mwSize)N, mxSINGLE_CLASS, mxREAL);

    float* U = (float*)mxGetData(Uarr);
    float* V = (float*)mxGetData(Varr);

    /* ---- SVA (length N), WORK, IWORK ---- */
    mxArray* SVAarr = mxCreateNumericMatrix((mwSize)N, 1, mxSINGLE_CLASS, mxREAL);
    float* SVA = (float*)mxGetData(SVAarr);

    /* If user didn’t pass lwork (or <=0), pick a safe default (same logic as double wrapper). */
    long LWORK_long = lwork_in;
    long t1 = 6L*N + 2L*N*N;
    long t2 = 2L*M + N;
    long t3 = 4L*N + N*(long)N;
    long t4 = 2L*N + N*(long)N + 6L;
    long minSafe = 7L;
    long safeDefault = t1;
    if (t2 > safeDefault) safeDefault = t2;
    if (t3 > safeDefault) safeDefault = t3;
    if (t4 > safeDefault) safeDefault = t4;
    if (minSafe > safeDefault) safeDefault = minSafe;
    if (LWORK_long <= 0) LWORK_long = safeDefault;
    if (LWORK_long < 7) LWORK_long = 7;

    int LWORK = (int) LWORK_long;
    mxArray* WORKarr = mxCreateNumericMatrix((mwSize)LWORK, 1, mxSINGLE_CLASS, mxREAL);
    float* WORK = (float*)mxGetData(WORKarr);

    int iwork_len = (M + 3*N > 3 ? (M + 3*N) : 3);
    mxArray* IWORKarr = mxCreateNumericMatrix((mwSize)iwork_len, 1, mxINT32_CLASS, mxREAL);
    int* IWORK = (int*) mxGetData(IWORKarr);

    /* ---- Copy/cast A (SGEJSV overwrites) ---- */
    mxArray* Awork = to_single_copy(Ain);
    float* A = (float*)mxGetData(Awork);

    /* ---- Call SGEJSV ---- */
    int INFO = 0;
    sgejsv(&JOBA, &JOBU, &JOBV, &JOBR, &JOBT, &JOBP,
           &M, &N, A, &LDA,
           SVA,
           U, &LDU,
           V, &LDV,
           WORK, &LWORK,
           IWORK,
           &INFO);

    /* S = diag( (WORK(1)/WORK(2))*SVA ) */
    float scale_num = (LWORK >= 1 ? WORK[0] : 1.0f);
    float scale_den = (LWORK >= 2 ? WORK[1] : 1.0f);
    float scale = (scale_den != 0.0f) ? (scale_num / scale_den) : 1.0f;

    mxArray *Sarr = mxCreateNumericMatrix(Nmw, Nmw, mxSINGLE_CLASS, mxREAL);
    float *S = (float*)mxGetData(Sarr);
    memset(S, 0, (size_t)Nmw * (size_t)Nmw * sizeof(float));

    for (int i = 0; i < N; ++i) {
        S[i + (size_t)i * (size_t)Nmw] = scale * SVA[i];
    }

    /* ---- Set plhs ---- */
    plhs[0] = Uarr;
    plhs[1] = Sarr;
    plhs[2] = Varr;
    plhs[3] = SVAarr;
    plhs[4] = WORKarr;
    plhs[5] = IWORKarr;

    mxArray* INFOarr = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    *(int*)mxGetData(INFOarr) = INFO;
    plhs[6] = INFOarr;

    mxDestroyArray(Awork);
}
