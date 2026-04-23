/* sgesvj_mex.c — MATLAB MEX wrapper for LAPACK SGESVJ (single-precision one-sided Jacobi SVD)
 *
 * Call from MATLAB:
 *   [Aout, S, V, sva, work, info] = sgesvj_mex(A, joba, jobu, jobv, mv, V0, lwork, work0)
 *
 * Inputs  (positional, MATLAB-friendly but mapped 1:1 to LAPACK)
 *   A     : single MxN, real, with M >= N. (Overwritten in-place by SGESVJ.)
 *   joba  : char in {'G','L','U'}         (matrix structure)             [in]
 *   jobu  : char in {'U','C','N'}         (left vectors / orthogonality) [in]
 *   jobv  : char in {'V','A','N'}         (right vectors / apply V0)     [in]
 *   mv    : integer >=0; used only if jobv=='A' (rows of V0 to update)   [in]
 *   V0    : single LDVxN; initial matrix if jobv=='A' (else []/ignored)  [in]
 *   lwork : integer;  -1 → workspace query; else LWORK >= max(6, M+N)    [in]
 *   work0 : single vec; optional seed of WORK. If jobu=='C' and ~empty,
 *           work0(1)=CTOL will be passed (must be >=1).                  [in]
 *
 * Outputs (raw LAPACK results; nothing is post-sorted or massaged)
 *   Aout  : the overwritten A (this is where SGESVJ returns U or U*Σ etc.)
 *   S     : NxN diagonal matrix (single) with singular values as per doc:
 *           if WORK(1)==1 → diag(SVA); else diag( WORK(1) * SVA ).        [out]
 *   V     : if jobv=='V' → N×N right singular vectors;
 *           if jobv=='A' → product (V_right * V0(1:mv,:));
 *           if jobv=='N' → 1×1 dummy (not referenced by SGESVJ).          [out]
 *   sva   : length-N SVA vector returned by SGESVJ (unscaled).            [out]
 *   work  : WORK vector on exit.  Notable entries per LAPACK:
 *             WORK(1)=SCALE; WORK(2)=#nonzero σ; WORK(3)=#σ > underflow;
 *             WORK(4)=#Jacobi sweeps; WORK(5)=max|cos| in last sweep;     [out]
 *             WORK(6)=max|sin(theta)| in last sweep.
 *   info  : INTEGER INFO exactly as LAPACK returns it.                    [out]
 */

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <ctype.h>

#if !defined(_WIN32)
#define sgesvj sgesvj_
#endif

/* Fortran prototype */
extern void sgesvj(char* JOBA, char* JOBU, char* JOBV,
                   int* M, int* N,
                   float* A, int* LDA,
                   float* SVA,
                   int* MV, float* V, int* LDV,
                   float* WORK, int* LWORK,
                   int* INFO);

/* Read a single char option and validate against allowed set */
static char read_opt_char(const mxArray* arg, const char* name, const char* allowed) {
    if (!mxIsChar(arg)) mexErrMsgIdAndTxt("sgesvj_mex:argtype", "%s must be a char.", name);
    char buf[8]; buf[0]='\0';
    if (mxGetString(arg, buf, sizeof(buf)) != 0 || buf[0]=='\0')
        mexErrMsgIdAndTxt("sgesvj_mex:argval", "Failed to read %s.", name);
    char c = (char)toupper((unsigned char)buf[0]);
    if (strchr(allowed, c) == NULL)
        mexErrMsgIdAndTxt("sgesvj_mex:argval", "%s = '%c' not in {%s}.", name, c, allowed);
    return c;
}

/* Make a 1x1 dummy single (for V when JOBV='N') */
static mxArray* make_dummy_single(void){
    mxArray* arr = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    float* p = (float*)mxGetData(arr);
    *p = 0.0f;
    return arr;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /* ---- Parse inputs ---- */
    if (nrhs < 7 || nrhs > 8)
        mexErrMsgIdAndTxt("sgesvj_mex:arity",
            " LAPACK one-sided Jacobi (single). \n Usage: [Aout,S,V,sva,work,info] = sgesvj_mex(A,joba,jobu,jobv,mv,V0,lwork[,work0])");

    const mxArray* Ain = prhs[0];
    if (!mxIsSingle(Ain) || mxIsComplex(Ain))
        mexErrMsgIdAndTxt("sgesvj_mex:type","A must be real single.");

    mwSize Mmw = mxGetM(Ain), Nmw = mxGetN(Ain);
    if (Mmw < Nmw)
        mexErrMsgIdAndTxt("sgesvj_mex:shape","SGESVJ requires M >= N.");

    char JOBA = read_opt_char(prhs[1], "joba", "GLU");   /* structure */
    char JOBU = read_opt_char(prhs[2], "jobu", "UCN");   /* left vecs / orthogonality */
    char JOBV = read_opt_char(prhs[3], "jobv", "VAN");   /* right vecs / apply */

    /* MV and V0 (only meaningful if JOBV='A') */
    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetNumberOfElements(prhs[4])!=1)
        mexErrMsgIdAndTxt("sgesvj_mex:mv","mv must be a real scalar (double).");
    int MV = (int) mxGetScalar(prhs[4]);
    const mxArray* V0in = prhs[5];

    /* LWORK */
    if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetNumberOfElements(prhs[6])!=1)
        mexErrMsgIdAndTxt("sgesvj_mex:lwork","lwork must be a real scalar (double).");
    int LWORK = (int) mxGetScalar(prhs[6]);  /* -1 → workspace query */

    /* Optional WORK seed (for JOBU='C': CTOL in WORK(1)) */
    const mxArray* WORKseed = (nrhs==8 ? prhs[7] : NULL);
    float ctol_from_user = 0.0f;
    int have_ctol = 0;
    if (WORKseed && mxGetNumberOfElements(WORKseed) >= 1) {
        if (!mxIsSingle(WORKseed) || mxIsComplex(WORKseed))
            mexErrMsgIdAndTxt("sgesvj_mex:work0","work0 must be a real single vector.");
        float* w0 = (float*)mxGetData(WORKseed);
        ctol_from_user = w0[0];
        have_ctol = 1;
    }

    /* ---- Dimensions and leading dims ---- */
    int M = (int) Mmw;
    int N = (int) Nmw;
    int LDA = (int) Mmw;

    /* ---- Prepare output A (copy, because SGESVJ overwrites) ---- */
    plhs[0] = mxDuplicateArray(Ain);
    float* A = (float*) mxGetData(plhs[0]);

    /* ---- Prepare V (size depends on JOBV) ---- */
    mxArray* Varr = NULL;
    float* V = NULL;
    int LDV = 1;

    if (JOBV=='V') {
        LDV = N;
        Varr = mxCreateNumericMatrix((mwSize)LDV, (mwSize)N, mxSINGLE_CLASS, mxREAL);
        V = (float*) mxGetData(Varr);
        memset(V, 0, (size_t)LDV*(size_t)N*sizeof(float));
    } else if (JOBV=='A') {
        /* Apply rotations to first MV rows of V. Require V0 with at least MV rows and N columns. */
        if (mxIsEmpty(V0in))
            mexErrMsgIdAndTxt("sgesvj_mex:V0","V0 must be provided when jobv='A'.");
        if (!mxIsSingle(V0in) || mxIsComplex(V0in))
            mexErrMsgIdAndTxt("sgesvj_mex:V0type","V0 must be real single.");
        mwSize V0m = mxGetM(V0in), V0n = mxGetN(V0in);
        if (V0n != Nmw || (int)V0m < MV)
            mexErrMsgIdAndTxt("sgesvj_mex:V0size","V0 must be at least MV-by-N.");
        LDV = (int) V0m;
        Varr = mxDuplicateArray(V0in);   /* SGESVJ will overwrite */
        V = (float*) mxGetData(Varr);
    } else { /* JOBV=='N' */
        Varr = make_dummy_single();
        V = (float*) mxGetData(Varr);  /* not referenced */
        LDV = 1;
        MV  = 0;
    }

    /* ---- SVA, WORK ---- */
    mxArray* SVAarr  = mxCreateNumericMatrix(Nmw, 1, mxSINGLE_CLASS, mxREAL);
    float* SVA       = (float*) mxGetData(SVAarr);

    /* If LWORK = -1 we do a workspace query; else require LWORK >= MAX(6, M+N) per docs. */
    if (LWORK != -1 && LWORK < ((M+N) > 6 ? (M+N) : 6))
        mexErrMsgIdAndTxt("sgesvj_mex:lworkmin","lwork must be >= max(6, M+N) or -1 for query.");

    mwSize work_len = (mwSize)((LWORK==-1) ? 1 : LWORK);
    mxArray* WORKarr = mxCreateNumericMatrix(work_len, 1, mxSINGLE_CLASS, mxREAL);
    float* WORK      = (float*) mxGetData(WORKarr);

    /* JOBU='C' allows user to set CTOL = WORK(1) before the call; must be >= 1 per docs. */
    if (JOBU=='C') {
        float ctol = have_ctol ? ctol_from_user : (float)M;  /* common choice: CTOL = M */
        if (ctol < 1.0f)
            mexErrMsgIdAndTxt("sgesvj_mex:ctol","For JOBU='C', work0(1)=CTOL must be >= 1.");
        WORK[0] = ctol;
    }

    /* ---- Call SGESVJ ---- */
    int INFO = 0;
    sgesvj(&JOBA, &JOBU, &JOBV,
           &M, &N,
           A, &LDA,
           SVA,
           &MV, V, &LDV,
           WORK, &LWORK,
           &INFO);

    /* ---- Build S (scaled diag per docs: if SCALE=WORK(1)≠1 then σ = SCALE*SVA) ---- */
    mxArray* Sarr = mxCreateNumericMatrix(Nmw, Nmw, mxSINGLE_CLASS, mxREAL);
    float* S = (float*) mxGetData(Sarr);
    memset(S, 0, (size_t)Nmw*(size_t)Nmw*sizeof(float));

    float SCALE = (LWORK==-1 ? 1.0f : WORK[0]);  /* in query, WORK(1) holds optimal LWORK, so just leave σ=SVA */
    for (int i=0; i<N; ++i) {
        float si = (LWORK==-1 ? 0.0f : SCALE * SVA[i]);  /* if query, no SVD computed → 0 */
        S[i + (size_t)i*(size_t)Nmw] = si;
    }

    /* ---- Set outputs: Aout, S, V, sva, work, info ---- */
    plhs[1] = Sarr;
    plhs[2] = Varr;
    plhs[3] = SVAarr;
    plhs[4] = WORKarr;

    mxArray* INFOarr = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    *(int*)mxGetData(INFOarr) = INFO;
    plhs[5] = INFOarr;
}
