/* dgesvj_mex.c — MATLAB MEX wrapper for LAPACK DGESVJ (one-sided Jacobi SVD)
 *
 * Call from MATLAB:
 *   [Aout, S, V, sva, work, info] = dgesvj_mex(A, joba, jobu, jobv, mv, V0, lwork, work0)
 *
 * Inputs  (positional, MATLAB-friendly but mapped 1:1 to LAPACK)
 *   A     : double MxN, real, with M >= N. (Overwritten in-place by DGESVJ.)
 *   joba  : char in {'G','L','U'}         (matrix structure)             [in]
 *   jobu  : char in {'U','C','N'}         (left vectors / orthogonality) [in]
 *   jobv  : char in {'V','A','N'}         (right vectors / apply V0)     [in]
 *   mv    : integer >=0; used only if jobv=='A' (rows of V0 to update)   [in]
 *   V0    : double LDVxN; initial matrix if jobv=='A' (else []/ignored)  [in]
 *   lwork : integer;  -1 → workspace query; else LWORK >= max(6, M+N)    [in]
 *   work0 : double vec; optional seed of WORK. If jobu=='C' and ~empty,
 *           work0(1)=CTOL will be passed (must be >=1).                  [in]
 *
 * Outputs (raw LAPACK results; nothing is post-sorted or massaged)
 *   Aout  : the overwritten A (this is where DGESVJ returns U or U*Σ etc.)
 *           See LAPACK docs: for JOBU='U'/'C': leading RANKA columns are U;
 *           for JOBU='N': A holds columns of U scaled by σ (early stop).  [out]
 *   S     : NxN diagonal matrix with singular values as per doc:
 *           if WORK(1)==1 → diag(SVA); else diag( WORK(1) * SVA ).        [out]
 *   V     : if jobv=='V' → N×N right singular vectors;
 *           if jobv=='A' → product (V_right * V0(1:mv,:));
 *           if jobv=='N' → 1×1 dummy (not referenced by DGESVJ).          [out]
 *   sva   : length-N SVA vector returned by DGESVJ (unscaled).            [out]
 *   work  : WORK vector on exit.  Notable entries per LAPACK:
 *             WORK(1)=SCALE; WORK(2)=#nonzero σ; WORK(3)=#σ > underflow;
 *             WORK(4)=#Jacobi sweeps; WORK(5)=max|cos| in last sweep;     [out]
 *             WORK(6)=max|sin(theta)| in last sweep. (See docs.)
 *   info  : INTEGER INFO exactly as LAPACK returns it.                     [out]
 *
 * Key references:
 *   - Full argument list & semantics; JOBA/JOBU/JOBV, MV/LDV rules, and
 *     WORK(1:6) meanings (incl. WORK(4)=#sweeps).                         [1][2]
 */

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <ctype.h>

#if !defined(_WIN32)
#define dgesvj dgesvj_
#endif

/* Fortran prototype */
extern void dgesvj(char* JOBA, char* JOBU, char* JOBV,
                   int* M, int* N,
                   double* A, int* LDA,
                   double* SVA,
                   int* MV, double* V, int* LDV,
                   double* WORK, int* LWORK,
                   int* INFO);

/* Read a single char option and validate against allowed set */
static char read_opt_char(const mxArray* arg, const char* name, const char* allowed) {
    if (!mxIsChar(arg)) mexErrMsgIdAndTxt("dgesvj_mex:argtype", "%s must be a char.", name);
    char buf[8]; buf[0]='\0';
    if (mxGetString(arg, buf, sizeof(buf)) != 0 || buf[0]=='\0')
        mexErrMsgIdAndTxt("dgesvj_mex:argval", "Failed to read %s.", name);
    char c = (char)toupper((unsigned char)buf[0]);
    if (strchr(allowed, c) == NULL)
        mexErrMsgIdAndTxt("dgesvj_mex:argval", "%s = '%c' not in {%s}.", name, c, allowed);
    return c;
}

/* Make a 1x1 dummy double (for V when JOBV='N') */
static mxArray* make_dummy_double(void){
    mxArray* arr = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(arr) = 0.0;
    return arr;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /* ---- Parse inputs ---- */
    if (nrhs < 7 || nrhs > 8)
        mexErrMsgIdAndTxt("dgesvj_mex:arity",
            " LAPACK one-sided Jacobi. \n Usage: [Aout,S,V,sva,work,info] = dgesvj_mex(A,joba,jobu,jobv,mv,V0,lwork[,work0])");

    const mxArray* Ain = prhs[0];
    if (!mxIsDouble(Ain) || mxIsComplex(Ain))
        mexErrMsgIdAndTxt("dgesvj_mex:type","A must be real double.");
    mwSize Mmw = mxGetM(Ain), Nmw = mxGetN(Ain);
    if (Mmw < Nmw)
        mexErrMsgIdAndTxt("dgesvj_mex:shape","DGESVJ requires M >= N.");

    char JOBA = read_opt_char(prhs[1], "joba", "GLU");   /* structure */
    char JOBU = read_opt_char(prhs[2], "jobu", "UCN");   /* left vecs / orthogonality */
    char JOBV = read_opt_char(prhs[3], "jobv", "VAN");   /* right vecs / apply */

    /* MV and V0 (only meaningful if JOBV='A') */
    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetNumberOfElements(prhs[4])!=1)
        mexErrMsgIdAndTxt("dgesvj_mex:mv","mv must be a real scalar.");
    int MV = (int) mxGetScalar(prhs[4]);
    const mxArray* V0in = prhs[5];

    /* LWORK */
    if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetNumberOfElements(prhs[6])!=1)
        mexErrMsgIdAndTxt("dgesvj_mex:lwork","lwork must be a real scalar.");
    int LWORK = (int) mxGetScalar(prhs[6]);  /* -1 → workspace query */

    /* Optional WORK seed (for JOBU='C': CTOL in WORK(1)) */
    const mxArray* WORKseed = (nrhs==8 ? prhs[7] : NULL);
    double ctol_from_user = 0.0;
    int have_ctol = 0;
    if (WORKseed && mxGetNumberOfElements(WORKseed) >= 1) {
        if (!mxIsDouble(WORKseed) || mxIsComplex(WORKseed))
            mexErrMsgIdAndTxt("dgesvj_mex:work0","work0 must be a real vector.");
        ctol_from_user = mxGetPr(WORKseed)[0];
        have_ctol = 1;
    }

    /* ---- Dimensions and leading dims ---- */
    int M = (int) Mmw;
    int N = (int) Nmw;
    int LDA = (int) Mmw;

    /* ---- Prepare output A (copy, because DGESVJ overwrites) ---- */
    plhs[0] = mxDuplicateArray(Ain);
    double* A = mxGetPr(plhs[0]);

    /* ---- Prepare V (size depends on JOBV) ---- */
    mxArray* Varr = NULL;
    double* V = NULL;
    int LDV = 1;

    if (JOBV=='V') {
        LDV = N;
        Varr = mxCreateDoubleMatrix(LDV, N, mxREAL);
        V = mxGetPr(Varr);
        memset(V, 0, (size_t)LDV*(size_t)N*sizeof(double));
    } else if (JOBV=='A') {
        /* Apply rotations to first MV rows of V. Require V0 with at least MV rows and N columns. */
        if (mxIsEmpty(V0in))
            mexErrMsgIdAndTxt("dgesvj_mex:V0","V0 must be provided when jobv='A'.");
        if (!mxIsDouble(V0in) || mxIsComplex(V0in))
            mexErrMsgIdAndTxt("dgesvj_mex:V0type","V0 must be real double.");
        mwSize V0m = mxGetM(V0in), V0n = mxGetN(V0in);
        if (V0n != Nmw || (int)V0m < MV)
            mexErrMsgIdAndTxt("dgesvj_mex:V0size","V0 must be at least MV-by-N.");
        LDV = (int) V0m;
        Varr = mxDuplicateArray(V0in);   /* DGESVJ will overwrite */
        V = mxGetPr(Varr);
    } else { /* JOBV=='N' */
        Varr = make_dummy_double();
        V = mxGetPr(Varr);  /* not referenced */
        LDV = 1;
        MV  = 0;
    }

    /* ---- SVA, WORK ---- */
    mxArray* SVAarr  = mxCreateDoubleMatrix(Nmw, 1, mxREAL);
    double* SVA      = mxGetPr(SVAarr);

    /* If LWORK = -1 we do a workspace query; else require LWORK >= MAX(6, M+N) per docs. */
    if (LWORK != -1 && LWORK < ((M+N) > 6 ? (M+N) : 6))
        mexErrMsgIdAndTxt("dgesvj_mex:lworkmin","lwork must be >= max(6, M+N) or -1 for query.");

    mxArray* WORKarr = mxCreateDoubleMatrix( (mwSize)((LWORK==-1)?1:LWORK), 1, mxREAL );
    double* WORK     = mxGetPr(WORKarr);

    /* JOBU='C' allows user to set CTOL = WORK(1) before the call; must be >= 1 per docs. */
    if (JOBU=='C') {
        double ctol = have_ctol ? ctol_from_user : (double)M;  /* common choice: CTOL = M */
        if (ctol < 1.0)
            mexErrMsgIdAndTxt("dgesvj_mex:ctol","For JOBU='C', work0(1)=CTOL must be >= 1.");
        WORK[0] = ctol;
    }

    /* ---- Call DGESVJ ---- */
    int INFO = 0;
    dgesvj(&JOBA, &JOBU, &JOBV,
           &M, &N,
           A, &LDA,
           SVA,
           &MV, V, &LDV,
           WORK, &LWORK,
           &INFO);

    /* ---- Build S (scaled diag per docs: if SCALE=WORK(1)≠1 then σ = SCALE*SVA) ---- */
    mxArray* Sarr = mxCreateDoubleMatrix(Nmw, Nmw, mxREAL);
    double* S = mxGetPr(Sarr);
    memset(S, 0, (size_t)Nmw*(size_t)Nmw*sizeof(double));
    double SCALE = (LWORK==-1 ? 1.0 : WORK[0]);  /* in query, WORK(1) holds optimal LWORK, so just leave σ=SVA */
    for (int i=0; i<N; ++i) {
        double si = (LWORK==-1 ? 0.0 : SCALE * SVA[i]);  /* if query, no SVD computed → 0 */
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

    /* (No extra convenience outputs; you can read nsweeps as round(work(4)).) */
}
