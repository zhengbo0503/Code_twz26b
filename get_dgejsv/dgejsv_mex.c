/* dgejsv_mex.c — MATLAB MEX wrapper for LAPACK DGEJSV (preconditioned Jacobi SVD)
 *
 * Call from MATLAB:
 *   [U, S, V, sva, work, iwork, info] = dgejsv_mex(A, joba, jobu, jobv, jobr, jobt, jobp, lwork)
 *
 * Inputs
 *   A     : (double, real) MxN with M >= N (required)
 *   joba  : char in {'C','E','F','G','A','R'}  (required)  [accuracy/conditioning option]
 *   jobu  : char in {'U','F','W','N'}          (required)  [left vectors: N cols, full M, workspace, none]
 *   jobv  : char in {'V','J','W','N'}          (required)  [right vectors: V/J (accum), workspace, none]
 *   jobr  : char in {'N','R'}                  (required)  [range restriction / allow killing tiny cols]
 *   jobt  : char in {'N','T'}                  (required)  [may transpose if square & entropy test says so]
 *   jobp  : char in {'N','P'}                  (required)  [structured perturbation of denormals]
 *   lwork : (optional) positive int. If omitted or <=0, a safe default is used
 *
 * Outputs
 *   U     : as per JOBU (MxN if 'U'; MxM if 'F'; minimal dummy if 'N'; used as workspace if 'W')  [out]
 *   S     : NxN diagonal matrix with scaled singular values: (work(1)/work(2))*sva   [convenience]
 *   V     : as per JOBV (NxN if 'V' or 'J' or 'W'; minimal dummy if 'N')             [out]
 *   sva   : length-N vector SVA returned by DGEJSV (see scaling notes)               [out]
 *   work  : double workspace returned by DGEJSV (length max(7, LWORK) we used)       [out]
 *   iwork : integer workspace returned by DGEJSV (length max(3, M+3*N))              [out]
 *   info  : int (Fortran INFO)                                                       [out]
 *
 * Key spec points (LAPACK 3.12.x):
 *   - Prototype: DGEJSV(JOBA,JOBU,JOBV,JOBR,JOBT,JOBP,M,N,A,LDA,SVA,U,LDU,V,LDV,WORK,LWORK,IWORK,INFO)
 *   - WORK on exit: WORK(1)=uscal2*scalem, WORK(2)=uscal1. Final sigmas are (WORK(1)/WORK(2))*SVA.
 *   - Minimal/optimal LWORK depends on job; we use a safe default if user doesn’t pass LWORK.
 *   See the official docs we cite in the instructions.
 */

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <ctype.h>
#include <math.h>

/* Fortran symbol name mangling (gfortran appends underscore on Unix-like systems) */
#if !defined(_WIN32)
#define dgejsv dgejsv_
#endif

/* Fortran prototype */
extern void dgejsv(char* JOBA, char* JOBU, char* JOBV, char* JOBR, char* JOBT, char* JOBP,
                   int* M, int* N, double* A, int* LDA,
                   double* SVA,
                   double* U, int* LDU,
                   double* V, int* LDV,
                   double* WORK, int* LWORK,
                   int* IWORK,
                   int* INFO);

/* Helper: read a single-character option */
static char read_opt_char(const mxArray* arg, const char* name, const char* allowed){
    if (!mxIsChar(arg)) mexErrMsgIdAndTxt("dgejsv_mex:argtype", "%s must be a char array.", name);
    char buf[8]; buf[0]='\0';
    if (mxGetString(arg, buf, sizeof(buf)) != 0 || buf[0]=='\0')
        mexErrMsgIdAndTxt("dgejsv_mex:argval", "Failed to read %s.", name);
    char c = (char)toupper((unsigned char)buf[0]);
    /* validate */
    if (strchr(allowed, c) == NULL)
        mexErrMsgIdAndTxt("dgejsv_mex:argval", "%s = '%c' not in {%s}.", name, c, allowed);
    return c;
}

/* Make a 1x1 dummy double array for unused U/V if needed */
static mxArray* make_dummy_double(){
    mxArray* arr = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(arr) = 0.0;
    return arr;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    /* ---- Parse inputs ---- */
    if (nrhs < 7 || nrhs > 8)
        mexErrMsgIdAndTxt("dgejsv_mex:arity",
                          "Usage: [U,S,V,sva,work,iwork,info] = dgejsv_mex(A,joba,jobu,jobv,jobr,jobt,jobp[,lwork])");

    const mxArray* Ain = prhs[0];
    if (!mxIsDouble(Ain) || mxIsComplex(Ain))
        mexErrMsgIdAndTxt("dgejsv_mex:type","A must be real double.");
    mwSize Mmw = mxGetM(Ain), Nmw = mxGetN(Ain);
    if (Mmw < Nmw)
        mexErrMsgIdAndTxt("dgejsv_mex:shape","Require M >= N.");

    char JOBA = read_opt_char(prhs[1], "joba", "CEFGAR");
    char JOBU = read_opt_char(prhs[2], "jobu", "UFWN");
    char JOBV = read_opt_char(prhs[3], "jobv", "VJWN");
    char JOBR = read_opt_char(prhs[4], "jobr", "NR");
    char JOBT = read_opt_char(prhs[5], "jobt", "TN");
    char JOBP = read_opt_char(prhs[6], "jobp", "PN");

    long lwork_in = 0;
    if (nrhs == 8) {
        if (!mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || mxGetNumberOfElements(prhs[7]) != 1)
            mexErrMsgIdAndTxt("dgejsv_mex:lwork","lwork must be a real scalar.");
        lwork_in = (long) mxGetScalar(prhs[7]);
    }

    /* ---- Dimensions (Fortran int = 32-bit) ---- */
    int M = (int) Mmw;
    int N = (int) Nmw;
    int LDA = (int) Mmw;

    /* ---- Allocate outputs according to JOBU/JOBV ----
       U dims: (LDU x UC), where UC = (JOBU=='F' ? M : N) if JOBU in {'U','F','W'}
               If JOBU=='N' and JOBT=='T' LAPACK may still reference U/V as workspace; we allocate minimally safe sizes below.
    */
    mxArray *Uarr = NULL, *Varr = NULL;

    int UC = (JOBU=='F') ? M : N;
    int LDU = ( (JOBU=='U') || (JOBU=='F') || (JOBU=='W') ) ? M : 1;
    if (JOBU=='N' && JOBT=='T') {
        /* Provide workspace if transposition is chosen internally; allocate MxN to be safe */
        LDU = M; UC = N;
    }
    if (LDU>0 && UC>0 && JOBU!='N') Uarr = mxCreateDoubleMatrix(LDU, UC, mxREAL);
    else if (JOBU=='N' && JOBT!='T') Uarr = make_dummy_double();
    else Uarr = mxCreateDoubleMatrix(LDU, UC, mxREAL); /* safe fallback */

    /* V dims: (LDV x N) when JOBV in {'V','J','W'}; else dummy (unless JOBT=='T', then may be used) */
    int LDV = ( (JOBV=='V') || (JOBV=='J') || (JOBV=='W') ) ? N : 1;
    if (JOBV=='N' && JOBT=='T') {
        /* Provide workspace if needed per docs; allocate NxN to be safe */
        LDV = N;
    }
    if (LDV>0 && N>0 && JOBV!='N') Varr = mxCreateDoubleMatrix(LDV, N, mxREAL);
    else if (JOBV=='N' && JOBT!='T') Varr = make_dummy_double();
    else Varr = mxCreateDoubleMatrix(LDV, N, mxREAL); /* safe fallback */

    double* U = mxGetPr(Uarr);
    double* V = mxGetPr(Varr);

    /* ---- SVA (length N), WORK, IWORK ---- */
    mxArray* SVAarr = mxCreateDoubleMatrix(N, 1, mxREAL);
    double* SVA = mxGetPr(SVAarr);

    /* If user didn’t pass lwork (or <=0), pick a safe default following SciPy/LAPACK guidance:
       LWORK >= max( 6*N + 2*N*N, 2*M + N, 4*N + N*N, 2*N + N*N + 6, 7 )
       (covers full-SVD worst cases with JOBU/JOBV per docs). */
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

    mxArray* WORKarr = mxCreateDoubleMatrix((mwSize)LWORK, 1, mxREAL);
    double* WORK = mxGetPr(WORKarr);

    int iwork_len = (M + 3*N > 3 ? (M + 3*N) : 3);
    mxArray* IWORKarr = mxCreateNumericMatrix(iwork_len, 1, mxINT32_CLASS, mxREAL);
    int* IWORK = (int*) mxGetData(IWORKarr);

    /* ---- Duplicate A (DGEJSV overwrites) ---- */
    mxArray* Awork = mxDuplicateArray(Ain);
    double* A = mxGetPr(Awork);

    /* ---- Call DGEJSV ---- */
    int INFO = 0;
    dgejsv(&JOBA, &JOBU, &JOBV, &JOBR, &JOBT, &JOBP,
           &M, &N, A, &LDA,
           SVA,
           U, &LDU,
           V, &LDV,
           WORK, &LWORK,
           IWORK,
           &INFO);

    /* Prepare outputs */
    /* S = diag( (WORK(1)/WORK(2))*SVA ) */
    double scale_num = (LWORK >= 1 ? WORK[0] : 1.0);
    double scale_den = (LWORK >= 2 ? WORK[1] : 1.0);
    double scale = (scale_den != 0.0) ? (scale_num / scale_den) : 1.0;

    /* Use the *existing* Nmw (mwSize) from above; do NOT redeclare it */
    mxArray *Sarr = mxCreateDoubleMatrix(Nmw, Nmw, mxREAL);
    double *S = mxGetPr(Sarr);
    memset(S, 0, (size_t)Nmw * (size_t)Nmw * sizeof(double));

    for (int i = 0; i < N; ++i) {
      /* column-major indexing: row i + col i * leading_dimension */
      S[i + (size_t)i * (size_t)Nmw] = scale * SVA[i];
    }

    /* ---- Set plhs ----
       Order: U, S, V, sva, work, iwork, info
    */
    plhs[0] = Uarr;
    plhs[1] = Sarr;
    plhs[2] = Varr;
    plhs[3] = SVAarr;
    plhs[4] = WORKarr;
    plhs[5] = IWORKarr;

    mxArray* INFOarr = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    *(int*)mxGetData(INFOarr) = INFO;
    plhs[6] = INFOarr;

    /* Clean temporary (A copy) */
    mxDestroyArray(Awork);
}
