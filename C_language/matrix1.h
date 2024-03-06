#ifndef MATRIX1
#define MATRIX1


/*lapack関連プロトタイプ宣言*/
//-------------------------------------------------------------------------------------------//

//行列乗算
void dgemm_(char *transA, char *transB, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldB, double *beta, double *C, int *ldc);
void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
//ユークリッドノルムを計算
double dnrm2_(int *n, double *x, int *incx);
//A*X = Bを計算
void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *B, int *ldb, int *info);
//倍
void dscal_(int *n, double *da, double *dx, int *incx);
//足し算など
void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
//シューア分解
void dgees_(char *jobvs, char *sort, bool (*SELECT)(double, double), int *n, double *A, int *lda, int *sdim, double *wr, double *wi, double *vs, int *ldvs, double *work, int *lwork, bool *bwork, int *info);
//qr分解用
void dgeqr2_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *info);
void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info); //こっちでいい

//行列normの計算 norm='f'ならdnrm2と(たぶん)同じ
double dlange_(char *norm, int *m, int *n, double *a, int *lda, double *work);

//行列コピー
void dlacpy_(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb);

void dorgqr_(int *m, int *n, int *k, double *A, int *lda, double *tau, double *work, int *lwork, int *info);

//実対称行列 固有値
void dsyev_(char *jobz, char *uplo, int *n, double *A, int *lda, double *w, double *work, int *lwork, int *info);

//AX+XB=C A,Bは(準)上三角
void dtrsyl_(char *trana, char *tranb, int *isgn, int *m, int *n, double *a, int *lda, double *b, int *ldb, double *c, int *ldc, double *scale, int *info);

//------------------------------------------------------------------------------------------//


void matrix_inv(double *a, int size);

void matrix_inv3(double *a, int n, double *ans);

void matrixA(double *a, double *a2, int size);

void mat_multi(double *a, double *b, double *a2, int size);

void mat_multi2(double *a, double *b, double *a2, int size, double alpha);

void mat_sum(double *a, double *b, int size, double da);

void mat_sum2(double *a, double *b, double *c, int row_size, double da);

void mat_sum_sym(double *a, double *b, int n);

void mat_sum_osoi(double *a, double *b, int n);

void print_mat(double *a, int n, int m);

void mat_diff(double *dif, double *A, double *sol, int size);

void mat_multidouble(double *a, double x, int size);

void setmatrix(double *A, double *I, int n, int m, double emax, double emin);

void setmatrix2(double *A, double *I, int n, int m, double emax, double emin, int seed);

void setmatrix3(double *A, int n, double emax, double emin, int seed);

void setmatrix_complex(double *A, int n, double rmin, double rmax, double imin, double imax, int seed, double *ans);

void setmatrix_ans(double *A, int n, double emax, double emin, int seed, double *ans, char ch);

void schur(double *a, double *vs, int size);

void mat_T(double *a, double *at, int size);

void mat_diff(double *dif, double *A, double *sol, int size);

void file_read(char *filename, double *a, int n);

void identity(double *I, int n);

double calc_rerr_id(double *A, int size);

double calc_err_id(double *A, int size);

void qr(double *q, double *r, int nsize);

double dif_norm(double *A, double *B, int n);

void matcpy(double *A, double *B, int n);

void setmatrix_X(double *A, int n, double emax, double emin, int seed, double cond, char qrs);

void setmatrix_sym(double A[], int n, double emax, double emin, int seed);

double normf(double *A, int n);

void sign_jordan(double *eig, double *evec, int size);

void setmatrix_sym2(double *A, int n, double emax, double emin, int seed, double *ans, char ch);

double mat_norm(double *A, int matsize, char norm);

#endif
