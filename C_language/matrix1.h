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

//逆行列計算
//sizeは行か列のサイズ
void matrix_inv(double *a, int size);

/*
逆行列計算関数2
行列AをAの逆行列Ainvに変換する．
matrix_invとは異なり，dgesvを適用してみる
AX = BでXを計算するLAPACKサブルーチンであり，B=I(単位行列)とすれば，
純粋にAの逆行列が得られる．
メモリ領域を新たに確保する必要があるが，dgetrfとdgetriを用いて計算
するよりも速いかもしれない．
size: 行列の行か列のサイズ

たぶんメモリ使用料がこっちの方が多いかも
*/
void matrix_inv2(double *a, double *ainv, int size);

void matrix_inv3(double *a, int n, double *ans);

//Aの累乗(A^2), a2に結果が入る
void matrixA(double *a, double *a2, int size);

//行列の積を計算 a*b = a2
void mat_multi(double *a, double *b, double *a2, int size);

//行列の積を計算 計算結果は3引数目の変数に入る
//sizeには行列の行数
//C = alpah*AB とdgemmのalpha引数を決めれるようにする
void mat_multi2(double *a, double *b, double *a2, int size, double alpha);

//行列の和を計算する関数 n*n正方行列 aにbを加える
//lapackサブルーチンdaxpyを利用
//出力結果は5番目の引数、DYに入っている
//dy = dy + αdxを計算。α=1とすれば純粋な加算
//α=-1とすれば減算ができる
//sizeには行列の要素数を入れる
//2023/6/13 もともとint daの引数になってたものをdouble daに変更．
//関数内でdouble宣言をしてるので問題ないが，もし何か問題あったらここを確認
void mat_sum(double *a, double *b, int size, double da);

//行列の和(差)を計算する関数
//mat_sum関数とは異なり，cに結果が入るようにする
//da = -1とすれば差となる
//row_sizeには行のサイズ(正方行列を前提にしてるので列サイズでも変わらない)
void mat_sum2(double *a, double *b, double *c, int row_size, double da);

//入力行列が対称な場合のときに2つの行列の和を計算するための関数
void mat_sum_sym(double *a, double *b, int n);

//mat_sumでlapackルーチンを使わない版．これはlapack版との性能を比較したいときに使うのがいいかも
void mat_sum_osoi(double *a, double *b, int n);


//行列を出力する関数
void print_mat(double *a, int n, int m);

//行列の差を求める
void mat_diff(double *dif, double *A, double *sol, int size);

//サイズsize*sizeの正方行列aに実数xを掛ける関数
//行列のときsize = n*nが入るようにする。
void mat_multidouble(double *a, double x, int size);

//n*m行列, Iは単位行列
void setmatrix(double *A, double *I, int n, int m, double emax, double emin);

//seedを引数によって変えられるようにする
void setmatrix2(double *A, double *I, int n, int m, double emax, double emin, int seed);

//単位行列を計算しない関数
void setmatrix3(double *A, int n, double emax, double emin, int seed);

void setmatrix_complex(double *A, int n, double rmin, double rmax, double imin, double imax, int seed, double *ans);

/*
一般実行列
定義から解を得る関数
引数にansを加える
このようにすることで負の固有値が含まれる場合の解析解を計算できるようにする
char ch = 'n'とするとき，固有値に負が含まれるようにする
*/
void setmatrix_ans(double *A, int n, double emax, double emin, int seed, double *ans, char ch);

//aをシューア分解し，計算後シューア標準形が入る．vsにはシューアベクトル行列が入る
void schur(double *a, double *vs, int size);


void mat_T(double *a, double *at, int size);

void mat_diff(double *dif, double *A, double *sol, int size);

void file_read(char *filename, double *a, int n);

void identity(double *I, int n);

//単位行列との相対誤差を求める 正方行列とする．引数にはn*nのnを入れる(行か列のサイズ)．
//返り値として相対誤差を返す
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

/*
行列のノルムを計算するための関数
lapackのdlangeをラップしただけの関数なので性能はdlange依存
今までは純粋にノルムだけを計算する機会がなかったが，行列の差のノルムを計算する関数は作成済み

*引数--------------------------
double *A: 正方行列が格納された1次元実数配列

int matsize: 行列の行もしくは列サイズ

char norm: 1ノルム，フロベニウスノルム，infinityノルム，もしくはAの絶対値最大の要素を求めるための
           引数
           max(abs(A(i,j))): M or m
           norm1(A) : 1 or O or o
           normInf(A): I or i
           normF: F or f or E or e
-------------------------------

*/
double mat_norm(double *A, int matsize, char norm);

#endif
