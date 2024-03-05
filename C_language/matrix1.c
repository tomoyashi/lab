/*
メモ

2023/10/13
配列の確保を動的確保するように全面的に変更する
一応"matrix1_backup.c"にバックアップファイルを保存

--> やめた 並列化するとき動的に確保しておくとエラー起きる 何かいい方法あれば

*/

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "matrix1.h"
//#include <mpi.h>

//#define SIZE 10

//行列の列サイズだけでいいものとすべての要素の数を入れる必要がある関数があるので要注意

/*プロトタイプ宣言はヘッダファイルで行っているからここで書く必要はなかった
void dgemm_(char *transA, char *transB, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldB, double *beta, double *C, int *ldc);
void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
double dnrm2_(int *n, double *x, int *incx);
void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *B, int *ldb, int *info);
void dscal_(int *n, double *da, double *dx, int *incx);
void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
void dgees_(char *jobvs, char *sort, bool (*SELECT)(double, double), int *n, double *A, int *lda, int *sdim, double *wr, double *wi, double *vs, int *ldvs, double *work, int *lwork, bool *bwork, int *info);
void dgeqr2_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *info);
*/

void dgees_(char *jobvs, char *sort, bool (*SELECT)(double, double), int *n, double *A, int *lda, int *sdim, double *wr, double *wi, double *vs, int *ldvs, double *work, int *lwork, bool *bwork, int *info);
bool SELECT(const double wr, const double wi){
    return true;
}

//引数の行列aを逆行列にする関数
void matrix_inv(double *a, int size){
    int n = size; //行のサイズ
    int m = size; //列のサイズ
    int lda = size; //mと同じ値
    int info; //計算が成功すれば0を返す
    int ipiv[size]; //要素数はm, nのうち小さいほうとする
    int lwork = size; //nと同じ値
    double work[size]; //要素数はlworkと同じ値
    dgetrf_(&n, &m, a, &lda, ipiv, &info);
    dgetri_(&m, a, &lda, ipiv, work, &lwork, &info);
}

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
void matrix_inv2(double *a, double *ainv, int size){
    int info, ipiv[size];
    identity(ainv, size); /*ainvには単位行列を入れておく*/
    double *acpy = (double*)(malloc(sizeof(double)*size*size)); /*aをそのまま用いると，出力が変ってしまうのでコピーして利用*/
    matcpy(a, acpy, size);
    dgesv_(&size, &size, acpy, &size, ipiv, ainv, &size, &info);
    free(acpy);
}

/*ansには単位行列が入っているようにする
*実行後 aが変ってしまうことに注意*
n:行列の行か列の大きさ
計算後ansに逆行列が入る
aには逆行列にした行列(計算後はLU分解された形になる)
ansに単位行列を入れておくようにすると，計算後ansにはaの逆行列が入る
matrix_inv2
*/
void matrix_inv3(double *a, int n, double *ans){
    int info, ipiv[n];
    dgesv_(&n, &n, a, &n, ipiv, ans, &n, &info);
    return;
}


//行列A^2を求める, a2に結果が入る
//sizeには正方行列R^(n*n)のnが入る
void matrixA(double a[], double a2[], int size){
    double alpha = 1.0, beta = 0.0;
    int n = size;
    dgemm_("n", "n", &n, &n, &n, &alpha, a, &n, a, &n, &beta, a2, &n);

    return;
}

//行列の積を計算 計算結果は3引数目の変数に入る
//sizeには行列の行数
void mat_multi(double a[], double b[], double a2[], int size){
    double alpha = 1.0, beta = 0.0;
    int n = size;
    dgemm_("n", "n", &n, &n, &n, &alpha, a, &n, b, &n, &beta, a2, &n);

    return;
}

//行列の積を計算 計算結果は3引数目の変数に入る
//sizeには行列の行数
//C = alpah*AB とdgemmのalpha引数を決めれるようにする
void mat_multi2(double *a, double *b, double *a2, int size, double alpha){
    double beta = 0.0;
    int n = size;
    dgemm_("n", "n", &n, &n, &n, &alpha, a, &n, b, &n, &beta, a2, &n);

    return;
}

//行列の和を計算する関数 n*n正方行列 aにbを加える a = a + b
//lapackサブルーチンdaxpyを利用
//出力結果は5番目の引数、DYに入っている(ここではdaxpyの5番目の引数には配列aを入れてるのでaの結果が更新される)
//dy = dy + αdxを計算。α=1(ここではda)とすれば純粋な加算
//sizeのところには行列の要素数を入れる
//2023/6/13 もともとint daの引数になってたものをdouble daに変更．
//関数内でdouble宣言をしてるので問題ないが，もし何か問題あったらここを確認
void mat_sum(double a[], double b[], int size, double da){
    int n=size, incx = 1, incy = 1;
    double da1= da; 
    daxpy_(&n, &da1, b, &incx, a, &incy);
    return;
}

//行列の和(差)を計算する関数
//mat_sum関数とは異なり，cに結果が入るようにする
//da = -1とすれば差となる
//row_sizeには行のサイズ(正方行列を前提にしてるので列サイズでも変わらない)
void mat_sum2(double *a, double *b, double *c, int row_size, double da){
    matcpy(a, c, row_size); /*cにaの内容をコピー*/
    int n = row_size*row_size, incx = 1, incy = 1; /*daxpyに与える変数を定義*/
    daxpy_(&n, &da, b, &incx, c, &incy);
    return;
}

//ある行列に対称行列を足すための関数
//対称性を利用する
//nは行数(正方行列)
//a = a + b
void mat_sum_sym(double a[], double b[], int n){
    for(int i=0; i < n; i++){
        for(int j = i; j < n; j++){
            if(i == j){
                a[i + j*n] += b[i + j*n];
            }else{
                a[i+j*n] += b[i+j*n];
                a[j + i*n] += b[i + j*n];
            }
        }
    }
}

void mat_sum_osoi(double a[], double b[], int n){
    for(int i=0; i<n*n; i++){
        a[i] += b[i];
    }
}


//行列を出力する関数
void print_mat(double a[], int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%f ", a[i + j*m]);
        }
        printf("\n");
    }
    printf("\n");
    return ;
}


//サイズsize*sizeの正方行列aに実数xを掛ける関数
//行列のときsize = n*nが入るようにする。
void mat_multidouble(double a[], double x, int size){
    int s = size, incx = 1;
    dscal_(&s, &x, a, &incx);
    return;
}


//n*m行列, Iは単位行列
void setmatrix(double A[], double I[], int n, int m, double emax, double emin){
    double X[n*m]; //まず適当な行列を作成する
    double Xinv[n*m];
    srand(2);//計算結果比較のためシード固定
    //srand((unsigned)time(NULL));
    for(int i = 0; i < n*m; ++i){
        X[i] = ((double)rand()/ ((double)RAND_MAX + 1));
        Xinv[i] = X[i];
    }
    //print_mat(X, n, n);
    double a[n*m];//ここ初期化すればよくないか？後で確認
    for(int i = 0; i < n*m; i++){
        a[i] = 0;
        I[i] = 0;
    }
    //ここ上のループでできるかも
    for(int i = 0; i < n; i++){
        I[i*n+i] = 1;
        a[i*n + i] = ((double)rand()/ ((double)RAND_MAX))*(emax-emin) + emin;
    }
    a[0] = emax; a[1*n + 1] = emin;//対角成分はi*n + i  ここでは固有値を好みの値にセット
    
    matrix_inv(Xinv, n);
    //print_mat(Xinv, n, n);
    double tmp[n*m];
    mat_multi(X, a, tmp, n);
    mat_multi(tmp, Xinv, A, n);   

    return ;
}

//行列生成シードを引数で指定できるようにする
void setmatrix2(double A[], double I[], int n, int m, double emax, double emin, int seed){
    double X[n*m]; //まず適当な行列を作成する
    double Xinv[n*m];
    srand(seed);//計算結果比較のためシード固定
    //srand((unsigned)time(NULL));
    for(int i = 0; i < n*m; ++i){
        X[i] = ((double)rand()/ ((double)RAND_MAX + 1));
        Xinv[i] = X[i];
    }
    //print_mat(X, n, n);
    double a[n*m];//ここ初期化すればよくないか？後で確認
    for(int i = 0; i < n*m; i++){
        a[i] = 0;
        I[i] = 0;
    }
    //ここ上のループでできるかも
    for(int i = 0; i < n; i++){
        I[i*n+i] = 1;
        a[i*n + i] = ((double)rand()/ ((double)RAND_MAX))*(emax-emin) + emin;
    }
    a[0] = emax; a[1*n + 1] = emin;//対角成分はi*n + i  ここでは固有値を好みの値にセット
    
    matrix_inv(Xinv, n);
    //print_mat(Xinv, n, n);
    double tmp[n*m];
    mat_multi(X, a, tmp, n);
    mat_multi(tmp, Xinv, A, n);   

    return ;
}

//setmatrix関数で今まで単位行列を作成していたが，誤差ノルムを求める関数を作成したため，その必要がなくなった．
//行列は正方行列
void setmatrix3(double A[], int n, double emax, double emin, int seed){
    double X[n*n], a[n*n]; //まず適当な行列を作成する
    double Xinv[n*n];
    srand(seed);//計算結果比較のためシード固定
    //srand((unsigned)time(NULL));
    for(int i = 0; i < n*n; ++i){
        X[i] = ((double)rand()/ ((double)RAND_MAX + 1));
        Xinv[i] = X[i];
    }
    //print_mat(X, n, n);
    for(int i=0; i < n*n; i++){
        if(i == 0){ //1つ目の対角成分
            a[i] = emax;
        }
        else if(i == (1*n+1)){ //2つ目の対角成分
            a[i] = emin;
        }
        else if((i/n)==(i%n)){ //それ以外の対角成分
            a[i] = ((double)rand()/ ((double)RAND_MAX))*(emax-emin) + emin;
        }
        else{ //対角成分以外は0
            a[i] = 0.0;
        }
    }
    
    matrix_inv(Xinv, n);
    double tmp[n*n];
    mat_multi(X, a, tmp, n); //tmp = X@a
    mat_multi(tmp, Xinv, A, n); //A = tmp@Xinv

    return ;
}

/*
一般実行列
定義から解を得る関数
引数にansを加える
このようにすることで負の固有値が含まれる場合の解析解を計算できるようにする
char ch = 'n'とするとき，固有値に負が含まれるようにする

2023_12_04 行列Aとか以外は動的に確保して，この関数内で解放するようにする

*/
void setmatrix_ans(double *A, int n, double emax, double emin, int seed, double *ans, char ch){
    double *X, *a, *Xinv;
    X = (double *)malloc(sizeof(double)*n*n);
    a = (double *)malloc(sizeof(double)*n*n);
    Xinv = (double *)malloc(sizeof(double)*n*n);
    //double X[n*n], a[n*n], Xinv[n*n]; 
    
    srand(seed);//計算結果比較のためシード固定
    //srand((unsigned)time(NULL));
    for(int i = 0; i < n*n; ++i){
        X[i] = ((double)rand()/ ((double)RAND_MAX + 1));
    }
    matcpy(X, Xinv, n);

    //print_mat(X, n, n);
    for(int i=0; i < n*n; i++){
        if(i == 0){ //1つ目の対角成分
            a[i] = emax;
        }
        else if(i == (1*n+1)){ //2つ目の対角成分
            a[i] = emin;
        }
        else if((i/n)==(i%n)){ //それ以外の対角成分
            a[i] = ((double)rand()/ ((double)RAND_MAX))*(emax-emin) + emin;
            if(ch == 'n'){
                a[i] = -a[i];
            }
        }
        else{ //対角成分以外は0
            a[i] = 0.0;
        }
    }
    
    matcpy(a, ans, n);
    sign_jordan(ans, X, n);

    matrix_inv(Xinv, n);
    double *tmp;
    tmp = (double *)malloc(sizeof(double)*n*n);
    mat_multi(X, a, tmp, n); //tmp = X@a
    mat_multi(tmp, Xinv, A, n); //A = tmp@Xinv

    free(X);
    free(a);
    free(Xinv);
    free(tmp);

    return ;
}

/*
複素固有値を含む行列を生成
ある複素固有値 a + bi (iは虚数単位)があるとする．
このとき，行列がA = XΛX^-1のように分解できるとき，
Λは対角成分に
[a, -b; b, a]
という行列が並ぶような行列である 


実部の範囲と虚部の範囲を決めるパラメータが必要
real part [rmin, rmax]
imaginaly part [imin, imax]

Parameter:
--------------------
double *A; [in, out] n*nサイズの正方行列
int n: [in] 行列サイズ
double rmin: [in] 固有値の実部の最小値
double rmax: [in] 固有値の実部の最大値
double imin: [in] 固有値の虚部の最小値
double imax: [in] 固有値の虚部の最大値
int seed: 乱数のシード値
double *ans: [in, out] n*nサイズの正方行列 入力固有値のときの行列符号関数の定義による解が入る
--------------------
*/
void setmatrix_complex(double *A, int n, double rmin, double rmax, double imin, double imax, int seed, double *ans){
    //A = X@a@Xinvで計算する．
    double *X, *a, *Xinv, *I;
    int matsize = n*n;
    X = (double *)malloc(sizeof(double)*matsize);
    a = (double *)calloc(matsize, sizeof(double)); //aはcallocで確保することで0で初期化
    Xinv = (double *)malloc(sizeof(double)*matsize);
    I = (double *)malloc(sizeof(double)*matsize);
    identity(I, n);

    srand(seed);//計算結果比較のためシード固定
    //srand((unsigned)time(NULL));
    for(int i = 0; i < n*n; ++i){
        X[i] = ((double)rand()/ ((double)RAND_MAX + 1));
    }
    matcpy(X, Xinv, n);

    //行列aの値を決める


    //iの増分が+2であることに注意
    for(int i=0; i < n; i+=2){
        //各ループで固有値の実部と虚部をランダムに決める(引数で与えられた範囲内で)
        double r_eig, i_eig;
        r_eig = ((double)rand()/ ((double)RAND_MAX))*(rmax - rmin) + rmin;
        i_eig = ((double)rand()/ ((double)RAND_MAX))*(imax - imin) + imin;
        a[i*n + i] = r_eig;
        a[(i+1)*n + i+1] = r_eig;
        a[(i+1)*n + i] = -i_eig;
        a[i*n + i+1] = i_eig;
    }
    
    //解析解を計算する
    //複素固有値の実部はすべて対角成分に並ぶため，sign_jordanで問題なく計算できる
    matcpy(a, ans, n);
    sign_jordan(ans, X, n);

    matrix_inv3(Xinv, n, I);
    double *tmp;
    tmp = (double *)malloc(sizeof(double)*matsize);
    mat_multi(X, a, tmp, n); //tmp = X@a
    mat_multi(tmp, I, A, n); //A = tmp@Xinv

    free(tmp);
    free(X);
    free(a);
    free(Xinv);
    free(I);

    return ;
}



//シューア分解を行う関数
//aをシューア分解し，計算後シューア標準形が入る．vsにはシューアベクトル行列が入る
void schur(double *a, double *vs, int size){
    double wr[size], wi[size], work[3*size];
    char jobvs = 'V', sort = 'N';
    int n = size, lda = size, ldvs = size, lwork = 3*size, sdim, info;
    bool bwork[size];
    dgees_(&jobvs, &sort, SELECT, &n, a, &lda, &sdim, wr, wi, vs, &ldvs, work, &lwork, bwork, &info);
    return;
}

//行列aの転置をatに入れる sizeは行の長さ
/*
void mat_T(double *a, double *at, int size){
    for(int i = 0; i < size; i++){
        at[i*size + i] = a[i*size + i];
    }
    //print_mat(a, size, size);
    for(int j = 1; j < size; j++){
        for(int i = j-1; i >= 0; i--){
            at[i*size + j]= a[j*size + i];
            at[j*size + i] = a[i*size + j];
            //printf("%d %d ", i*size + j, j*size + i);
        }
    }
    return;
}
*/

void mat_T(double *a, double *at, int size){
    for(int i=0; i < size; i++){
        for(int j = i; j < size; j++){
            if(i == j){
                at[i + j*size] = a[i + j*size];
            }else{
                at[i+j*size] = a[j+i*size];
                at[j + i*size] = a[i + j*size];
            }
        }
    }
}


//行列の差を求める
void mat_diff(double dif[], double A[], double sol[], int size){
    for(int i=0; i < size; i++){
        dif[i] = A[i] - sol[i]; 
    }
    return;
}


//nは行のサイズ，aにはファイルの値がlapack用に入れられる．filenameはファイル名(拡張子含む)
void file_read(char *filename, double *a, int n){
    FILE *fp;
    if((fp = fopen(filename, "r")) == NULL){
        printf("file can not open\n");
        exit(1);
    }
    //無限ループ怖い．注意
    int mat_row_num = 0, mat_column_num = 0;
    for(;;){
        /*
        bufに行を読み込むバッファオーバーランするかもしれない．
        おかしくなったらここ確認
        エラー発生したらbufのサイズ大きくする方がよき
        buf
        1つの数字あたり18文字あると考えて，それがn個あり，空白がn個あり，一応10を足しておく．
        */
        char buf[22*n+n];

        if(fgets(buf, sizeof(buf), fp) == NULL){
            if(feof(fp)){
                break;
            }
            else{
                fputs("error happen!\n", stderr);
                exit(EXIT_FAILURE);
            }
        }

        //末尾が改行文字であれば，'\0'で上書きする
        char* p = strchr(buf, '\n');
        if(p != NULL){
            *p = '\0';
        }

        /*
        以下で行をdelimiterで区切られた数値をそれぞれ配列に入れていく．
        ここでは，delimeterは空白となっている．データファイルの種類によって変更する．
        stは各数値の始まりの位置を覚えておく．
        文字を走査していき，delimiterにあったらbufのstポインタからi-st+1(これが数値の長さ)をtmpに格納する．
        tmpをatofによってdouble型に変換．
        atofはエラー判定ができないため注意．strtofの方が安全そうである．
        */
        //puts(buf);
        int st = 0;
        for(int i = 0; buf[i] != '\0'; i++){
            if(buf[i+1] == '\0'){
                char tmp[32];
                strncpy(tmp, buf+st, i - st + 1);
                double num = atof(tmp);
                a[mat_row_num + n*mat_column_num] = num;
                //printf("%f ", num);
                mat_column_num = 0;
            }
            else if(buf[i] == ' '){
                char tmp[32];
                strncpy(tmp, buf+st, i - st);
                double num = atof(tmp);
                a[mat_row_num + n*mat_column_num] = num;
                //printf("%f ", a[mat_num]);
                st = i+1;
                mat_column_num++;
            }
        }
        mat_row_num++;
        //free(buf);
    }
    if(fclose(fp) == EOF){
        fputs("file cant close!!\n", stderr);
        exit(EXIT_FAILURE);
    }
}


//関数実行後，配列Iに単位行列が格納される．sizeは行の大きさ(正方行列なので列も同じ)
void identity(double *I, int n){
    int i;
    for(i=0; i < n*n; i++){
        I[i] = 0.0;
    }
    for(i=0; i < n; i++){
        I[i*n + i] = 1.0;
    }
}

//単位行列との相対誤差を求める 正方行列とする．引数にはn*nのnを入れる(行か列のサイズ)．
//返り値として相対誤差を返す
double calc_rerr_id(double *A, int size){
    //incxは1にすることで全要素の計算ができる
    int nn = size*size, incx = 1;
    double dif[size*size], I[size*size], normdif, normsol;
    //対角成分のときは1でそれ以外は0なので，それぞれから引くようにする
    for(int i = 0; i < size*size; i++){
        if(i%size == (i/size)){//対角成分のとき
            dif[i] = 1.0 - A[i];
            I[i] = 1.0;
        }
        else{
            dif[i] = 0.0 - A[i];
            I[i] = 0.0;
        }
    }

    normdif = dnrm2_(&nn, dif, &incx);
    normsol = dnrm2_(&nn, I, &incx);

    return normdif/normsol;
}

//単位行列の絶対誤差を求める 正方行列とする．引数にはn*nのnを入れる．(行か列のサイズ)
//返り値として絶対誤差を求める
double calc_err_id(double *A, int size){
    //incxは1にすることで全要素の計算ができる
    int nn = size*size, incx = 1;
    double dif[size*size], normdif;
    //対角成分のときは1でそれ以外は0なので，それぞれから引くようにする
    for(int i = 0; i < size*size; i++){
        if(i%size == (i/size)){//対角成分のとき
            dif[i] = 1.0 - A[i];
        }
        else{
            dif[i] = 0.0 - A[i];
        }
    }

    normdif = dnrm2_(&nn, dif, &incx);

    return normdif;
}

//以下の関数はこのファイル内でのみ使用するのでstatic関数とする．
//もし，他でも使用する場合は，staticを外してヘッダファイルで関数宣言を行う．
/*
static int max_val(int n1, int n2){
    return (n1 >= n2) ? n1: n2;
}*/

//qr分解する．rのところに入力行列を入れる
void qr(double *q, double *r, int size){
    int n = size, lda = size, info, lwork=-1;

    double tau[n] , work_query;

    dgeqrf_(&n, &n, r, &lda, tau, &work_query, &lwork, &info);
    lwork = (int)work_query;
    double work[lwork];
    dgeqrf_(&n, &n, r, &lda, tau, work, &lwork, &info);
    matcpy(r, q, n); //rをコピーしてqをdorgqrによって求める
    dorgqr_(&n, &n, &n, q, &lda, tau, work, &lwork, &info);
    //print_mat(q, n, n);
    
    //rは上三角部分のみ．それ以外のところも利用したいときはプログラムを変える
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i > j){
                r[i + j*n] = 0.0;
            }
        }
    }
    //print_mat(q, n, n);  
}



/*---------------------------------------------------
2つの行列(ベクトル)の差のフロベニウスノルムを返す関数
正方行列

daxpyを使う

dlangeはdnrm2と違って引数normを変更することで異なる行列ノルムが計算できる．
なので，もし必要であればdif_normの引数にもchar型の引数を入れることによって関数側から変更できるようにしてもいいかもしれない
nは行列の列か行のサイズ
---------------------------------------------------------*/
double dif_norm(double *A, double *B, int n){
    //まずA-Bを求める
    double dif[n*n], da = -1.0, work[n], ans;
    memcpy(dif, A, sizeof(dif));
    int incx = 1, incy = 1, nn = n*n;
    daxpy_(&nn, &da, B, &incx, dif, &incy);
    //ノルムの計算
    char norm = 'f';
    ans = dlange_(&norm, &n, &n, dif, &n, work);

    return ans;
}

//正方行列AをBにコピー nは行か列サイズ
void matcpy(double *A, double *B, int n){
    char uplo = 'a'; //aじゃくなてもなんでもいい
    dlacpy_(&uplo, &n, &n, A, &n, B, &n);
    return ;
}

/*--------------------------------------------------------------------------
固有値を事前に設定した行列Aは次のように作られる
A = XΛX^(-1)
ここでXについても条件数を指定して作られるようにしたい．
まず条件数最低の1の場合はXが直交行列となり，Aは対称行列となる．
Xの条件数が大きくなるほど対称から遠のいていくようになるだろう

今まで同様にΛについても最大固有値，最小固有値を決める

*正方行列とする
引数: double *A(in, out),int n(row size), double emax(max e'value),
      double emin(min e'value), int seed(random seed), double cond(X's condition number)

流れ
    条件数condの行列Xを作成する(条件数1の場合は)
        if(cond == 1)X = Q
        else X = BbB^(-1) このbは条件数がcondになるように調整する
            diag(b) = {x: 1 < x < cond}
    
    行列Bはまずランダムに生成し，それをQR分解して行列Qを用いる(直交行列が使いたい)
        なぜ直交行列なのか？行列が正規行列でないと固有値の幅で条件数を推定できないため
    上記で作成したXを用いて固有値分布が(emin, emax)の行列を作成する
    A = XaX^(-1)


2023/2/20
引数にqrsを追加
qrs = 'Y': このとき，BをQR分解してQを新たなBとする
qrs = 'N': QR分解をしない
----------------------------------------------------------------------------*/

void setmatrix_X(double *A, int n, double emax, double emin, int seed, double cond, char qrs){
    int mat_size = n*n;
    double B[mat_size], Binv[mat_size], X[mat_size], Xinv[mat_size], b[mat_size], a[mat_size];
    srand(seed);
    //srand((unsigned)time(NULL));

    //以下のループでBおよびBinv, a, bを作成
    for(int i = 0; i < mat_size; i++){
        B[i] = (double)rand()/((double)RAND_MAX+1);
        Binv[i] = B[i];

        if(i == 0){//1つ目の対角成分
            b[i] = cond;
            a[i] = emax;
        }
        else if(i == (1)*n+1){//最後の対角成分
            b[i] = 1.0;
            a[i] = emin;
        }
        else if((i/n) == (i%n)){//それ以外の対角成分
            b[i] = ((double)rand()/((double)RAND_MAX))*(cond-1.0) + 1.0;
            a[i] = ((double)rand()/ ((double)RAND_MAX))*(emax-emin) + emin;
        }
        else{//対角成分以外は0
            b[i] = 0.0;
            a[i] = 0.0;
        }
    }
    if(qrs == 'Y'){
        double q[mat_size], r[mat_size];
        matcpy(B, r, n);
        qr(q, r, n);
        matcpy(q, B, n);
        matcpy(B, Binv, n);
    }

    matrix_inv(Binv, n);
    double tmp[mat_size];
    mat_multi(B, b, tmp, n);
    mat_multi(tmp, Binv, X, n);//この段階でX完成
    //XをXinvにコピーして逆行列を作成する
    matcpy(X, Xinv, n);
    matrix_inv(Xinv, n);
    mat_multi(X, a, tmp, n);
    mat_multi(tmp, Xinv, A, n);

    return ;
}

/*-----------------------------------------------------------

ランダムで固有値範囲が指定された範囲内である対称行列を作成

    まずランダムに行列Xを作成
    そのXをQR分解
    Qは直交行列なので・・・
    QaQ^(-1)は対称行列となる

--------------------------------------------------------------*/

void setmatrix_sym(double *A, int n, double emax, double emin, int seed){
    double X[n*n], a[n*n], q[n*n], r[n*n]; //まず適当な行列を作成する
    double qinv[n*n];
    srand(seed);//計算結果比較のためシード固定
    //srand((unsigned)time(NULL));
    for(int i = 0; i < n*n; ++i){
        X[i] = ((double)rand()/ ((double)RAND_MAX + 1));
    }
    matcpy(X, r, n);
    qr(q, r, n);
    matcpy(q, qinv, n);
    //print_mat(X, n, n);
    for(int i=0; i < n*n; i++){
        if(i == 0){ //1つ目の対角成分
            a[i] = emax;
        }
        else if(i == (1*n+1)){ //2つ目の対角成分
            a[i] = emin;
        }
        else if((i/n)==(i%n)){ //それ以外の対角成分
            a[i] = ((double)rand()/ ((double)RAND_MAX))*(emax-emin) + emin;
            //printf("%f\n", a[i]);
        }
        else{ //対角成分以外は0
            a[i] = 0.0;
        }
    }
    
    matrix_inv(qinv, n);
    double tmp[n*n];
    mat_multi(q, a, tmp, n); //tmp = X@a
    mat_multi(tmp, qinv, A, n); //A = tmp@Xinv

    return ;
}

double normf(double *A, int n){
    int nn = n*n, incx = 1;
    return dnrm2_(&nn, A, &incx);
}

/*
定義から解を得る関数
引数にansを加える
このようにすることで負の固有値が含まれる場合の解析解を計算できるようにする
char ch = 'n'とするとき，固有値に負が含まれるようにする
*/
void setmatrix_sym2(double *A, int n, double emax, double emin, int seed, double *ans, char ch){
    double X[n*n], a[n*n], q[n*n], r[n*n]; 
    double qinv[n*n];
    srand(seed);//計算結果比較のためシード固定
    //srand((unsigned)time(NULL));
    for(int i = 0; i < n*n; ++i){
        X[i] = ((double)rand()/ ((double)RAND_MAX + 1));
    }
    matcpy(X, r, n);
    qr(q, r, n);
    matcpy(q, qinv, n);
    //print_mat(X, n, n);
    for(int i=0; i < n*n; i++){
        if(i == 0){ //1つ目の対角成分
            a[i] = emax;
        }
        else if(i == (1*n+1)){ //2つ目の対角成分
            a[i] = emin;
        }
        else if((i/n)==(i%n)){ //それ以外の対角成分
            a[i] = ((double)rand()/ ((double)RAND_MAX))*(emax-emin) + emin;
            if(ch == 'n'){
                a[i] = -a[i];
            }
        }
        else{ //対角成分以外は0
            a[i] = 0.0;
        }
    }
    
    matcpy(a, ans, n);
    sign_jordan(ans, q, n);

    matrix_inv(qinv, n);
    double tmp[n*n];
    mat_multi(q, a, tmp, n); //tmp = X@a
    mat_multi(tmp, qinv, A, n); //A = tmp@Xinv

    return ;
}


/*
固有値分解できているときは行列符号関数を定義から計算できる
固有値が対角成分に並んだ対角行列と固有ベクトル行列，行列サイズn*nのnを引数に取る

double *eig[in, out]: 対角行列．計算後，結果はここに入る
*/
void sign_jordan(double *eig, double *evec, int size){
    double evec_inv[size*size], tmp[size*size];
    
    //sign(eig)
    for(int i=0; i < size; i++){
        double eigv = eig[i + size*i];
        if(eigv > 0){
            eig[i+size*i] = 1.0;
        }
        else if(eigv < 0){
            eig[i + size*i] = -1.0;
        }
        else{
            printf("error: eigenvalue can not be zero. %lf eig[%d]\n", eigv, i);
            return ;
        }
        //0の場合はエラー(double型で0になることもないとは思うけど)
    }
    
    matcpy(evec, evec_inv, size);
    matrix_inv(evec_inv, size);

    mat_multi(evec, eig, tmp, size);
    mat_multi(tmp, evec_inv, eig, size);
    return ;
}


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
double mat_norm(double *A, int matsize, char norm){
    double work[matsize];
    return dlange_(&norm, &matsize, &matsize, A, &matsize, work);
}
