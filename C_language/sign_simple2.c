#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "matrix1.h"
#include "mytimer.h"

//#define SIZE 10
static double eps = 1e-15;

/*
逐次計算DE公式

2023/10/13
func1_2関数, deformula2関数を追加
被積分関数を変形した後にDE公式を使用
*/


void func1(double x, double *AA, int n, double *ans){
    for(int i=0; i<n*n; i++){
        //対角成分かどうかで場合分け
        if((i/n) == (i%n)){
            ans[i] = x*x + AA[i];
        }else{
            ans[i] = AA[i];
        }
    }

    return ;
}

/*
被積分変形後
AinvにAの逆行列
ansに被積分関数の行列が入るようにする
nは行列サイズ(行)
*/
///*
void func1_2(double x, double *A, double *Ainv, int n, double *ans){
    int i, nn = n*n;
    double tmp[nn], xx = x*x;
    for(i = 0; i < nn; i++){
        tmp[i] = xx*Ainv[i] + A[i];
        //ans[i] = 0.0;
    }
    
    for(i=0; i < n; i++){
        ans[i*n + i] = 1.0;
    }
    identity(ans, n);
    //matrix_inv(ans, n);
    matrix_inv3(tmp, n, ans);
    return;
}//*/
/*
void func1_2(double x, double *A, double *Ainv, int n, double *ans){
    int i, nn = n*n;
    double  xx = x*x;
    for(i = 0; i < nn; i++){
        ans[i] = xx*Ainv[i] + A[i];
        //ans[i] = 0.0;
    }
    matrix_inv(ans, n);
    //matrix_inv(ans, size);
    return;
}*/

/*
double func2
変換関数φ(x)について引数にxの値を渡して結果を返す
*/

double func2(double x){
    return exp(0.5*M_PI*sinh(x));
}

/*
double func3
変換関数を微分した関数について引数にxの値を渡して結果を返す
φ'(x) 
*/
double func3(double x){
    return M_PI*0.5*exp(0.5*M_PI*sinh(x))*cosh(x);
}


/*
void deformula()
DE公式によって積分計算をする関数

    h  :  刻み幅
    n  :  行列(A:n*n)サイズ
    A  :  行列Aの配列
    AA :  行列A@Aの配列
    N  :  最大ループ回数(無限ループ回避用)
    ans:  計算結果の入る配列
*/

void deformula(double h, int n, double *A, double *AA, int N, double *ans, int *npm){
    //まずx=0のときの計算をして初期化する
    int mat_size = n*n, ipiv[n], i, info;
    double tmp1[mat_size];

    matcpy(A, ans, n);
    func1(func2(0.0), AA, n, tmp1);
    dgesv_(&n, &n, tmp1, &n, ipiv, ans, &n, &info);
    mat_multidouble(ans, func3(0.0), mat_size);
    //printf("%lf", func3(0.0));
    npm[0] = N;
    npm[1] = N;
    for(i=1; i < N; i++){
        double x = i*h, tmp_array[mat_size];
        //printf("j=%d f2=%f\n", j, func2(x));
        memcpy(tmp_array, A, sizeof(tmp_array));
        func1(func2(x), AA, n, tmp1);
        dgesv_(&n, &n, tmp1, &n, ipiv, tmp_array, &n, &info);
        mat_multidouble(tmp_array, func3(x), mat_size);
        mat_sum(ans, tmp_array, mat_size, 1);
        if(normf(tmp_array, n) <= eps){
            npm[0] = i;
            break;
        }
    }
    for(i=1; i < N; i++){
        double x = -i*h, tmp_array[mat_size];
        //printf("j=%d f2=%f\n", j, func2(x));
        memcpy(tmp_array, A, sizeof(tmp_array));
        func1(func2(x), AA, n, tmp1);
        dgesv_(&n, &n, tmp1, &n, ipiv, tmp_array, &n, &info);
        mat_multidouble(tmp_array, func3(x), mat_size);
        mat_sum(ans, tmp_array, mat_size, 1);
        if(normf(tmp_array, n) <= eps){
            npm[1] = i;
            break;
        }
    }
    mat_multidouble(ans, 2.0*h/M_PI, mat_size);
}


/*
被積分関数変形後の場合のDE公式関数
*/
void deformula2(double h, int n, double *A, double *Ainv, int N, double *ans, int *npm){
    //まずx=0のときの計算をして初期化する
    int mat_size = n*n, i;
    //double tmp1[mat_size];

    matcpy(A, ans, n);
    func1_2(func2(0.0), A, Ainv, n, ans);
    mat_multidouble(ans, func3(0.0), mat_size);
    //printf("%lf", func3(0.0));
    npm[0] = N;
    npm[1] = N;
    for(i=1; i < N; i++){
        double x = i*h, tmp_array[mat_size], dif_norm;
        func1_2(func2(x), A, Ainv, n, tmp_array);
        mat_multidouble(tmp_array, func3(x), mat_size);
        mat_sum(ans, tmp_array, mat_size, 1);
        dif_norm = normf(tmp_array, n);
        //printf("test:i%d %f\n", i, dif_norm);
        ///*
        if(dif_norm <= eps){
            npm[0] = i;
            break;
        }
    }
    for(i=1; i < N; i++){
        double x = -i*h, tmp_array[mat_size], dif_norm;
        func1_2(func2(x), A, Ainv, n, tmp_array);
        mat_multidouble(tmp_array, func3(x), mat_size);
        mat_sum(ans, tmp_array, mat_size, 1);
        dif_norm = normf(tmp_array, n);
        
        if(dif_norm <= eps){
            npm[1] = i;
            break;
        }
    }
    mat_multidouble(ans, 2.0*h/M_PI, mat_size);
}

int main(int argc, char* argv[]){
    int n = 1000, N = 40, npm[2], loop =1, mat_size=n*n;

    double emin = 0.1, emax = 10, A[mat_size], AA[mat_size], ans[mat_size], Ainv[mat_size];
    //double tmp[n*n];
    //char ch = 'n';
    //setmatrix_sym2(A, n, emax, emin, 2, tmp, ch);
    //setmatrix2(A, tmp, n, n, emax, emin, 1);
    //double c = 1/sqrt(emin*emax);
    setmatrix_sym(A, n, emax, emin, 1);
    //mat_multidouble(A, c, n*n);
    double h = 0.1;
    //print_mat(tmp, n, n);
    matcpy(A, Ainv, n);
    matrix_inv(Ainv, n); //A^-1の計算
    //matrixA(A, AA, n);

    for(int i = 0; i < loop; i++){
        double etime;
        timer_start();
        //deformula(h, n, A, AA, N, ans, npm);
        deformula2(h, n, A, Ainv, N, ans, npm);
        etime = timer_end();
        double err = calc_err_id(ans, n);
        //double err = dif_norm(ans, tmp, n);
        printf("%d %lf %.14lf %lf %d %lf %d\n", n, h, err, log10(err), npm[0]+npm[1]+1, etime, (int)(emax/emin));
        h = h*0.9;
    }

    return 0;
}
