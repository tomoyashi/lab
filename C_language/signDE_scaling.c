#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "matrix1.h"
#include "mytimer.h"

static double eps = 1e-15;

/*
逐次計算DE公式
スケーリングを行えるようにする
*/



void func1_imp(double x, double *A, double *Ainv, int n, double *ans){
    for(int i=0; i<n*n; i++){
        ans[i] = x*x*Ainv[i] + A[i];
    }
    matrix_inv(ans, n);
    return ;
}

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
被積分関数変形後の行列符号関数の積分表示形式に対してDE公式を適用する関数
*/
void deformula(double h, int n, double *A, double *Ainv, int N, double *ans, int *npm){
    //まずx=0のときの計算をして初期化する
    int mat_size = n*n, ipiv[n], i, info;
    double tmp1[mat_size];
    //double emax = mat_norm(A, )

    func1_imp(func2(0.0), A, Ainv, n, ans);
    mat_multidouble(ans, func3(0.0), mat_size);
    //printf("%lf", func3(0.0));
    npm[0] = N;
    npm[1] = N;
    double x;
    for(i=1; i < N; i++){
        x = i*h;
        //printf("j=%d f2=%f\n", j, func2(x));
        func1_imp(func2(x), A, Ainv, n, tmp1);
        mat_multidouble(tmp1, func3(x), mat_size);
        mat_sum(ans, tmp1, mat_size, 1);
        if(normf(tmp1, n) <= eps){
            npm[0] = i;
            break;
        }
    }
    for(i=1; i < N; i++){
        x = -i*h;
        //printf("j=%d f2=%f\n", j, func2(x));
        func1_imp(func2(x), A, Ainv, n, tmp1);
        mat_multidouble(tmp1, func3(x), mat_size);
        mat_sum(ans, tmp1, mat_size, 1);
        if(normf(tmp1, n) <= eps){
            npm[1] = i;
            break;
        }
    }
    mat_multidouble(ans, 2.0*h/M_PI, mat_size);
}


int main(int argc, char* argv[]){
    int n = 1000, N = 10000, npm[2], loop;

    double emin = 0.01, emax = 100, A[n*n], ans[n*n], Ainv[n*n], def_ans[n*n];
    double emin_s, emax_s; //近似最大固有値，最小固有値

    double c = 1/sqrt(emin*emax), c_s, AnormI, AinvnormI; //実際の固有値でスケーリング値cを計算
    setmatrix_ans(A, n, emax, emin, 3, def_ans, 'n'); //固有値に負を含む行列を作成
    //以下でAとAinvのinftyノルムを計算．
    //スケーリングにも条件数推定にも使える
    AnormI = mat_norm(A, n, 'i');
    matcpy(A, Ainv, n);
    matrix_inv(Ainv, n);
    AinvnormI = mat_norm(Ainv, n, 'i');

    //以下で近似スケーリングのために必要な値を計算する
    emax_s = AnormI; //infty normが固有値の最大値の上界を与える
    emin_s = 1.0/AinvnormI; //逆行列の∞ノルムの逆数が固有値の最小値の上界を与える
    double condnum = AnormI*AinvnormI;
    c_s = 1/sqrt(emin_s * emax_s);
    printf("emins:%f emaxs%f c:%f cs:%f\n", emin_s, emax_s, c, c_s);

    /*
    flag変数でどのスケーリングをするか決める
    's': 近似固有値によるスケーリング 
    't': 真の固有値によるスケーリング
    その他: スケーリングなし
    */
    char flag = 's';
    if(flag == 's'){
        mat_multidouble(A, c_s, n*n); //ここでAに近似スケーリング値c_sを掛ける
        matcpy(A, Ainv, n);
        //スケーリング後の行列に対しても逆行列を計算する必要がある
        matrix_inv(Ainv, n);
    }
    else if(flag == 't'){
        mat_multidouble(A, c, n*n); 
        matcpy(A, Ainv, n);
        matrix_inv(Ainv, n);
    }
    
    //初期刻み幅hを決めて，そのhをループごとに小さくしてDE公式を適用していく
    double h = 0.5;
    loop = 5;
    for(int i = 0; i < loop; i++){
        double etime;
        timer_start();
        deformula(h, n, A, Ainv, N, ans, npm);
        etime = timer_end();
        double err = dif_norm(ans, def_ans, n);
        printf("%d %lf %.14lf %lf %d %lf %d %f\n", n, h, err, log10(err), npm[0]+npm[1]+1, etime, (int)(emax/emin), log10(condnum));
        
        h = h*0.5;
        /*
        if(err <= 1e-9){
            break;
        }*/
    }

    return 0;
}