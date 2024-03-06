#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <stdbool.h>
#include "matrix1.h"
#include "mytimer.h"

/*
AX - BX = Cを解く
lapackのdtrsylルーチンを用いる
このルーチンはBartels-Stewartアルゴリズムが基になっている
また，The Hessenberg-Schurアルゴリズムの方を用いていると思われる
dtrsylのA, Bはnetlibのページにも書いてある通り，DHSEQRでシューア標準形にする

netlibのページ
dtrsyl: https://netlib.org/lapack/explore-html-3.6.1/d3/db6/group__double_s_ycomputational_gac63d16310c88b690ac17430e70ac4cb5.html

dhseqr: https://netlib.org/lapack/explore-html/d9/dc6/group__hseqr_ga62c3f96d2f67f96d6dc10334e118e451.html#ga62c3f96d2f67f96d6dc10334e118e451

sizeを変えてテスト

富岳コンパイル
mpifccpx -Kfast,SVE,openmp matrix1.c syl_bar_test.c mytimer.c  -o syl_bar1 -SSL2 -lm

*/


int main(){
    int size = 100, rec = 10;
    for(int i = 0; i < rec; i++){
        double A[size*size], B[size*size], U[size*size], V[size*size], R[size*size], S[size*size], Y[size*size], F[size*size], C[size*size];
        setmatrix_sym(A, size, 11, 0.1, 1);
        setmatrix_sym(B, size, 12, 0.2, 2);
        timer_start();
        matcpy(A, R, size);
        matcpy(B, S, size);

        identity(C, size);
        mat_T(B, S, size);
        //print_mat(A, size, size);
        //timer_start();
        schur(R, U, size); //Aをschur分解，schur標準形がR, schurベクトル行列がU
        schur(S, V, size); //B^Tをschur分解，schur標準形がV, schurベクトル行列がV
        double etime1 = timer_end();
        //printf("schur*2time: %f\n", etime1);
        double Ut[size*size], tmp[size*size], Vt[size*size], ans[size*size], XB[size*size];
        mat_T(U, Ut, size);
        mat_T(V, Vt, size);
        mat_multi(Ut, C, tmp, size);
        mat_multi(tmp, V, F, size); //F= U^t C V を作成
        //print_mat(F, size, size);

        char trana='N', tranb = 'T';
        int isgn = -1, info;
        double scale;
        //double etime2 = timer_end();
        dtrsyl_(&trana, &tranb, &isgn, &size, &size, R, &size, S, &size, F, &size, &scale, &info);
        //printf("dtrsyl+etc: %f dtrsyl: %f\n", timer_end() - etime1, timer_end() - etime2);
        //printf("scale %f\n", scale);
        //最後にX = U Y V^T を計算
        mat_multi(U, F, tmp, size);
        mat_multi(tmp, Vt, ans, size);
        //print_mat(ans, size, size);
        double etime = timer_end();

        mat_multi(A, ans, tmp, size);
        mat_multi(ans, B, XB, size);
        
        mat_sum(tmp, XB, size*size, -1);
        double error = calc_err_id(tmp, size);
        printf("n: %d error: %f log(error): %f time(s): %f\n", size, error, log10(error), etime);
        size+=100;
    }

    //print_mat(tmp, size, size);
}