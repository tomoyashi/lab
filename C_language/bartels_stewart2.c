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
AX - XB = Cを解く
lapackのdtrsylルーチンを用いる
このルーチンはBartels-Stewartアルゴリズムが基になっている
A, BをSchur分解し，上三角行列にする．その後，dtrsylを使用

netlibのページ
dtrsyl: https://netlib.org/lapack/explore-html-3.6.1/d3/db6/group__double_s_ycomputational_gac63d16310c88b690ac17430e70ac4cb5.html


*/

void dtrsyl_(char *trana, char *tranb, int *isgn, int *m, int *n, double *a, int *lda, double *b, int *ldb, double *c, int *ldc, double *scale, int *info);

int main(){
    int size = 1000;
    double A[size*size], B[size*size], U[size*size], V[size*size], R[size*size], S[size*size], Y[size*size], F[size*size], C[size*size];
    setmatrix3(A, size, 100, 0.1, 3);
    setmatrix3(B, size, 11, 0.2, 2);
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
    printf("schur*2time: %f\n", etime1);
    double Ut[size*size], tmp[size*size], Vt[size*size], ans[size*size], XB[size*size];
    mat_T(U, Ut, size);
    mat_T(V, Vt, size);
    mat_multi(Ut, C, tmp, size);
    mat_multi(tmp, V, F, size); //F= U^t C V を作成
    //print_mat(F, size, size);

    char trana='N', tranb = 'T';
    int isgn = -1, info;
    double scale;
    double etime2 = timer_end();
    dtrsyl_(&trana, &tranb, &isgn, &size, &size, R, &size, S, &size, F, &size, &scale, &info);
    printf("dtrsyl+etc: %f dtrsyl: %f\n", timer_end() - etime1, timer_end() - etime2);
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

    //print_mat(tmp, size, size);
}