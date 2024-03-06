#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "matrix1.h"
#include "mytimer.h"
/*
特にスケーリングなどをしない，シンプルなニュートン法により，
sign(A)を計算．

reference:
The Matrix Sign Function Method and the Computation of Invariant Subspaces, 
Ralph Byers, Chunyang He, Volker Mehrmann

X_0 = A
と，初期化し繰り返し計算をおこなう

algorithm:
X_0 <- A
for k in 0 to max_iter
    X_(k+1) = (X_k + X_k^(-1))*0.5
    if norm(X_(k+1) - X_k, 1) <= c


*/


/*
*---------------------------------*
ニュートン法を行う関数
引数のmaxiterか，条件に合致するまでforループで反復計算を行う
最終的な結果がAに入っている

引数
      A: 入力行列
matsize: 行列の行もしくは列のサイズ(今回は正方行列なのでどちらでもよい)
maxiter: 繰り返しの最大回数

返り値:
繰り返し回数
    途中で打ち切ったら，そのときまでの繰り返し回数
    打ち切らなかったら，最大の繰り返し回数

*--------------------------------------*
*/
int newton_method(double *A, int matsize, int max_iter){
    int i, c = 10000*matsize, iter_num = max_iter;
    double eps = 1e-15;/*計算機イプシロン*/
    for(i=0; i<max_iter; i++){
        /*
            以下でループ内でしか使わない行列メモリの確保
            Ainv: Aの逆行列
            Aold: 計算前のAを保存しておくもの
        */
        double *Ainv = (double *)(malloc(sizeof(double)*matsize*matsize));
        double *Aold = (double *)(malloc(sizeof(double)*matsize*matsize));
        matcpy(A, Aold, matsize); /* AをAoldにコピー */
        matcpy(A, Ainv, matsize);/*AをAinvにコピーして次で逆行列計算*/
        matrix_inv(Ainv, matsize); /* Aの逆行列を計算 */
        mat_sum(A, Ainv, matsize*matsize, 1); /*AにAinvを足したものがAに入る*/
        mat_multidouble(A, 0.5, matsize*matsize); /*Aに0.5を掛ける*/
        
        //以下で収束判定
        double *Acopy = (double *)(malloc(sizeof(double)*matsize*matsize));
        mat_sum2(A, Aold, Acopy, matsize, -1); /*AcopyにA-Aoldの結果が入るようにする*/
        double Adif_norm1 = mat_norm(Acopy, matsize, '1'); /*norm1(A - Aold)*/
        double Anew_norm1_2 =  c*eps*pow(mat_norm(A, matsize, '1'), 2); /* c*eps*norm1(A)**2を計算 */
        //printf("%f %f\n", Adif_norm1, Anew_norm1_2);
        if(Adif_norm1 <= Anew_norm1_2){
            iter_num = i+1; /*繰り返し回数はそのときのループの回数となる*/
            free(Ainv);
            free(Aold);
            free(Acopy);
            break;
        }
        free(Ainv);
        free(Aold);
        free(Acopy);
    }

    return iter_num;
}


int main()
{
    double *A, emin = 0.1, emax = 10, *ANS, *Ainv, norm1A, norm1Ainv, condnum;
    int matsize = 1000;
    A = (double *)(malloc(sizeof(double)*matsize*matsize));
    ANS = (double *)(malloc(sizeof(double)*matsize*matsize));
    //setmatrix3(A, matsize, emax, emin, 1);
    //setmatrix_sym2(A, matsize, emax, emin, 1, ANS, 'p'); //最後の引数が'n'のとき固有値に負を含む
    Ainv = (double*)(malloc(sizeof(double)*matsize*matsize));
    setmatrix_ans(A, matsize, emax, emin, 1, ANS, 'n');
    //print_mat(A, matsize, matsize);
    matcpy(A, Ainv, matsize);
    matrix_inv(Ainv, matsize);
    norm1A = mat_norm(A, matsize, '1');
    norm1Ainv = mat_norm(Ainv, matsize, '1');
    condnum = norm1A * norm1Ainv;
    timer_start();
    int iter_num = newton_method(A, matsize, 20);
    double time_ = timer_end();
    //double rerr = calc_rerr_id(A, matsize);
    double err = dif_norm(A, ANS, matsize);

    printf("iter_num: %d error: %.8f time: %f err_log: %f log(cond): %f\n", iter_num, err, time_, log10(err), log10(condnum));


    free(A);

    return 0;
}
