

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

//#define SIZE 1000
double eps = 1e-15;

///*
void func1(double x, double *A, double *Ainv, int n, double *ans){
    int i;
    double tmp[n*n];
    for(i = 0; i < n*n; i++){
        tmp[i] = x*x*Ainv[i] + A[i];
        //ans[i] = 0.0;
    }
    
    identity(ans, n);
    //matrix_inv(ans, n);
    matrix_inv3(tmp, n, ans);
    return;
}//*/


double func2(double x){
    return exp(M_PI*0.5*sinh(x));
}

double func3(double x){
    return M_PI*0.5*exp(M_PI*0.5*sinh(x)) * cosh(x);
}

/*
h: 刻み幅
size: A \in n*nR 行列サイズの列か行のサイズ
A: 入力行列
Ainv: Aの逆行列
maxn: 最大反復回数
nprocs: プロセス数
myrank: 自分のランク
part_ans: このランクでの部分和，後でmpi_reduceで全プロセスに足される
points: 並列化のため，自分が正か負のどっちの計算するかを知る

*/
void deformula(double h, int size, double *A, double *Ainv, int maxn, int nprocs, int myrank, double *part_ans, int points[]){    
    //double eps1 = eps*size*size;    
    int size2 = size*size, i;
    
    //プロセス0のときのみx=0のときの値をtmpに入るようにする．
    if(myrank == 0){
        func1(func2(0.0), A, Ainv, size, part_ans);
        mat_multidouble(part_ans, func3(0.0), size2);
    }
    int n_loop, istart;
    if(maxn%nprocs == 0){
        n_loop = maxn/nprocs;
        istart = myrank*n_loop + 1;
    }else{
        n_loop = maxn/nprocs + 1;
        istart = myrank*n_loop + 1;
        int marge = maxn-(maxn/nprocs)*nprocs;
        if(myrank == marge){
            //istart = myrank*n_loop + 1;
            n_loop = maxn/nprocs;
        }else if(myrank >= marge+1){
            istart = myrank*(maxn/nprocs) + marge + 1;
            n_loop = maxn/nprocs;
        }
    }
    //
    for(i = istart; (i < istart + n_loop) && (i <= maxn) ; i++){
        double x = points[i]*h;
        double tmp_part[size2];
        func1(func2(x), A, Ainv, size, tmp_part);
        mat_multidouble(tmp_part, func3(x), size2);
        mat_sum(part_ans, tmp_part, size2, 1);
    }
    //printf("\n");
}

/*
    H = [A, -C; 0,-B]を作成
    A : m*m
    B: n*n
    C: m*n
*/

void make_H(double *A, int m, double *B, int n, double *C, double *H){
    int i, j;

    //AをHに入れる
    for(i=0; i<m; i++){
        for(j = 0; j<m; j++){
            H[i*(m+n) + j] = A[i*m + j];
        }
    }
    //-CをHに入れる
    for(i = m; i < m + n; i++){
        for(j = 0; j < m; j++){
            H[i*(m+n) + j] = -C[(i-m)*m + j];
        }
    }
    //-BをHに入れる
    for(i = m; i < m + n; i++){
        for(j = m; j < m + n; j++){
            H[(i)*(m+n) + j] = -B[(i-m)*m + j - m];
        }
    }
}


int main(int argc, char *argv[])
{
    int m = 1000, n=m, Hsize = (m+n)*(m+n);
    double emin = 0.1, emax=10, A[m*m], Hinv[Hsize], tmp[Hsize], B[n*n], C[m*n], H[Hsize], deans[Hsize], X[m*n], etime, tmp2[Hsize];
    double HinvnormI, HnormI, condnum; //Ainvのinfノルム，Aのinfノルム，Aの条件数
    //double I[SIZE*SIZE];    setmatrix2(A, I, SIZE, SIZE, emax, emin, 4);
    //int cond = atoi(argv[1]);
    
    //setmatrix_X(A, SIZE, emax, emin, 1, cond, 'Y');
    //setmatrix_ans(A, size, emax, emin, 1, def_ans, 'n');

    setmatrix_sym(A, m, 10, 0.2, 1);
    setmatrix_sym(B, n, 12, 2, 2);

    
    identity(C, n); //本当はm*nなのでしっかりやらなきゃだが，ここではm=nなのでご愛嬌
    make_H(A, m, B, n, C, H);

    double h = 0.08;    
    int nmax = 120;//こちらの場合はnmaxは正負の分点数の合計
    int points[nmax+1];
    //1回のループで4個分やるようにしてる．問題があったら見直す
    for(int i = 1; i < nmax+1; i+=4){
        points[i] = i/2+1;
        points[i+1] = -(i/2+1);
        points[i+2] = i/2+2;
        points[i+3] = -(i/2+2);
    }
    MPI_Init(&argc, &argv);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    //MPI_Barrier(MPI_COMM_WORLD);
    timer_start();
    //int npm[2]={0,0}, npmsum[2]={0,0};
    if(myrank == 0){
        //条件数の計算やA^-1の計算を行う
        HnormI = mat_norm(H, m+n, 'i');
        
        matcpy(H, tmp, m+n);
        //ここでのH^-1の計算もmatrix_inv3を使うように変更した
        identity(Hinv, m+n);
        //tmpにHのコピー，Hinvに単位行列を入れると，計算後に逆行列が入る
        matrix_inv3(tmp, m+n, Hinv);

        HinvnormI = mat_norm(Hinv, m+n, 'i');
        condnum = HnormI*HinvnormI;

        //printf("Hinv\n");
        //print_mat(Hinv, m+n, n+m);

    }
    
    MPI_Bcast(&Hinv[0], Hsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    deformula(h, m+n, H, Hinv, nmax, nprocs, myrank, deans, points);
    //printf("de\n");
    //print_mat(deans, m+n, m+n);
    MPI_Reduce(deans, tmp, Hsize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(myrank == 0){
        mat_multidouble(tmp, h/M_PI*2.0, Hsize);
        etime = timer_end();

        //Hの右上ブロックが-2Xなので，それを取得して，-0.5を掛ける
        for(int i = m; i < m+n; i++){
            for(int j = 0; j < m; j++){
                X[(i-m)*m + j] = -0.5*tmp[(i)*(m+n) + j];
            }
        }
        //AX - XBを計算
        double XB[n*m], tmp[m*n]; //tmpの要素数をここで変更

        mat_multi(A, X, tmp, m);
        mat_multi(X, B, XB, n);

        mat_sum(tmp, XB, n*n, 1);

        double err = dif_norm(tmp, C, n);
        //print_mat(X, m, n);
        //printf h:刻み幅, 分点数, 絶対誤差, log10(絶対誤差), 行列サイズ(行), プロセス数，計算時間
        printf("h:刻み幅 分点数 絶対誤差 log10(絶対誤差) 行列サイズ(行) 条件数 log(条件数) プロセス数 計算時間\n");
        printf("%f %d %.15f %f %d %f %f %d %0.8f\n", h, nmax+1, err, log10(err), n+m, condnum, log10(condnum), nprocs, etime);
        //printf("%f %d %.13f %f %d %d %0.8f %d\n", h, nmax*2+1, rerr, log10(rerr), SIZE, nprocs, etime, cond);
    }


    MPI_Finalize();
    //fclose(fp);
    return 0;
}


