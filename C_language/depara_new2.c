/*
新たに作り直す．
このプログラムでは，諸々の行列計算(行列積など)を行う関数をヘッダファイルに入れて呼び出せるようにする．


また，並列計算の見直しを行う．

できれば，行列をファイルから読み込んで計算できるようにしたい．

2023/10/13 ちゃんと内容を書いてなかったので，この日にプログラム内容を書く
このプログラムはDE公式でsign(A)を計算する
また，被積分関数が
f(A) = (t^2 A^-1 + A)^-1 
というように改善後のものである

また，逆行列計算をmatrix1.cのmatrix_inv3関数を使用 これはdgesvのAX=BのBを単位行列にすることで逆行列を計算
dgetrfとdgetriを用いて逆行列を計算するより速い

*/

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
maxn: 標本点数 (正負+0点)
nprocs: プロセス数
myrank: 自分のランク
part_ans: このランクでの部分和，後でmpi_reduceで全プロセスに足される
points: 並列化のため，自分が正か負のどっちの計算するかを知る

*/
void deformula(double h, int size, double *A, double *Ainv, int maxn, int nprocs, int myrank, double *part_ans, int points[]){    
    //double eps1 = eps*size*size;    
    int size2 = size*size, i;
    
    /*
     if(myrank == 0){
        func1(func2(0.0), A, Ainv, size, part_ans);
        mat_multidouble(part_ans, func3(0.0), size2);
    }
    */

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
    istart--; //rank0でのistartが0はじまりになるようにする
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


int main(int argc, char *argv[])
{
    int size = 1000, mat_size = size*size;
    double emin = 0.1, emax=10, A[mat_size], etime, Ainv[mat_size], def_ans[mat_size], tmp[mat_size], tmp2[mat_size], Atmp[mat_size];
    double AinvnormI, AnormI, condnum; //Ainvのinfノルム，Aのinfノルム，Aの条件数
    //double I[SIZE*SIZE];    setmatrix2(A, I, SIZE, SIZE, emax, emin, 4);
    //int cond = atoi(argv[1]);
    
    //setmatrix_X(A, SIZE, emax, emin, 1, cond, 'Y');
    setmatrix_ans(A, size, emax, emin, 1, def_ans, 'n');

    double h = 0.1;    
    int nmax = 80;//こちらの場合はnmaxは正負の分点数の合計
    int points[nmax+1];
    //1回のループで4個分やるようにしてる．問題があったら見直す
    for(int i = 1; i < nmax+1; i+=4){
        points[i] = i/2+1;
        points[i+1] = -(i/2+1);
        points[i+2] = i/2+2;
        points[i+3] = -(i/2+2);
    }
    points[0] = 0;
    MPI_Init(&argc, &argv);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    //MPI_Barrier(MPI_COMM_WORLD);
    timer_start();
    //int npm[2]={0,0}, npmsum[2]={0,0};
    if(myrank == 0){
        //条件数の計算やA^-1の計算を行う
        AnormI = mat_norm(A, size, 'i');
        
        matcpy(A, Atmp, size);
        //ここでのA^-1の計算もmatrix_inv3を使うように変更した
        identity(Ainv, size);
        //AtmpにAのコピー，Ainvに単位行列を入れると，計算後に逆行列が入る
        matrix_inv3(Atmp, size, Ainv);

        AinvnormI = mat_norm(Ainv, size, 'i');
        condnum = AnormI*AinvnormI;

    }
    
    //deformulaでtmpに各プロセスでの部分和が入る-> mpi_reduceでtmp2にその合計を入れる
    //tmpは0で初期化する必要がある．ここでは静的に確保してるので自動で0初期化される
    MPI_Bcast(&Ainv[0], mat_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    deformula(h, size, A, Ainv, nmax, nprocs, myrank, tmp, points);
    MPI_Reduce(tmp, tmp2, mat_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(myrank == 0){
        mat_multidouble(tmp2, h/M_PI*2.0, mat_size);
        etime = timer_end();
        //double rerr = calc_rerr_id(tmp2, SIZE);
        //double rerr = calc_err_id(tmp2, SIZE);
        double err = dif_norm(tmp2, def_ans, size);
        //print_mat(tmp2, SIZE, SIZE);
        //printf h:刻み幅, 分点数, 絶対誤差, log10(絶対誤差), 行列サイズ(行), プロセス数，計算時間
        printf("h:刻み幅 分点数 絶対誤差 log10(絶対誤差) 行列サイズ(行) 条件数 log(条件数) プロセス数 計算時間\n");
        printf("%f %d %.15f %f %d %f %f %d %0.8f\n", h, nmax+1, err, log10(err), size, condnum, log10(condnum), nprocs, etime);
        //printf("%f %d %.13f %f %d %d %0.8f %d\n", h, nmax*2+1, rerr, log10(rerr), SIZE, nprocs, etime, cond);
    }


    MPI_Finalize();
    //fclose(fp);
    return 0;
}
