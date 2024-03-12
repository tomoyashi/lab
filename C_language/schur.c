#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "mytimer.h"
#include "matrix1.h"
double eps = 1e-14;
double sum_time = 0.0, laptime = 0.0;

/*
void dgemm_(char *transA, char *transB, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldB, double *beta, double *C, int *ldc);

jobVS: if ='N' シューアベクトルは計算されない． else = 'V' 計算される



void dgees_(char *jobvs, char *sort, int *select, int *n, double *A, int *lda, int *sdim, double *wr, double *wi, double *vs, int *ldvs, double *work, int *lwork, bool *bwork, int *info);


static struct timespec time_start, time_end;

void timer_start(){
    clock_gettime(CLOCK_REALTIME, &time_start);
}

double timer_end(){
    clock_gettime(CLOCK_REALTIME, &time_end);
    return (time_end.tv_sec - time_start.tv_sec) + (time_end.tv_nsec - time_start.tv_nsec)/1e+9;
}
*/
double signs(double s){
    if(s > 0){
        return 1.0;
    }else if(s < 0){
        return -1.0;
    }
    else{
        //printf("error %lf\n", s);
        return 0;
    }
}

//行列の転置 アルゴリズムでは共役転置をしているが実行列を対象としているので純粋な転置を行う．
//今後の課題は共役転置
void transm(double *a, double *at){
    return;
}


//matrix signum function via schur method 
void msf_schur(double *a, double *q, int size, double *ans){
    //まずschur分解する
    timer_start();
    schur(a, q, size);
    laptime = timer_end(); sum_time+=laptime;
    printf("dgees: %f\n", laptime);
    timer_start();
    double u[size*size];
    //uを0で初期化
    #pragma omp parallel for
    for(int i=0; i < size*size; i++){
        u[i] = 0;
    }
    #pragma omp parallel for 
    for(int i=0; i < size; i++){
        //printf("%lf\n", a[i*size+i]);
        u[i*size + i] = signs(a[i*size+i]);
    }
    laptime = timer_end(); sum_time+=laptime;
    printf("u setting: %f\n", laptime);
    timer_start();
    int i, j, k;
    double tmp_;
    #pragma omp parallel for private(i, k, tmp_)
    for(j = 1; j < size; j++){
        for(i=j-1; i > 0; i--){
            //printf("i = %d j = %d\n", i, j);
            if(u[i*size + i] +u[j*size + j] != 0){
                
                tmp_ = 0.0;
                for(k = i+1; k <= j-1; k++){
                    tmp_ += u[i*size + k]*u[k*size + j];
                }
                u[i*size + j] = -tmp_/(u[i*size+i]+u[j*size+j]);
            }else{
                tmp_ = 0.0;
                for(k = i+1; k <= j-1; k++){
                    tmp_ += u[i*size + k]*a[k*size + j] - a[i*size + k]*u[k*size+j];
                }
                u[i*size + j] = a[i*size + j]*(u[i*size+i] - u[j*size + j])/(a[i*size + i] - a[j*size + j]) + tmp_/(a[i*size + i] - a[j*size + j]);
            }
        }
    }
    laptime = timer_end(); sum_time+=laptime;
    printf("u another: %f\n", laptime);
    timer_start();
    double tmp[size*size], qt[size*size];
    mat_T(q, qt, size);
    //print_mat(u, size, size);
    mat_multi(q, u, tmp, size);
    mat_multi(tmp, qt, ans, size);
    laptime = timer_end(); sum_time+=laptime;
    printf("matmulti*2: %f\n", laptime);
    //timer_start();
    return ;
}


int main(int argc, char *argv[]){
    timer_start();
    int size= 1000;
    double emin = 0.001, emax = 1000;

    //double cond = atoi(argv[1]);
    double a[size*size], q[size*size], ans[size*size], def_ans[size*size];
    laptime = timer_end(); sum_time+=laptime;
    printf("initialize: %f\n", laptime);
    timer_start();
    /* 
    double *a, *q, *ans, *aaa;
    a = (double *)malloc(sizeof(double)*size*size);
    q = (double *)malloc(sizeof(double)*size*size);
    ans = (double *)malloc(sizeof(double)*size*size);
    aaa = (double *)malloc(sizeof(double)*size*size);
    */
    identity(def_ans, size);
    laptime = timer_end(); sum_time += laptime; 
    printf("identity: %f\n", laptime);
    timer_start();
    //setmatrix_sym2(a, size, emax, emin, 3, aaa);
    setmatrix_sym(a, size, emax, emin, 1);
    laptime = timer_end(); sum_time += laptime; 
    printf("setmatrix_sym: %f\n", laptime);

    //setmatrix_ans(a, size, emax, emin, 3, def_ans, 'n');
    //setmatrix_X(a, size, emax, emin, 1, cond, 'Y'); 
    //print_mat(a, size, size);
    //print_mat(I, size, size);
    //clock_gettime(CLOCK_REALTIME, &ts);
    //printf("tv_sec= %ld, tv_nsec= %ld\n", ts.tv_sec, ts.tv_nsec);
    //for(int i = 0; i < 1; i++){
        msf_schur(a, q, size, ans);
    //}
    

    timer_start();
    //printf("time=%lf\n", time);
    //print_mat(ans, size, size);
    //double normf = dif_norm(ans, aaa, size);
    double dif = dif_norm(ans, def_ans, size);

    //double rel_err = calc_rerr_id(ans, size);
    //printf("size of matrix, minimum eigenvalue, max eigenvalue, log(err)\n");
    //printf 絶対誤差，log10(絶対誤差), 行列サイズ(行), 計算時間    
    printf("%.10f %f %d %0.8f %lf %lf \n",dif, log10(dif), size, sum_time, emin, emax);
    printf("%.16f %f %f\n", dif, (emax/emin), log10(emax/emin));
    laptime = timer_end(); sum_time += laptime; 
    printf("output time: %f\n", laptime);
    //printf("%d %lf %lf %lf %lf\n", size, emin, emax, log10(normf), log10(rel_err));
/*
    free(a);
    free(q);
    free(ans);
    free(aaa);
*/
    return 0;

}
