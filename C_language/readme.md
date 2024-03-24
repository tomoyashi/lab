C言語のプログラムがあります．  
以下でプログラムの説明を行っていきます．

# プログラムファイルの説明
## matrix1.*
### matrix1ファイルの使い方
matrix1.h, matrix1.cというファイルがあるがこれらは行列計算用の関数をまとめたものです．このファイルで例えば逆行列計算や行列の和などといった計算をLAPACKのサブルーチンを呼び出すことで計算できるようにしています．LAPACKの関数を用いているため，多くの最適化が行われていて計算が速くなります．  
このプログラムを用いる場合はmainのファイル内では次のようにインクルードすることで，mainファイル内でmatrix1.cの関数を使えるようなります．  
```c
#include "matrix1.h"
```
また，コンパイル時は次のようにします．
```
gcc main.c matrix1.c -llapack -lblas -lm 
```
毎回このようにコンパイルしてもいいですが，面倒な場合はmakefileを作るのもいいかもしれません．

### matrix1内の関数についての説明
基本的には，matrix1.c内で各関数の前にコメントアウトで内容が記載されています．

以下によく使うファイルのみ詳細を記述します．  
[matrix1.*の内容](matrix1_doc.md)

## mytimer.*
mytimer.cおよびmytimer.hは時間計測用のプログラムです．mpiを用いるときは``MPI_Wtime``関数でも時間を計測できますが、mpiを使わないときなどすべてで同じ形式で時間を計測したかったため作成しました．  
使い方はmatrix1.*の場合と同様にまずmainのファイルで次のようにインクルードします．
```c
#include "mytimer.h"
```
ここで、``func1(n)``という関数の時間を計測したい場合は次のようにします．
```c
timer_start();
func1(n);
double func1_time = timer_end();
printf("%f\n", func1_time);
```
上記のようにtimer_end()を呼び出した時点でtimer_start()からの時間をdouble型で返します．  
複数区間を測定したい場合は次のようにします．
```c
timer_start();
func1(n);
double time1 = timer_end();
func2(n);
double time2 = timer_end();
printf("func1(n)time %.4f func2(n)time %.4f totaltime %.4f\n", time1, time2-time1, time2);
```
上記の複数区間の測定に関して実際の結果をコンパイルから示すと次のようになります．(プログラムは``timer_test.c``参照)
```
$ gcc mytimer.c timer_test.c  -lm 
$ ./a.out
func1(n)time 0.00017570 func2(n)time 2.59215808 totaltime 2.59233378
```
この例の場合はfunc2が一番時間がかかっているということがわかります．注意点として上記の場合はtime2を計測するときに差を計算するので、そこで誤差が入ってしまう可能性があります．もし厳密に時間を測定したい場合は区間ごとにtimer_start()で計測しなおしてください．また、時間測定は``clock_gettime``関数により行っています． 

## depara_new.c
これは行列符号関数をDE公式で計算するプログラムをMPIにより並列化したものです．  
DE公式の並列化部分については修士論文内のアルゴリズム2のようにしています．  
行列は正方行列のみを想定していて、そのサイズはmain関数内のsize変数で決めています．

