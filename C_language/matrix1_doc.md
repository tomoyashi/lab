## matrix_inv(double *a, int size)
行列の逆行列を計算するための関数．実行後，第1引数の配列aがaの逆行列になっています．
sizeは行列の行か列の大きさです．ここでは，正方行列を想定しているのでこのようにしています．  
この関数ではLAPACKの``dgetrf``，``dgetri``を用いています．dgetrfはLU分解を行い，dgetriはdgetrfで得られた結果を用いて逆行列を計算するというようなサブルーチンとなっています．  
$A\mathbf{x}=\mathbf{b}$という方程式を計算するときに行列$A$をLU分解しますが，この$A$だけ同じで$\mathbf{b}$が違う場合は一回$A$をLU分解すればいいためdgetrfを一回計算すればよくなる．しかし，本実験でのプログラムでは毎回異なる行列の逆行列を計算するため，その強みは活かせません．

## matrix_inv3(double *a, int n, double *ans)
