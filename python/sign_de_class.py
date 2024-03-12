
import numpy as np
import matrix_cls
import matplotlib.pyplot as plt
import numpy.linalg as LA
import seaborn as sns
import scipy as sp

"""
DE公式用の関数またはクラスを新たに考えていきたい
どういったもの？
今までの計算法では各刻み幅hに対して収束するまで計算を行っていた．
これでは，hを毎回変えて調べる必要があり大変であり，どのときに最小の誤差になったかが
自動で調べられない．もちろん，今までの方がいい点もあるが，今知りたいのは「ある行列に対してDE公式を適用したときの最小の誤差」である．そのため，分点数や刻み幅など毎の解析は必要ない．

追記
上記の要件を満たす関数は，min_dif関数であり，あるhに対して収束まで反復計算を繰り返したい場合は，
インスタンスを作成してde_formula関数を実行すればいいだけ．

"""
#被積分関数を変えられるようにしておく．
#積分区間は半無限区間のものしか考えないので，それのみ．増やす場合は，書き換える

class DEformula_calculator(object):
    """
    """
    def __init__(self, A,):
        #入力行列Aは他で作ったのを使う(matrix_cls)
        self.A = A
        #入力サイズ Aがnumpyのndarrayである前提で作成してるので，そうでない場合は注意
        self.size = A.shape[0]

        #A^2を何回も使うかも知れないので，先に作っておく
        self.A2 = self.A @ self.A

        self.I = np.eye(self.size)

        #ここに最も良い解が入るようにする．min_dif関数を実行後でないとゼロ行列が返される
        self.ans = np.zeros((self.size, self.size))
        #以上の解のときの正と負の反復回数は以下に
        self.Npm = [0, 0]
        #そのときの刻み幅
        self.hans = 0
        
        #何回も利用するかもしれないので，先に計算しておく
        self.Ainv = LA.inv(self.A)
        
        #反復回数の最大値をインスタンス変数として初期化
        self.maxn = 1000
    
    #A^2をXΛ^2X^-1によって計算. 一般にはこのように計算できないが，A^2をA@Aと計算するより丸め誤差が少なくなると予想される(実験用関数)
    def set_A2(self, Lambda, X, Xinv):
        L2 = Lambda@Lambda 
        self.A2 = X@L2@Xinv 
        return self.A2
    
    #定義のままの被積分関数
    def integrand1(self, x, ):
        return LA.inv(x**2*self.I + self.A2)
    
    #A^-1を被積分関数に掛けた場合
    def integrand2(self, x, ):
        return LA.inv(x**2*self.Ainv + self.A)
    
   #被積分関数を複素数に拡張した場合 
    def integrand3(self, x, ):
        Icomp = self.I.astype(np.complex128) #単位行列を複素数型に変更
        positive = LA.inv(self.A - (x*1j)*self.I) 
        #(A-itI)^-1 と (A + itI)^-1は複素共役なので片方計算すればいい
        return positive + np.conjugate(positive)
    
    #変換関数(半無限区間)
    def transform(self, x):
        return np.exp(np.pi*0.5*np.sinh(x))
    
    #変換関数を微分したもの
    def transform_dif(self, x):
        return np.pi*0.5*np.exp(np.pi*0.5*np.sinh(x))*np.cosh(x)

    
    def de_formula(self, h, func, stop=True):
        """
        定義のままの被積分関数に対するDE公式

        parameters
        -----------
        h: float
            刻みh
        func: function
            被積分関数を返す関数を渡す
            ここでは，例えばself.integrand1を渡す

        returns:
        -----------
        ans: array_like
            solution of DE formula for sign(A)
        npm: list
            npm[0]: int
            the number of point in positive
            npm[1]: int
            the number of point in negative
        stop: boolian
            True: 部分和が小さくなったら打ち切る
            False: 打ち切らずに，決められた最大回数まで足し合わせる *基本的にはTrueにして使う
        """
        #sは部分和の合計，これに足し合わせていく
        s = func(self.transform(0.0))*self.transform_dif(0.0)
        maxn = self.maxn 
        eps = 1e-15
        npm = [maxn, maxn]
        for i in range(1, maxn+1):
            x = i*h 
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps and stop:
                npm[0] = i 
                break
        for i in range(1, maxn+1):
            x = -i*h 
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps and stop:
                npm[1] = i 
                break
        ans = self.A@s*2/np.pi*h
        return ans, npm
    
    def de_formula_half(self, h, func):
        """
        一般形の行列符号関数の積分表示形にDE公式を適用する
        
        この関数は，1回前のあるhの計算のときに収束するまで計算した値sign(A)
        が，許容誤差を満たさない場合に，その解を用いて更に小さい刻み幅で計算する
        ための関数．
        詳しくは「FORTRAN77数値計算プログラミング」のDE公式の章に書いてある．
        簡単に言うと，あるhの刻み幅に対してDE公式を計算して誤差が許容誤差に満たさなかった場合に，またhを1/2にして計算をするがこのとき，
        前のループでの計算が半分を占める．そのため，前のループで計算した値と，前のループで計算してない標本点部分の値を足し合わせる．
        こうすることで，hを小さくしたときの計算量が1/2に減る．

        この関数はmin_dif関数内で使われる
        
        parameters
        -----------
        h: float
            刻みh
        func: function
            被積分関数を返す関数を渡す
            ここでは，例えばself.integrand2 を渡す

        returns:
        -----------
        ans: array_like
            solution of DE formula for sign(A)
        npm: list
            npm[0]: int
            the number of point in positive
            npm[1]: int
            the number of point in negative
        """
        #sは部分和の合計，これに足し合わせていく
        #s = func(self.transform(0.0))*self.transform_dif(0.0)
        maxn = self.maxn
        eps = 1e-15
        npm = [0, 0]
        s = np.zeros((self.size, self.size))
        for i in range(0, maxn+1):
            x = (2*i+1)*h*0.5
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps:
                npm[0] = i 
                break
        for i in range(0, maxn+1):
            x = -(2*i+1)*h*0.5
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps:
                npm[1] = i 
                break
        ans = self.A@s*2/np.pi*h
        return ans, npm
    def de_formula2(self, h, func, stop=True):
        """
        被積分関数を変形した後にDE公式を適用する
        
        parameters
        -----------
        h: float
            刻みh
        func: function
            被積分関数を返す関数を渡す
            ここでは，例えばself.integrand2 を渡す

        returns:
        -----------
        ans: array_like
            solution of DE formula for sign(A)
        npm: list
            npm[0]: int
            the number of point in positive
            npm[1]: int
            the number of point in negative
        stop: boolian
            True: 部分和が小さくなったら打ち切る
            False: 打ち切らずに，決められた最大回数まで足し合わせる *基本的にはTrueにして使う
        """
        #sは部分和の合計，これに足し合わせていく
        s = func(self.transform(0.0))*self.transform_dif(0.0)
        maxn = self.maxn
        eps = 1e-15
        npm = [maxn, maxn]
        for i in range(1, maxn+1):
            x = i*h 
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps and stop:
                npm[0] = i 
                break
        for i in range(1, maxn+1):
            x = -i*h 
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps and stop:
                npm[1] = i 
                break
        ans = s*2/np.pi*h
        return ans, npm
    def de_formula2_half(self, h, func, stop=True):
        """
        被積分関数を変形した後にDE公式を適用する
        
        この関数は，1回前のあるhの計算のときに収束するまで計算した値sign(A)
        が，許容誤差を満たさない場合に，その解を用いて更に小さい刻み幅で計算する
        ための関数．
        
        parameters
        -----------
        h: float
            刻みh
        func: function
            被積分関数を返す関数を渡す
            ここでは，例えばself.integrand2 を渡す

        returns:
        -----------
        ans: array_like
            solution of DE formula for sign(A)
        npm: list
            npm[0]: int
            the number of point in positive
            npm[1]: int
            the number of point in negative
        stop: boolian
            True: 部分和が小さくなったら打ち切る
            False: 打ち切らずに，決められた最大回数まで足し合わせる *基本的にはTrueにして使う
        """
        #sは部分和の合計，これに足し合わせていく
        #s = func(self.transform(0.0))*self.transform_dif(0.0)
        maxn = self.maxn
        eps = 1e-15
        npm = [maxn ,maxn]
        s = np.zeros((self.size, self.size))
        for i in range(0, maxn+1):
            x = (2*i+1)*h*0.5
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps and stop:
                npm[0] = i 
                break
        for i in range(0, maxn+1):
            x = -(2*i+1)*h*0.5
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps and stop:
                npm[1] = i 
                break
        ans = s*2/np.pi*h
        return ans, npm
    def de_formula_complex(self, h, func):
        """
        複素数に拡張した行列符号関数の積分表示形式にDE公式を適用
        
        parameters
        -----------
        h: float
            刻みh
        func: function
            被積分関数を返す関数を渡す
            ここでは，例えばself.integrand3を渡す

        returns:
        -----------
        ans: array_like
            solution of DE formula for sign(A)
        npm: list
            npm[0]: int
            the number of point in positive
            npm[1]: int
            the number of point in negative
        """
        #sは部分和の合計，これに足し合わせていく
        s = func(self.transform(0.0))*self.transform_dif(0.0)
        maxn = self.maxn
        eps = 1e-15
        npm = [0, 0]
        for i in range(1, maxn+1):
            x = i*h 
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps:
                npm[0] = i 
                break
        for i in range(1, maxn+1):
            x = -i*h 
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps:
                npm[1] = i 
                break
        ans = s*h/np.pi
        return ans, npm
    def de_formula_complex_half(self, h, func):
        """
        複素数に拡張した行列符号関数の積分表示形式にDE公式を適用
        
        この関数は，1回前のあるhの計算のときに収束するまで計算した値sign(A)
        が，許容誤差を満たさない場合に，その解を用いて更に小さい刻み幅で計算する
        ための関数．
        
        parameters
        -----------
        h: float
            刻みh
        func: function
            被積分関数を返す関数を渡す
            ここでは，例えばself.integrand2 を渡す

        returns:
        -----------
        ans: array_like
            solution of DE formula for sign(A)
        npm: list
            npm[0]: int
            the number of point in positive
            npm[1]: int
            the number of point in negative
        """
        #sは部分和の合計，これに足し合わせていく
        #s = func(self.transform(0.0))*self.transform_dif(0.0)
        maxn = self.maxn
        eps = 1e-15
        npm = [0, 0]
        s = np.zeros((self.size, self.size), dtype=np.complex128)
        for i in range(0, maxn+1):
            x = (2*i+1)*h*0.5
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps:
                npm[0] = i 
                break
        for i in range(0, maxn+1):
            x = -(2*i+1)*h*0.5
            t = func(self.transform(x))*self.transform_dif(x)
            s += t
            #tmpでtのフロベニウスノルムを計算し，それがeps以下になったら反復を終了する
            tmp = LA.norm(t, 'fro')
            if tmp < eps:
                npm[1] = i 
                break
        ans = s*h/np.pi
        return ans, npm
    """
    def min_dif(self, step_num, def_ans, method = "1"):
        
        #この関数で，刻み幅hを変えて収束するまで計算を行い，最小の誤差を返す．
        #そのときの，解についてはself.ansに保存しておく．
        #解をdef_ansに渡して，その差を誤差とする

        #methodにはDE公式を使うか
        
        h = 0.5
        min = float("inf") #無限大にして最初の誤差は必ずこれより小さくなるようにする
        if method == "1":
            for _ in range(step_num):
                ans, npm = self.de_formula(h, self.integrand1)
                dif = LA.norm(ans-def_ans, "fro")
                if dif < min:
                    min = dif
                    self.ans = ans
                    self.Npm = npm
                    self.hans = h 
                h*= 0.5
        elif method=="2":
            for _ in range(step_num):
                ans, npm = self.de_formula2(h, self.integrand2)
                dif = LA.norm(ans-def_ans, "fro")
                if dif < min:
                    min = dif
                    self.ans = ans
                    self.Npm = npm
                    self.hans = h 
                h*= 0.5
        elif method=="comp":
            for _ in range(step_num):
                ans, npm = self.de_formula_complex(h, self.integrand3)
                dif = LA.norm(ans - def_ans, "fro")
                if dif < min:
                    min = dif
                    self.ans = ans
                    self.Npm = npm
                    self.hans = h 
                h*=0.5 
        return min
    """
    def min_dif(self, step_num, def_ans, method = "1", h = 0.125):
        """
        この関数で，刻み幅hを変えて収束するまで計算を行い，最小の誤差を返す．
        そのときの，解についてはself.ansに保存しておく．
        解をdef_ansに渡して，その差を誤差とする

        methodにはDE公式を使うか

        hを変えながら計算していくが，前回の誤差と今回の誤差で大きな差がなくなったら終了するように変更
        また，そのときの標本点数も保持するようにする(2024/1/12変更)
        """
        #h=0.125 #引数に追加するようにした
        min = float("inf") #無限大にして最初の誤差は必ずこれより小さくなるようにする
        if method == "1":
            ans, npm = self.de_formula(h, self.integrand1)
            dif = LA.norm(ans-def_ans, "fro")
            #ここif文いらない？
            if dif < min:
                min = dif
                self.ans = ans
                self.Npm = npm
                self.hans = h 
            for _ in range(step_num-1):
                t, npm2 = self.de_formula_half(h, self.integrand1)
                ans = 0.5*(ans + t)
                new_dif = LA.norm(ans-def_ans, "fro")
                #print(f"log(new)-log(old): {np.log10(new_dif) - np.log10(dif)} newerror: {new_dif}")
                if abs(np.log10(new_dif/dif)) < 1:
                    #log(new)-log(old) = log(new/old)
                    #今回の誤差の対数と前回の誤差の対数の差の絶対値が1より小さいなら収束してるので，前回のループの解でよくて，ansやh,Npmの更新をせずにループ終了
                    break 
                #そうでない場合はdifをnew_difで更新，標本点数も更新
                npm[0] = npm[0] + npm2[0]
                npm[1] = npm[1] + npm2[1]
                dif = new_dif
                #print(f"error: {dif} logerror: {np.log10(dif)} np: {npm[0]}, nm: {npm[1]} h: {h}")
                #difが今までの最小誤差より小さいか判定
                if dif < min:
                    min = dif
                    self.ans = ans
                    self.Npm = npm
                    self.hans = h 
                h*= 0.5
        elif method=="2":
            ans, npm = self.de_formula2(h, self.integrand2)
            dif = LA.norm(ans-def_ans, "fro")
            if dif < min:
                min = dif
                self.ans = ans
                self.Npm = npm
                self.hans = h 
            for _ in range(step_num-1):
                t, npm2 = self.de_formula2_half(h, self.integrand2)
                ans = 0.5*(ans + t)
                new_dif=LA.norm(ans-def_ans, "fro")
                if abs(np.log10(new_dif/dif)) < 1:
                    #log(new)-log(old) = log(new/old)
                    #今回の誤差の対数と前回の誤差の対数の差の絶対値が1より小さいなら収束してるので，前回のループの解でよくて，ansやh,Npmの更新をせずにループ終了
                    break 
                #そうでない場合はdifをnew_difで更新，標本点数も更新
                npm[0] = npm[0] + npm2[0]
                npm[1] = npm[1] + npm2[1]
                dif = new_dif
                if dif < min:
                    min = dif
                    self.ans = ans
                    self.Npm = npm
                    self.hans = h 
                h*= 0.5
        elif method=="comp":
            ans, npm = self.de_formula_complex(h, self.integrand3)
            dif = LA.norm(ans-def_ans, "fro")
            if dif < min:
                min = dif
                self.ans = ans
                self.Npm = npm
                self.hans = h 
            for _ in range(step_num-1):
                t, npm2 = self.de_formula_complex_half(h, self.integrand3)
                ans = 0.5*(ans + t)
                new_dif=LA.norm(ans-def_ans, "fro")
                if abs(np.log10(new_dif/dif)) < 1:
                    #log(new)-log(old) = log(new/old)
                    #今回の誤差の対数と前回の誤差の対数の差の絶対値が1より小さいなら収束してるので，前回のループの解でよくて，ansやh,Npmの更新をせずにループ終了
                    break 
                #そうでない場合はdifをnew_difで更新，標本点数も更新
                npm[0] = npm[0] + npm2[0]
                npm[1] = npm[1] + npm2[1]
                dif = new_dif
                if dif < min:
                    min = dif
                    self.ans = ans
                    self.Npm = npm
                    self.hans = h 
                h*=0.5 
        return min
    def solve_step(self, step_num, def_ans, method = "1", h = 0.125, h_delta = 0.5):
        """
        この関数で，刻み幅hについて収束するまで標本点を増やして計算する
        hをstep_n回変えていき，上記のことを行う

        刻み幅h, 標本点数(正，負，合計)のタプル，誤差
        をstep_num個分もった配列を返す

        h: 初期刻み幅
        h_delta: hを減らしてくときにどれだけの倍率で減らすか．
        """
        #h=0.125 引数に追加するようにした
        #ret_values: リスト. これに要素を追加していく
        ret_values = []
        if method == "1":
            ans, npm = self.de_formula(h, self.integrand1)
            dif = LA.norm(ans-def_ans, "fro")
            ret_values.append([h, (npm[0], npm[1], npm[0]+npm[1]+1), dif])
            #npmを足し合わせていきたいので最初はループ外
            for _ in range(step_num-1):
                h*= h_delta
                ans, npm = self.de_formula(h, self.integrand1)
                dif = LA.norm(ans-def_ans, "fro")
                ret_values.append([h, (npm[0], npm[1], npm[0]+npm[1]+1), dif])
                
        elif method == "2":
            ans, npm = self.de_formula2(h, self.integrand2)
            dif = LA.norm(ans-def_ans, "fro")
            ret_values.append([h, (npm[0], npm[1], npm[0]+npm[1]+1), dif])
            #npmを足し合わせていきたいので最初はループ外
            for _ in range(step_num-1):
                h*= h_delta
                ans, npm = self.de_formula2(h, self.integrand2)
                dif = LA.norm(ans-def_ans, "fro")
                ret_values.append([h, (npm[0], npm[1], npm[0]+npm[1]+1), dif])
        if method == "comp":
            ans, npm = self.de_formula_complex(h, self.integrand3)
            dif = LA.norm(ans-def_ans, "fro")
            ret_values.append([h, (npm[0], npm[1], npm[0]+npm[1]+1), dif])
            #npmを足し合わせていきたいので最初はループ外
            for _ in range(step_num-1):
                h*= h_delta
                ans, npm = self.de_formula_complex(h, self.integrand3)
                dif = LA.norm(ans-def_ans, "fro")
                ret_values.append([h, (npm[0], npm[1], npm[0]+npm[1]+1), dif])
        return ret_values
    def simple_de(self, h, iter, method):
        """
        刻み幅，反復回数を引数に入れて計算
        反復回数は引数のiterによってインスタンス変数のself.maxnを変えるようにする
        そのため，途中で収束してしまったらiter以下で打ち切ってしまう
        この仕様を変えたかったら，新しいDE公式の関数を定義する必要あるかも

        parameter
        -----------------
        h: float  
        刻み幅

        iter: int
        反復回数

        method: string
        どの計算法でDE公式を行うか
        "1": 通常の被積分関数 f(x) = A(A^2 + t^2I)^-1
        "2": A^-1を掛けた被積分関数: f(x) = (A + t^2 A^-1)^-1
        "comp": 部分分数分解 : f(x) = ((A+itI)^-1 + (A+itI)^-1)

        return
        -----------------
        ans: array_like 
        解 行列

        """
        self.maxn = iter 
        if method == "1":
            ans, npm = self.de_formula(h, self.integrand1)
            
        elif method=="2":
            ans, npm = self.de_formula2(h, self.integrand2)
        elif method=="comp":
            ans, npm = self.de_formula_complex(h, self.integrand3)
        return ans,npm

    def de_fixedpoint_h(self, Np, d, method='1'):
        """hをN+とdから定義により求めて，その刻み幅，標本点数分でDE公式を計算する


        Parameters
        ----------
        Np : int
            正の部分の標本点数．負の部分の標本点数はこれと同じ値になるようにする
        d : float
            定義により求めたdの値を用いて計算
        method: string
            どの被積分関数の場合か
            '1': 元の被積分関数
            '2': (At^2 + A^-1)^-1
            'comp': 部分分数分解
        """
        h = np.log(8*d*Np)/Np
        self.maxn = Np 
        if method == "1":
            ans, _ = self.de_formula(h, self.integrand1, stop=False)
            
        elif method=="2":
            ans, _ = self.de_formula2(h, self.integrand2, stop=False)
        elif method=="comp":
            ans, _ = self.de_formula_complex(h, self.integrand3, stop=False)
        return ans


    def solve_ans(self,):
        """
        self.Aの解を固有値分解によって求める

        Returns
        -------
        sign(A)

        """
        eva, evv = LA.eig(self.A)
        sign_func = np.vectorize(self.my_sign)
        sign_eva = sign_func(eva) #固有値すべてにsign関数を適用
        lambda_matrix = np.diag(sign_eva) #sign_evaを対角成分にもつ対角行列
        ans = evv@lambda_matrix@LA.inv(evv)
        return ans
        
    def my_sign(self, z):
        if z.real > 0:
            return 1
        elif z.real < 0:
            return -1
        else:
            print("error:the real z is not to be zero")
            return 0
    def scaling_change(self,):
        """
        scalingによる高速化を行う
        A = cA
        とすることで固有値分布が収束の速くなるようにする
        c = 1/(sqrt(maxlambda*minlambda)
        maxlambda = norm(A, inf)
        minlambda = 1/(Ainv, inf)
        
        計算を行う前にこの関数を実行しておくと，
        self.A -> self.A * c
        self.Ainv -> (self.A*c)inv
        となった状態になる
        
        Returns
        -------
        None.

        """
        lambdamax_a = LA.norm(self.A, np.inf)
        lambdamin_a = (LA.norm(self.Ainv, np.inf))**(-1)
        c = 1/(np.sqrt(lambdamax_a * lambdamin_a))
        self.A = c*self.A 
        self.Ainv = LA.inv(self.A)
        return
    

#Schur分解を用いて行列符号関数を計算するためのクラス
class schur_calculator(object):
    """
        Parameters
        ----------
        A : ndarray
            入力行列．n*nの正方行列とする
            
        Returns
        -------
        None.
    """
    def __init__(self, A,):
        #入力行列
        self.A = A
        #入力行列のサイズ
        self.size = A.shape[0]

    def signs(self, s):
        """
        スカラーsign関数を計算するための関数
        Parameters
        ---------------
        s: float 
            sign(s)を計算するので，その引数s 
            sがcomplexならfloatにする

        returns
        ----------
        sign(s) : float(int ?) or None
            s=0の場合については定義されておらず，noneが返される．
            それ以外は，1か-1が返される
        """
        s = s.real
        if s > 0.0:
            return 1.0
        elif s < 0.0:
            return -1.0
        else:
            print("error: s=0 is not defined.")
    def hermitian(self, arr):
        """
        行列の共役転置をするための関数
        """
        return np.conjugate(arr.T)
    def solve(self):
        T, Q, _ = sp.linalg.schur(self.A, sort="lhp")
        #print(T)
        u = np.zeros((self.size, self.size))
        ud = np.array(range(self.size))
        t = np.diag(T)
        #tをLHPが先に来るように順序を変える(負の固有値が先頭に来るように並び換える)
        #安直にソート
        t = np.sort(t)
        ufunc = np.vectorize(self.signs) 
        ud = ufunc(t)
        np.fill_diagonal(u, ud)
        #print(u)
        eps_u = 1e-16
        for j in range(1, self.size):
            for i in range(j-1, 0, -1):
                if u[i, i] + u[j, j] != 0:
                    u[i, j] = -np.sum(u[i, i+1:j]*u[i+1:j, j])/(u[i,i] + u[j, j])
                else:
                    u[i, j] = T[i, j]*(u[i, i] - u[j, j])/(T[i,i] - T[j,j]) +np.sum(u[i,i+1:j]*T[i+1:j, j] - T[i, i+1:j]*u[i+1:j, j])/(T[i,i] - T[j, j])
                    #print("t ", i,j)
                    #print(T[i,i], T[j,j])
                    #print(u[i,i], u[j,j])
        return Q@u@(self.hermitian(Q))
    def solve_no_sort(self):
        T, Q = sp.linalg.schur(self.A)
        #print(T)
        u = np.zeros((self.size, self.size))
        ud = np.array(range(self.size))
        t = np.diag(T)
        ufunc = np.vectorize(self.signs) 
        ud = ufunc(t)
        np.fill_diagonal(u, ud)
        #print(u)
        eps_u = 1e-16
        for j in range(1, self.size):
            for i in range(j-1, 0, -1):
                if u[i, i] + u[j, j] != 0:
                    u[i, j] = -np.sum(u[i, i+1:j]*u[i+1:j, j])/(u[i,i] + u[j, j])
                else:
                    u[i, j] = T[i, j]*(u[i, i] - u[j, j])/(T[i,i] - T[j,j]) +np.sum(u[i,i+1:j]*T[i+1:j, j] - T[i, i+1:j]*u[i+1:j, j])/(T[i,i] - T[j, j])
                    #print("t ", i,j)
                    #print(T[i,i], T[j,j])
                    #print(u[i,i], u[j,j])
        return Q@u@(self.hermitian(Q))
    def using_eig(self, ):
        """
        固有値分解を用いた解析解の計算
        下のsolve_ansと同じ関数なので消してよし
        
        Parameter
        --------------------
        none

        Returns:
        --------------
        ans: ndarray
            解析解 sign(A)

        """
        eig, eigv = LA.eig(self.A)
        #print("eig: ", eig)
        sign_func = np.vectorize(self.signs)
        sign_eig = sign_func(eig) #固有値すべてにsigns関数を適用
        lambda_matrix = np.diag(sign_eig) #sign_eigを対角成分にもつ対角行列
        ans = eigv@lambda_matrix@LA.inv(eigv)
        return ans
    def solve_ans(self,):
        """
        self.Aの解を固有値分解によって求める

        Returns
        -------
        sign(A)

        """
        eva, evv = LA.eig(self.A)
        sign_func = np.vectorize(self.signs)
        sign_eva = sign_func(eva) #固有値すべてにsign関数を適用
        lambda_matrix = np.diag(sign_eva) #sign_evaを対角成分にもつ対角行列
        ans = evv@lambda_matrix@LA.inv(evv)
        return ans
    
class newton_calculator(object):
    """
    ニュートン法による行列符号関数計算クラス
    論文: the matrix sign function method and the computation of invariant subspace
    を参考にした
    収束条件の判定をこの論文参考
    他にもいろいろある

    最終的にほしいのは，
        計算結果
        数値誤差
        反復回数
        入力行列の条件数

    
    Parameters
    ----------
    A : ndarray
        入力行列．n*nの正方行列とする
        
    Returns
    -------
    None.
    
    """
    def __init__(self, A,):
        #入力行列
        self.A = A
        #入力行列のサイズ
        self.size = A.shape[0]
        #反復回数 solve関数実行後に更新
        self.iter = None 
    def signs(self, s):
        """
        スカラーsign関数を計算するための関数
        Parameters
        ---------------
        s: float 
            sign(s)を計算するので，その引数s 
            sがcomplexならfloatにする

        returns
        ----------
        sign(s) : float(int ?) or None
            s=0の場合については定義されておらず，noneが返される．
            それ以外は，1か-1が返される
        """
        s = s.real
        if s > 0.0:
            return 1.0
        elif s < 0.0:
            return -1.0
        else:
            print("error: s=0 is not defined.")
    def solve_ans(self,):
        """
        self.Aの行列符号関数の解を固有値分解によって求める

        Returns
        -------
        sign(A)

        """
        eva, evv = LA.eig(self.A)
        sign_func = np.vectorize(self.signs)
        sign_eva = sign_func(eva) #固有値すべてにsign関数を適用
        lambda_matrix = np.diag(sign_eva) #sign_evaを対角成分にもつ対角行列
        ans = evv@lambda_matrix@LA.inv(evv)
        return ans

    def solve_newton(self, max_iter = 30):
        """
        シンプルなnewton法で計算する
        X_{k+1} = 0.5*(X_k + X_k^-1)
        X_{0} = A

        Parameters
        ----------
        max_iter: int
            最大反復回数
            
        Returns
        -------
        solve_ans: ndarray
            計算結果
        """
        c = 1000*self.size
        iter_num = max_iter
        eps = 1e-16 #計算機イプシロン 精度によって変更 今回は倍精度
        solve_ans = np.copy(self.A) #初期化

        #以下で固有値分解によって解析解を求める
        def_ans = self.solve_ans()

        for i in range(max_iter):
            Ainv = LA.inv(solve_ans)
            #古い解を保存
            solve_ans_old = np.copy(solve_ans)
            #解の更新
            solve_ans = 0.5*(solve_ans + Ainv)
            
            #反復回数と絶対誤差を出力
            #print("iter_num: %f log(error) %f" % (i+1, np.log10(LA.norm(solve_ans - def_ans, 'fro'))))

            #収束性の判定
            norm_new2 = LA.norm(solve_ans, 1)**2
            norm_dif = LA.norm(solve_ans - solve_ans_old, 1)
            if c*eps*norm_new2 >= norm_dif:
                iter_num = i+1
                break
        self.iter = iter_num
        return solve_ans, def_ans
    def solve_newton_schulz(self, max_iter = 30):
        """
        newton法では逆行列を計算する必要があるが，newton-schulz法では逆行列を近似して計算するため，逆行列計算の必要がない．ただし行列乗算の回数は増える
        X_{k+1} = 1/2X_{k}(3I - X_{k}^2)
        X_{0} = A

        この計算法は||I-A^2|| < 1

        Parameters
        ----------
        max_iter: int
            最大反復回数
            
        Returns
        -------
        solve_ans: ndarray
            計算結果
        """
        c = 1000*self.size
        iter_num = max_iter
        eps = 1e-16 #計算機イプシロン 精度によって変更 今回は倍精度
        solve_ans = np.copy(self.A) #初期化

        #以下で固有値分解によって解析解を求める
        def_ans = self.solve_ans()

        for i in range(max_iter):
            #古い解を保存
            solve_ans_old = np.copy(solve_ans)
            #解の更新
            solve_ans = 0.5*solve_ans_old@(3*np.eye(self.size) - solve_ans_old@solve_ans_old)
            
            #反復回数と絶対誤差を出力
            print("iter_num: %f log(error) %f" % (i+1, np.log10(LA.norm(solve_ans - def_ans, 'fro'))))

            #収束性の判定
            norm_new2 = LA.norm(solve_ans, 1)**2
            norm_dif = LA.norm(solve_ans - solve_ans_old, 1)
            
            if c*eps*norm_new2 >= norm_dif:
                iter_num = i+1
                break
            
        self.iter = iter_num
        return solve_ans
    def get_iternum(self,):
        return self.iter

            
