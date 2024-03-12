# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:16:44 2023

@author: miyas

行列を作成するクラス
"""

import numpy as np
import numpy.linalg as LA


class Make_matrix(object):
    def __init__(self, emin = 0.1, emax = 10, size = 100, mattype = "normal"):
        """
        対角化可能な任意の行列を作るための関数

        Parameters
        ----------
        emin : float
            最小固有値 The default is 0.1.
        emax : float
            最大固有値 The default is 10.
        size : int
            行列サイズn The default is 100.
        mattype : string
            The default is "normal".
            if "normal" the matrix type is non symmetric
            if "sym" the matrix type is symmetric
            
            you need other type matrix, please add other type and 
            describe here
        Returns
        -------
        None.

        """
        self.emin = emin
        self.emax = emax
        self.size = size
        self.mattype = mattype
        self.a = np.zeros((self.size, self.size))
        
    def matrix(self, random_state = 1):
        np.random.seed(random_state)
        a = np.zeros((self.size, self.size))
        if self.mattype == "normal":
            X = np.random.randn(self.size, self.size)
            Xinv = LA.inv(X)
            #a = np.zeros((self.size, self.size))
            a[0, 0] = self.emax
            a[1, 1] = self.emin
            for i in range(2, self.size):
                a[i, i] = (self.emax - self.emin)*np.random.rand() + self.emin
            self.a = np.copy(a)
            return X@a@Xinv 
        elif self.mattype == "sym":
            #対称行列　直交行列で対角化可能であるという性質を用いる
            X = np.random.randn(self.size, self.size)
            Q, R = LA.qr(X)
            Qinv = LA.inv(Q)
            #a = np.zeros((self.size, self.size))
            a[0, 0] = self.emax
            a[1, 1] = self.emin
            for i in range(2, self.size):
                a[i, i] = (self.emax - self.emin)*np.random.rand() + self.emin
            self.a = np.copy(a)
            return Q@a@Qinv
        else :
            print("error: please check your mattype name. (return zeros matrix)")
            return np.zeros((self.size, self.size))
        
    def get_a(self):
        return self.a
    
"""
上のクラスは固有値分布をインスタンス化したときにするようにしてるが，以下ではmatrix関数を呼び出したときに
決めるようにしている
"""

class Make_matrix2(object):
    def __init__(self, size = 100, mattype = "normal"):
        """
        Parameters
        ----------
        size : TYPE, optional
            DESCRIPTION. The default is 100.
        mattype : TYPE, optional
            DESCRIPTION. The default is "normal".
            if "normal" the matrix type is non symmetric
            if "sym" the matrix type is symmetric
            
            you need other type matrix, please add other type and 
            describe here
        Returns
        -------
        None.

        """

        self.size = size
        self.mattype = mattype
        self.a = np.zeros((self.size, self.size))
        #Xcond2はXの2ノルム条件数をもつ
        self.Xcond2 = 0.0
        self.X = None
    
    def matrix(self, random_state = 1, emax=10, emin = 0.1 ):
        np.random.seed(random_state)
        a = np.zeros((self.size, self.size))
        if self.mattype == "normal":
            X = np.random.randn(self.size, self.size)
            self.X = np.copy(X)
            self.Xcond2 = np.linalg.cond(X, p=2)
            Xinv = LA.inv(X)
            #a = np.zeros((self.size, self.size))
            a[0, 0] = emax
            a[1, 1] = emin
            for i in range(2, self.size):
                a[i, i] = (emax - emin)*np.random.rand() + emin
            self.a = np.copy(a)
            return X@a@Xinv 
        elif self.mattype == "sym":
            #対称行列　直交行列で対角化可能であるという性質を用いる
            X = np.random.randn(self.size, self.size)
            Q, _ = LA.qr(X)
            Qinv = LA.inv(Q)
            self.X = np.copy(Q)
            self.Xcond2 = np.linalg.cond(Q, p=2)
            #a = np.zeros((self.size, self.size))
            a[0, 0] = emax
            a[1, 1] = emin
            for i in range(2, self.size):
                a[i, i] = (emax - emin)*np.random.rand() + emin
            self.a = np.copy(a)
            return Q@a@Qinv
        else :
            print("error: please check your mattype name. (return zeros matrix)")
            return np.zeros((self.size, self.size))
    def matrix_nega(self, random_state = 1, emax = 10.0, emin = 0.1, negative_state = 2):
        """
        matrix関数では，固有値がすべて正にしてあるが，この関数では固有値に負の値も含むようにする
        最小固有値は必ず負にしている
        negative_state引数は乱数の引数でありどれだけ負の固有値を含むかを決める
        """
        rng1 = np.random.default_rng(random_state + 1)
        rng_lambda = np.random.default_rng(random_state)
        rng_nega = np.random.default_rng(negative_state)
        a = np.zeros((self.size, self.size))
        if self.mattype == "normal":
            X = rng1.random((self.size, self.size))
            self.X = np.copy(X)
            self.Xcond2 = np.linalg.cond(X, p=2)
            Xinv = LA.inv(X)
            #a = np.zeros((self.size, self.size))
            a[0, 0] = emax
            a[1, 1] = -emin
            for i in range(2, self.size):
                ransuu = (emax - emin)*rng_lambda.random()
                a[i, i] = ransuu if rng_nega.random() > 0.5 else -ransuu 
            self.a = np.copy(a)
            return X@a@Xinv 
        elif self.mattype == "sym":
            #対称行列　直交行列で対角化可能であるという性質を用いる
            X = rng1.random((self.size, self.size))
            Q, _ = LA.qr(X)
            self.X = np.copy(Q)
            self.Xcond2 = np.linalg.cond(Q, p=2)
            Qinv = LA.inv(Q)
            #a = np.zeros((self.size, self.size))
            a[0, 0] = emax
            a[1, 1] = -emin
            for i in range(2, self.size):
                ransuu = (emax - emin)*rng_lambda.random()
                a[i, i] = ransuu if rng_nega.random() > 0.5 else -ransuu
            self.a = np.copy(a)
            return Q@a@Qinv
        else :
            print("error: please check your mattype name. (return zeros matrix)")
            return np.zeros((self.size, self.size))
        
    #A = XaX^-1として行列Aを作成したときの行列aを取り出す関数
    def get_a(self):
        return self.a
    def get_Xcond2(self):
        return self.Xcond2
    def get_X(self):
        if self.X is not None:
            return self.X
        else:
            print("X is not defined")
            return None