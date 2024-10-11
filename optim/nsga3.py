# -*- coding: utf-8 -*-
# 作者: w.k.x.
# 链接: https://shivakasu.github.io/2018/12/28/work1/#4-%E6%BA%90%E7%A0%81
# 来源: SHIVAKASU
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
# 运行前修改代码最后一行 main 函数的参数。其中 Problem 是待测问题，
# 可选择的有 ‘test’ 、 ‘DTLZ1’ ~ ‘DTLZ7’ 、 ‘WFG1’ ~ ‘WFG9’ 。
# M 是目标个数，可设为 2 或 3。 Generations 是算法迭代次数，可设为任意正整数。
# Reference:
# K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm
# Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
# Problems With Box Constraints," in IEEE Transactions on Evolutionary Computation,
# vol. 18, no. 4, pp. 577-601, Aug. 2014, doi: 10.1109/TEVC.2013.2281535.


from scipy.special import comb
from itertools import combinations
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reference:
# Y. Tian, X. Xiang, X. Zhang, R. Cheng and Y. Jin, "Sampling Reference Points on
# the Pareto Fronts of Benchmark Multi-Objective Optimization Problems," 2018 IEEE
# Congress on Evolutionary Computation (CEC), 2018, pp. 1-6, doi: 10.1109/CEC.2018.8477730.

# 生成参考点
# Das and Dennis's method
def T_weight(H, M):
    N = int(comb(H + M - 1, M - 1))
    # 若H=4，M=3，则共产生N=15个参考点
    tmp1 = np.array(list(combinations(np.arange(1, H + M), M - 1)))
    # np.arange(1, H + M) 输出 array([1, ... , H + M -1])
    # tmp1 = {ndarray:(15,2)} ...
    # [[1 2], [1 3], [1 4], [1 5], [1 6], ...
    # [2 3], [2 4], [2 5], [2 6], ...
    # [3 4], [3 5], [3 6], ...
    # [4 5], [4 6],
    # [5 6]]
    tmp2 = np.tile(np.arange(0, M - 1), (int(comb(H + M - 1, M - 1)), 1))
    # np.arange(0, M - 1) 输出 array([0,...,M -2])
    # array([0,...,M -2]) 沿y轴复制N倍
    # tmp1 = {ndarray:(15,2)} ...
    # [[0 1], [0 1], [0 1], [0 1], [0 1], ...
    # [0 1], [0 1], [0 1], [0 1], ...
    # [0 1], [0 1], [0 1], ...
    # [0 1], [0 1], ...
    # [0 1]]
    Temp = tmp1 - tmp2 - 1
    # Temp = {ndarray:(15,2)} ...
    # [[0 0], [0 1], [0 2], [0 3], [0 4], ...
    # [1 1], [1 2], [1 3], [1 4], ...
    # [2 2], [2 3], [2 4], ...
    # [3 3], [3 4], ...
    # [4 4]]
    W = np.zeros((N, M))
    # W = {ndarray:(15,3)} ...
    # [[0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.], ...
    # [0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.], ...
    # [0. 0. 0.], [0. 0. 0.], [0. 0. 0.], ...
    # [0. 0. 0.], [0. 0. 0.], ...
    # [0. 0. 0.]]
    W[:, 0] = Temp[:, 0] - 0
    # W = {ndarray:(15,3)} ...
    # [[0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.], [0. 0. 0.], ...
    # [1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.], ...
    # [2. 0. 0.], [2. 0. 0.], [2. 0. 0.], ...
    # [3. 0. 0.], [3. 0. 0.], ...
    # [4. 0. 0.]]
    for i in range(1, M - 1):  # i 从 1 到 M-2
        W[:, i] = Temp[:, i] - Temp[:, i - 1]
        # W = {ndarray:(15,3)} ...
        # [[0. 0. 0.], [0. 1. 0.], [0. 2. 0.], [0. 3. 0.], [0. 4. 0.], ...
        # [1. 0. 0.], [1. 1. 0.], [1. 2. 0.], [1. 3. 0.], ...
        # [2. 0. 0.], [2. 1. 0.], [2. 2. 0.], ...
        # [3. 0. 0.], [3. 1. 0.], ...
        # [4. 0. 0.]]
    W[:, -1] = H - Temp[:, -1]
    # W = {ndarray:(15,3)} ...
    # [[0. 0. 4.], [0. 1. 3.], [0. 2. 2.], [0. 3. 1.], [0. 4. 0.], ...
    # [1. 0. 3.], [1. 1. 2.], [1. 2. 1.], [1. 3. 0.], ...
    # [2. 0. 2.], [2. 1. 1.], [2. 2. 0.], ...
    # [3. 0. 1.], [3. 1. 0.], ...
    # [4. 0. 0.]]
    W = W / H
    # W = {ndarray:(15,3)} ...
    # [[0.   0.   1.  ], [0.   0.25 0.75], [0.   0.5  0.5 ], [0.   0.75 0.25], [0.   1.   0.  ], ...
    # [0.25 0.   0.75], [0.25 0.25 0.5 ], [0.25 0.5  0.25], [0.25 0.75 0.  ], ...
    # [0.5  0.   0.5 ], [0.5  0.25 0.25], [0.5  0.5  0.  ], ...
    # [0.75 0.   0.25], [0.75 0.25 0.  ], ...
    # [1.   0.   0.  ]]
    return N, W  # N：生成一致性参考点的个数


# 生成一致性参考点
# Deb and Jain's Method
def F_weight(p1, p2, M):
    # M=3时，p1=13，p2=0
    N, W = T_weight(p1, M)
    if p2 > 0:
        N2, W2 = T_weight(p2, M) # 生成内层参考点
        N = N + N2  # S = S1与 S2的并集
        W = np.concatenate([W, W2 / 2 + 1 / (2 * M)])  # s'ij = si / 2 + 1 / (2 * M)
    return N, W  # N：生成一致性参考点的个数

# Reference:
# Y. Tian, R. Cheng, X. Zhang and Y. Jin, "PlatEMO: A MATLAB Platform for Evolutionary
# Multi-Objective Optimization [Educational Forum]," in IEEE Computational Intelligence
# Magazine, vol. 12, no. 4, pp. 73-87, Nov. 2017, doi: 10.1109/MCI.2017.2742868.
# 测试函数选择
def F_objective(Operation, p, M, Input):
    if p == 'test':
        Output, Boundary = F_test(Operation, p, M, Input)
    elif p[:3] == 'WFG':
        Output, Boundary = F_WFG(Operation, p, M, Input)
    elif p[:4] == 'DTLZ':
        Output, Boundary = F_DTLZ(Operation, p, M, Input)
    return Output, Boundary


def F_test(Operation, Problem, M, Input):
    Boundary = None
    if Operation == 'init':
        D = 10
        MaxValue = np.ones((1, D)) * 5
        MinValue = np.ones((1, D)) * (-5)
        Population = np.random.rand(Input, D)
        Population = Population * np.tile(MaxValue, (Input, 1)) + (1 - Population) * np.tile(MinValue, (Input, 1))
        Output = Population
        Boundary = np.array([MaxValue, MinValue])
    elif Operation == 'value':
        Population = Input
        FunctionValue = np.zeros((Population.shape[0], M))
        x1 = Population[:, 0]
        x2 = Population[:, 1]
        FunctionValue[:, 0] = x1 ** 4 - 10 * x1 ** 2 + x1 * x2 + x2 ** 4 - (x1 ** 2) * (x2 ** 2)
        FunctionValue[:, 1] = x2 ** 4 - (x1 ** 2) * (x2 ** 2) + x1 ** 4 + x1 * x2
        Output = FunctionValue
    return Output, Boundary


def F_DTLZ(Operation, Problem, M, Input):
    Boundary = None
    K = [5, 10, 10, 10, 10, 10, 20]
    K = K[int(Problem[-1]) - 1]
    if Operation == 'init':
        D = M + K - 1
        MaxValue = np.ones((1, D))
        MinValue = np.zeros((1, D))
        Population = np.random.rand(Input, D)
        Population = Population * np.tile(MaxValue, (Input, 1)) + (1 - Population) * np.tile(MinValue, (Input, 1))
        Output = Population
        Boundary = np.array([MaxValue, MinValue])
    elif Operation == 'value':
        Population = Input
        FunctionValue = np.zeros((Population.shape[0], M))
        if Problem == 'DTLZ1':
            g = 100 * (K + np.sum(
                (Population[:, M - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (Population[:, M - 1:] - 0.5)), axis=1))
            for i in range(M):
                FunctionValue[:, i] = 0.5 * np.prod(Population[:, 0:M - i - 1], axis=1) * (1 + g)
                if i > 0:
                    FunctionValue[:, i] = FunctionValue[:, i] * (1 - Population[:, M - i - 1])
        elif Problem == 'DTLZ2':
            g = np.sum((Population[:, M - 1:] - 0.5) ** 2, axis=1)
            for i in range(M):
                FunctionValue[:, i] = (1 + g) * np.prod(np.cos(0.5 * np.pi * Population[:, 0:M - i - 1]), axis=1)
                if i > 0:
                    FunctionValue[:, i] = FunctionValue[:, i] * np.sin(0.5 * np.pi * Population[:, M - i - 1])
        elif Problem == 'DTLZ3':
            g = 100 * (K + np.sum(
                (Population[:, M - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (Population[:, M - 1:] - 0.5)), axis=1))
            for i in range(M):
                FunctionValue[:, i] = (1 + g) * np.prod(np.cos(0.5 * np.pi * Population[:, 0:M - i - 1]), axis=1)
                if i > 0:
                    FunctionValue[:, i] = FunctionValue[:, i] * np.sin(0.5 * np.pi * Population[:, M - i - 1])
        elif Problem == 'DTLZ4':
            Population[:, 0:M - 2] = Population[:, 0:M - 2] ** 100
            g = np.sum((Population[:, M - 1:] - 0.5) ** 2, axis=1)
            for i in range(M):
                FunctionValue[:, i] = (1 + g) * np.prod(np.cos(0.5 * np.pi * Population[:, 0:M - i - 1]), axis=1)
                if i > 0:
                    FunctionValue[:, i] = FunctionValue[:, i] * np.sin(0.5 * np.pi * Population[:, M - i - 1])
        elif Problem == 'DTLZ5':
            g = np.sum((Population[:, M - 1:] - 0.5) ** 2, axis=1)
            Temp = np.tile(g, (1, M - 2))
            if M > 2:
                Temp = Temp.reshape(Temp.shape[1], Temp.shape[0])
            Population[:, 1:M - 1] = (1 + 2 * Temp * Population[:, 1:M - 1]) / (2 + 2 * Temp)
            for i in range(M):
                FunctionValue[:, i] = (1 + g) * np.prod(np.cos(0.5 * np.pi * Population[:, 0:M - i - 1]), axis=1)
                if i > 0:
                    FunctionValue[:, i] = FunctionValue[:, i] * np.sin(0.5 * np.pi * Population[:, M - i - 1])
        elif Problem == 'DTLZ6':
            g = np.sum(Population[:, M - 1:] ** 0.1, axis=1)
            Temp = np.tile(g, (1, M - 2))
            if M > 2:
                Temp = Temp.reshape(Temp.shape[1], Temp.shape[0])
            Population[:, 1:M - 1] = (1 + 2 * Temp * Population[:, 1:M - 1]) / (2 + 2 * Temp)
            for i in range(M):
                FunctionValue[:, i] = (1 + g) * np.prod(np.cos(0.5 * np.pi * Population[:, 0:M - i - 1]), axis=1)
                if i > 0:
                    FunctionValue[:, i] = FunctionValue[:, i] * np.sin(0.5 * np.pi * Population[:, M - i - 1])
        elif Problem == 'DTLZ7':
            g = 1 + 9 * np.mean(Population[:, M - 1:], axis=1)
            FunctionValue[:, 0:M - 1] = Population[:, 0:M - 1]
            if M > 2:
                Temp = np.tile(g, (1, M - 2))
                Temp = Temp.reshape(Temp.shape[1], Temp.shape[0])
            else:
                Temp = np.tile(g, (1, M - 1))
            h = M - np.sum(FunctionValue[:, 0:M - 1] / (1 + Temp) * (1 + np.sin(3 * np.pi * FunctionValue[:, 0:M - 1])),
                           axis=1)
            FunctionValue[:, M - 1] = (1 + g) * h
        Output = FunctionValue
    return Output, Boundary


def F_WFG(Operation, Problem, M, Input):
    K = [4, 4, 6, 8, 10, 0, 7, 0, 9]
    K = K[M - 2]
    L = 10
    Boundary = None
    if Operation == 'init':
        D = K + L
        MaxValue = np.arange(1, D + 1) * 2
        MinValue = np.zeros(D)
        Population = np.random.rand(Input, D)
        Population = Population * np.tile(MaxValue, (Input, 1)) + (1 - Population) * np.tile(MinValue, (Input, 1))
        Output = Population
        Boundary = np.array([MaxValue, MinValue])
    elif Operation == 'value':
        Population = Input
        N = Population.shape[0]
        D = 1
        S = np.tile(np.arange(1, M + 1) * 2, (N, 1))
        if Problem == 'WFG3':
            A = np.concatenate([np.ones((N, 1)), np.zeros((N, M - 2))], axis=1)
        else:
            A = np.ones((N, M - 1))
        z01 = Population / np.tile(np.arange(1, Population.shape[1] + 1) * 2, (N, 1))
        if Problem == 'WFG1':
            t1 = np.zeros((N, K + L))
            t1[:, 0:K] = z01[:, 0:K]
            t1[:, K:] = s_linear(z01[:, K:], 0.35)
            t2 = np.zeros((N, K + L))
            t2[:, 0:K] = t1[:, 0:K]
            t2[:, K:] = b_flat(t1[:, K:], 0.8, 0.75, 0.85)
            t3 = np.zeros((N, K + L))
            t3 = t2 ** 0.02
            t4 = np.zeros((N, M))
            for i in range(M - 1):
                t4[:, i] = r_sum(t3[:, i * K // (M - 1):(i + 1) * K // (M - 1)],
                                 np.arange(2 * (i * K // (M - 1)), 2 * (i + 1) * K // (M - 1), 2))
            t4[:, M - 1] = r_sum(t3[:, K:K + L], np.arange(2 * K, 2 * (K + L), 2))
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t4[:, M - 1], A[:, i]], axis=0) * (t4[:, i] - 0.5) + 0.5
            x[:, M - 1] = t4[:, M - 1]
            h = convex(x)
            h[:, M - 1] = mixed(x)
        elif Problem == 'WFG2':
            t1 = np.zeros((N, K + L))
            t1[:, 0:K] = z01[:, 0:K]
            t1[:, K:] = s_linear(z01[:, K:], 0.35)
            t2 = np.zeros((N, K + L // 2))
            t2[:, 0:K] = t1[:, 0:K]
            for i in range(K, K + L // 2):
                t2[:, i] = r_nonsep(t1[:, K + 2 * (i - K) - 2:K + 2 * (i - K)], 2)
            t3 = np.zeros((N, M))
            for i in range(M - 1):
                t3[:, i] = r_sum(t2[:, i * K // (M - 1):(i + 1) * K // (M - 1)], np.ones(K // (M - 1)))
            t3[:, M - 1] = r_sum(t2[:, K:K + L // 2], np.ones(L // 2))
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t3[:, M - 1], A[:, i]], axis=0) * (t3[:, i] - 0.5) + 0.5
            x[:, M - 1] = t3[:, M - 1]
            h = convex(x)
            h[:, M - 1] = disc(x)
        elif Problem == 'WFG3':
            t1 = np.zeros((N, K + L))
            t1[:, 0:K] = z01[:, 0:K]
            t1[:, K:] = s_linear(z01[:, K:], 0.35)
            t2 = np.zeros((N, K + L // 2))
            t2[:, 0:K] = t1[:, 0:K]
            for i in range(K, K + L // 2):
                t2[:, i] = r_nonsep(t1[:, K + 2 * (i - K) - 2:K + 2 * (i - K)], 2)
            t3 = np.zeros((N, M))
            for i in range(M - 1):
                t3[:, i] = r_sum(t2[:, i * K // (M - 1):(i + 1) * K // (M - 1)], np.ones(K // (M - 1)))
            t3[:, M - 1] = r_sum(t2[:, K:K + L // 2], np.ones(L // 2))
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t3[:, M - 1], A[:, i]], axis=0) * (t3[:, i] - 0.5) + 0.5
            x[:, M - 1] = t3[:, M - 1]
            h = linear(x)
        elif Problem == 'WFG4':
            t1 = np.zeros((N, K + L))
            t1 = s_multi(z01, 30, 10, 0.35)
            t2 = np.zeros((N, M))
            for i in range(M - 1):
                t2[:, i] = r_sum(t1[:, i * K // (M - 1):(i + 1) * K // (M - 1)], np.ones(K // (M - 1)))
            t2[:, M - 1] = r_sum(t1[:, K:K + L], np.ones(L))
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t2[:, M - 1], A[:, i]], axis=0) * (t2[:, i] - 0.5) + 0.5
            x[:, M - 1] = t2[:, M - 1]
            h = concave(x)
            h[:, M - 1] = disc(x)
        elif Problem == 'WFG5':
            t1 = np.zeros((N, K + L))
            t1 = s_decept(z01, 0.35, 0.001, 0.05)
            t2 = np.zeros((N, M))
            for i in range(M - 1):
                t2[:, i] = r_sum(t1[:, i * K // (M - 1):(i + 1) * K // (M - 1)], np.ones(K // (M - 1)))
            t2[:, M - 1] = r_sum(t1[:, K:K + L], np.ones(L))
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t2[:, M - 1], A[:, i]], axis=0) * (t2[:, i] - 0.5) + 0.5
            x[:, M - 1] = t2[:, M - 1]
            h = concave(x)
            h[:, M - 1] = disc(x)
        elif Problem == 'WFG6':
            t1 = np.zeros((N, K + L))
            t1[:, 0:K] = z01[:, 0:K]
            t1[:, K:] = s_linear(z01[:, K:], 0.35)
            t2 = np.zeros((N, M))
            for i in range(M - 1):
                t2[:, i] = r_nonsep(t1[:, i * K // (M - 1):(i + 1) * K // (M - 1)], 2)
            t2[:, M - 1] = r_nonsep(t1[:, K - 1:], L)
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t2[:, M - 1], A[:, i]], axis=0) * (t2[:, i] - 0.5) + 0.5
            x[:, M - 1] = t2[:, M - 1]
            h = concave(x)
            h[:, M - 1] = disc(x)
        elif Problem == 'WFG7':
            t1 = np.zeros((N, K + L))
            for i in range(K):
                t1[:, i] = b_param(z01[:, i], r_sum(z01[:, i:], np.ones(K + L - i)), 0.98 / 49.98, 0.02, 50)
            t1[:, K:] = z01[:, K:]
            t2 = np.zeros((N, K + L))
            t2[:, 0:K] = t1[:, 0:K]
            t2[:, K:] = s_linear(t1[:, K:], 0.35)
            t3 = np.zeros((N, M))
            for i in range(M - 1):
                t3[:, i] = r_sum(t2[:, i * K // (M - 1):(i + 1) * K // (M - 1)], np.ones(K // (M - 1)))
            t3[:, M - 1] = r_sum(t2[:, K:K + L], np.ones(L))
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t3[:, M - 1], A[:, i]], axis=0) * (t3[:, i] - 0.5) + 0.5
            x[:, M - 1] = t3[:, M - 1]
            h = concave(x)
            h[:, M - 1] = disc(x)
        elif Problem == 'WFG8':
            t1 = np.zeros((N, K + L))
            t1[:, 0:K] = z01[:, 0:K]
            for i in range(K, K + L):
                t1[:, i] = b_param(z01[:, i], r_sum(z01[:, 0:i - 1], np.ones(i - 1)), 0.98 / 49.98, 0.02, 50)
            t2 = np.zeros((N, K + L))
            t2[:, 0:K] = t1[:, 0:K]
            t2[:, K:] = s_linear(t1[:, K:], 0.35)
            t3 = np.zeros((N, M))
            for i in range(M - 1):
                t3[:, i] = r_sum(t2[:, i * K // (M - 1):(i + 1) * K // (M - 1)], np.ones(K // (M - 1)))
            t3[:, M - 1] = r_sum(t2[:, K:K + L], np.ones(L))
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t3[:, M - 1], A[:, i]], axis=0) * (t3[:, i] - 0.5) + 0.5
            x[:, M - 1] = t3[:, M - 1]
            h = concave(x)
            h[:, M - 1] = disc(x)
        elif Problem == 'WFG9':
            t1 = np.zeros((N, K + L))
            for i in range(K + L - 1):
                t1[:, i] = b_param(z01[:, i], r_sum(z01[:, i:], np.ones(K + L - i)), 0.98 / 49.98, 0.02, 50)
            t1[:, -1] = z01[:, -1]
            t2 = np.zeros((N, K + L))
            t2[:, 0:K] = s_decept(t1[:, 0:K], 0.35, 0.001, 0.05)
            t2[:, K:] = s_multi(t1[:, K:], 30, 95, 0.35)
            t3 = np.zeros((N, M))
            for i in range(M - 1):
                t3[:, i] = r_nonsep(t2[:, i * K // (M - 1):(i + 1) * K // (M - 1)], K // (M - 1))
            t3[:, M - 1] = r_nonsep(t2[:, K:], L)
            x = np.zeros((N, M))
            for i in range(M - 1):
                x[:, i] = np.max([t3[:, M - 1], A[:, i]], axis=0) * (t3[:, i] - 0.5) + 0.5
            x[:, M - 1] = t3[:, M - 1]
            h = concave(x)
            h[:, M - 1] = disc(x)
        Output = np.tile(D * x[:, M - 1].reshape((N, 1)), (1, M)) + S * h
    return Output, Boundary


def b_flat(y, A, B, C):
    Output = A + np.minimum(0, np.floor(y - B)) * A * (B - y) / B - np.minimum(0, np.floor(C - y)) * (1 - A) * (
                y - C) / (1 - C)
    Output = np.round(Output, -6)
    return Output


def b_param(y, Y, A, B, C):
    Output = y ** (B + (C - B) * (A - (1 - 2 * Y) * np.abs(np.floor(0.5 - Y) + A)))
    return Output


def s_linear(y, A):
    Output = np.abs(y - A) / np.abs(np.floor(A - y) + A)
    return Output


def s_decept(y, A, B, C):
    Output = 1 + (np.abs(y - A) - B) * (np.floor(y - A + B) * (1 - C + (A - B) / B) / (A - B) + np.floor(A + B - y) * (
                1 - C + (1 - A - B) / B) / (1 - A - B) + 1 / B)
    return Output


def s_multi(y, A, B, C):
    Output = (1 + np.cos((4 * A + 2) * np.pi * (0.5 - np.abs(y - C) / 2. / (np.floor(C - y) + C))) + 4 * B * (
                np.abs(y - C) / 2. / (np.floor(C - y) + C)) ** 2) / (B + 2)
    return Output


def r_sum(y, w):
    Output = np.sum(y * np.tile(w, (y.shape[0], 1)), axis=1) / np.sum(w)
    return Output


def r_nonsep(y, A):
    Output = np.zeros(y.shape[0])
    for j in range(y.shape[1]):
        Temp = np.zeros(y.shape[0])
        for k in range(A - 1):
            Temp = Temp + np.abs(y[:, j] - y[:, (j + k) % y.shape[1]])
        Output = Output + y[:, j] + Temp
    Output = Output / (y.shape[1] / A) / np.ceil(A / 2) / (1 + 2 * A - 2 * np.ceil(A / 2))
    return Output


def linear(x):
    Output = np.zeros(x.shape)
    for i in range(x.shape[1]):
        Output[:, i] = np.prod(x[:, 0:x.shape[1] - i - 1], axis=1)
        if i > 0:
            Output[:, i] = Output[:, i] * (1 - x[:, x.shape[1] - i - 1])
    return Output


def convex(x):
    Output = np.zeros(x.shape)
    for i in range(x.shape[1]):
        Output[:, i] = np.prod(1 - np.cos(x[:, 0:-1 - i] * np.pi / 2), axis=1)
        if i > 0:
            Output[:, i] = Output[:, i] * (1 - np.sin(x[:, x.shape[1] - i] * np.pi / 2))
    return Output


def concave(x):
    Output = np.zeros(x.shape)
    for i in range(x.shape[1]):
        Output[:, i] = np.prod(np.sin(x[:, 0:-1 - i] * np.pi / 2), axis=1)
        if i > 0:
            Output[:, i] = Output[:, i] * (np.cos(x[:, x.shape[1] - i] * np.pi / 2))
    return Output


def mixed(x):
    Output = 1 - x[:, 0] - np.cos(10 * np.pi * x[:, 0] + np.pi / 2) / 10 / np.pi
    return Output


def disc(x):
    Output = 1 - x[:, 0] * (np.cos(5 * np.pi * x[:, 0])) ** 2
    return Output


# 生成交配池
def F_mating(Population):
    N, D = Population.shape  # N为种群规模，D为决策空间维数
    MatingPool = np.zeros((N, D))  # 初始化交配池
    Rank = list(range(N))
    np.random.shuffle(Rank)  # 打乱顺序
    Pointer = 1
    for i in range(0, N, 2):
        k = np.zeros((1, 2))
        for j in range(2):
            if Pointer >= N:
                np.random.shuffle(Rank)  # 打乱顺序
                Pointer = 1
            k = Rank[Pointer - 1:Pointer + 1]
            Pointer = Pointer + 2
        MatingPool[i, :] = Population[k[0], :]
        if i + 1 < N:
            MatingPool[i + 1, :] = Population[k[1], :]
    return MatingPool  # 交配池


# 遗传算子
# Reference:
# K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective
# genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
# vol. 6, no. 2, pp. 182-197, April 2002, doi: 10.1109/4235.996017.
def F_generator(MatingPool, Boundary, MaxOffspring):
    N, D = MatingPool.shape  # N为种群规模，D为决策空间维数
    ProC = 1  # 交叉概率
    ProM = 1 / D  # 变异概率
    DisC = 20  # 交叉参数t1
    DisM = 20  # 变异参数t2
    Offspring = np.zeros((N, D))  # 子代

    # Qt = 交叉+变异（Pt）
    # 模拟二进制交叉
    # SBX主要是模拟基于二进制串的单点交叉工作原理,将其作用于以实数表示的染色体。
    # 两个父代染色体经过交叉操作产生两个子代染色体,使得父代染色体的有关模式信息在子代染色体中得以保留。
    # Reference:
    # https://blog.csdn.net/Ryze666/article/details/123826212
    for i in range(0, N, 2):
        beta = np.zeros((1, D))
        miu = np.random.rand(1, D)  # 随机数
        beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))  # beta = (rand*2)^(1/(1+DisC), rand<=0.5
        beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))  # beta = (1/(2-rand*2))^(1/(1+DisC)), otherwise
        beta = beta * (-1) ** np.random.randint(2, size=(1, D))
        beta[np.random.rand(1, D) > ProC] = 1
        if i + 1 < N:
            Offspring[i, :] = (MatingPool[i, :] + MatingPool[i + 1, :]) / 2 + beta * (
                        MatingPool[i, :] - MatingPool[i + 1, :]) / 2
            Offspring[i + 1, :] = (MatingPool[i, :] + MatingPool[i + 1, :]) / 2 - beta * (
                        MatingPool[i, :] - MatingPool[i + 1, :]) / 2
            # ci1 = 0.5 * ((1+beta)*xi1+(1-beta)*xi2)
            # ci2 = 0.5 * ((1-beta)*xi1+(1+beta)*xi2)
    Offspring = Offspring[0:MaxOffspring, :]
    # 使子代在定义域内
    if MaxOffspring == 1:
        MaxValue = Boundary[0, :]  # 最大值
        MinValue = Boundary[1, :]  # 最小值
    else:
        MaxValue = np.tile(Boundary[0, :], (MaxOffspring, 1))  # 最大值
        MinValue = np.tile(Boundary[1, :], (MaxOffspring, 1))  # 最小值

    # 多项式变异
    # Reference:
    # http://www.bubuko.com/infodetail-1933372.html
    k = np.random.rand(MaxOffspring, D)
    miu = np.random.rand(MaxOffspring, D)
    Temp = (k <= ProM) * (miu < 0.5)
    Offspring[Temp] = Offspring[Temp] + (MaxValue[Temp] - MinValue[Temp]) * ((2 * miu[Temp] + (1 - 2 * miu[Temp]) * (
                1 - (Offspring[Temp] - MinValue[Temp]) / (MaxValue[Temp] - MinValue[Temp])) ** (DisM + 1)) ** (
                                                                                             1 / (DisM + 1)) - 1)
    Temp = (k <= ProM) * (miu >= 0.5)
    Offspring[Temp] = Offspring[Temp] + (MaxValue[Temp] - MinValue[Temp]) * (1 - (
                2 * (1 - miu[Temp]) + 2 * (miu[Temp] - 0.5) * (
                    1 - (MaxValue[Temp] - Offspring[Temp]) / (MaxValue[Temp] - MinValue[Temp])) ** (DisM + 1)) ** (
                                                                                              1 / (DisM + 1)))
    # v'k = vk + delta * (uk-lk)
    # delta = (2*u+(1-2*u)*(1-delta1)^(nm+1))^(1/(nm+1)) - 1,  if u <= 0.5
    # delta = 1 - (2*(1-u)+2*(u-0.5)*(1-delta2)^(nm+1))^(1/(nm+1)),  if u > 0.5
    # delta1 = (vk-lk)/(uk-lk)
    # delta2 = (uk-vk)/(uk-lk)
    # 使子代在定义域内
    Offspring[Offspring > MaxValue] = MaxValue[Offspring > MaxValue]
    Offspring[Offspring < MinValue] = MinValue[Offspring < MinValue]
    return Offspring  # 子代


# 使用非支配排序对种群进行排序
def F_sort(FunctionValue):
    Kind = 2
    N, M = FunctionValue.shape  # N 为种群规模，M 为目标数量
    MaxFront = 0  # 最高等级初始化为0
    cz = np.zeros(N)
    FrontValue = np.zeros(N) + np.inf  # 初始化所有等级为np.inf
    Rank = FunctionValue.argsort(axis=0)[:, 0]
    FunctionValue = FunctionValue[np.lexsort(FunctionValue[:, ::-1].T)]
    while (Kind == 1 and np.sum(cz) < N) or (Kind == 2 and np.sum(cz) < N / 2) or (Kind == 3 and MaxFront < 1):
        MaxFront = MaxFront + 1
        d = deepcopy(cz)
        for i in range(N):  # 对于每个p,N为支配体的个数
            if d[i] == 0:  # p 支配的个体集合初始化为零
                for j in range(i + 1, N):  # 对于每一个q,N为支配体的个数
                    if d[j] == 0:  # q 支配的个体集合初始化为零
                        k = 1
                        for m in range(1, M):  # 在每一维上
                            # 判断个体i和个体j的支配关系
                            if FunctionValue[i, m] > FunctionValue[j, m]:  # 如果p支配q
                                k = 0  # 将 q 添加到由 p 支配的解集中
                                break
                        if k == 1:  # 如果q支配p
                            d[j] = 1  # 将 p 添加到由 q 支配的解集中
                FrontValue[Rank[i]] = MaxFront
                cz[i] = 1
    return FrontValue, MaxFront


def F_choose(FunctionValue1, FunctionValue2, K, Z):
    FunctionValue = np.concatenate([FunctionValue1, FunctionValue2])
    N, M = FunctionValue.shape
    N1 = FunctionValue1.shape[0]
    N2 = FunctionValue2.shape[0]
    NoZ = Z.shape[0]
    Zmin = np.min(FunctionValue, axis=0)  # 计算理想点
    Extreme = np.zeros(M)
    w = np.zeros((M, M)) + 0.000001 + np.eye(M)  # wj=(epsilong ,..., epsilong)^T
    # epsilong= 10^(−6)，且 wjj = 1
    for i in range(M):  # for j=1 to M do
        # 通过寻找使以下标量化函数最小的解（ x  St ）来确定每个目标轴上的极值点
        Extreme[i] = np.argmin(np.max(FunctionValue / np.tile(w[i, :], (N, 1)), axis=1))
        # ASF(x,w) = maxMi=1 fi'(x) / wi,  for x Si
    Hyperplane = np.linalg.lstsq(FunctionValue[Extreme.astype('int'), :], np.ones((M, 1)))[0]
    a = 1 / Hyperplane  # 计算超平面的截距 ai
    if np.any(np.isnan(a)):
        a = np.max(FunctionValue, axis=0).T
    # 进行目标函数的归一化
    FunctionValue = (FunctionValue - np.tile(Zmin, (N, 1))) / (np.tile(a.T, (N, 1)) - np.tile(Zmin, (N, 1)))  # 映射后的目标
    # fin(x) = ( fi(x) - zimin ) / ( ai - zimin ) ,for i=1,2...,M
    Distance = np.zeros((N, NoZ))
    normZ = np.sum(Z ** 2, axis=1) ** 0.5
    normF = np.sum(FunctionValue ** 2, axis=1) ** 0.5

    # 关联操作
    for i in range(N):  # 对于每个参考点
        normFZ = np.sum((np.tile(FunctionValue[i, :], (NoZ, 1)) - Z) ** 2, axis=1) ** 0.5
        for j in range(NoZ):  # 对于每个z
            S1 = normF[i]
            S2 = normZ[j]
            S3 = normFZ[j]
            p = (S1 + S2 + S3) / 2
            Distance[i, j] = 2 * np.sqrt(p * (p - S1) * (p - S2) * (p - S3)) / S2
            # 计算 St 的每个种群个体与每条参考线的垂直距离
    # 在归一化目标空间中，参考线最接近种群个体的参考点被认为与该种群个体相关联
    d = np.min(Distance.T, axis=0)  # Assignd(s)
    pii = np.argmin(Distance.T, axis=0)  # Assignπ(s)
    rho = np.zeros(NoZ)  # 第 j 个参考点的小生境计数表示为 ρj
    for i in range(N1):
        rho[pii[i]] = rho[pii[i]] + 1
    Choose = np.zeros(N2)
    Zchoose = np.ones(NoZ)

    # 小生境保留操作
    k = 1
    while k <= K:  # while k <= K do
        Temp = np.argwhere(Zchoose == 1).ravel()
        j = np.argmin(rho[Temp])  # 确定具有最小 ρj 的参考点集 Jmin = j:argmin j,ρj
        j = Temp[j]
        I1 = (np.ravel(Choose) == 0).nonzero()  # 在有多个这样的参考点的情况下，随机选择一个j
        I2 = (np.ravel(pii[N1:]) == j).nonzero()
        I = np.intersect1d(I1, I2)
        if len(I) > 0:
            if rho[j] == 0:  # 意味着没有与参考点 j 相关联的 Pt+1 个体
                s = np.argmin(d[N1 + I])  # Pt+1 = Pt+1 并 ( s:argmins Ij_ d(s) )
                # 参考线垂直距离最短的那个个体被添加到 Pt+1
            else:  # 意味着 St/Fl 中已经有一个与参考点相关联的个体存在
                s = np.random.randint(len(I))  # Pt+1 = Pt+1 并 rand(Ij_)
            Choose[I[s]] = 1  # Fl = Fl / s
            # 从前面 Fl 中随机选择一个与参考点 j 相关联的个体（如果存在的话）添加到 Pt+1
            rho[j] = rho[j] + 1  # 小生境计数更新
            k = k + 1
        else:
            Zchoose[j] = 0  # Zr = Zr / {j_}
    return Choose


# 主函数
def main(Problem, M, Generations):
    assert M == 2 or M == 3  # M 是目标个数，可设为 2 或 3
    p1 = [99, 13, 7, 5, 4, 3, 3, 2, 3]  # 外层参考点的个数
    p2 = [0, 0, 0, 0, 1, 2, 2, 2, 2]  # 内层参考点的个数
    p1 = p1[M - 2]
    p2 = p2[M - 2]
    N, Z = F_weight(p1, p2, M)  # 生成一致性参考点
    breakpoint()
    Z[Z == 0] = 0.000001  # 对于 wi = 0 ，我们用一个很小的数字 10^(−6) 来代替它
    Population, Boundary = F_objective('init', Problem, M, N)  # 测试函数选择
    plt.ion()  # 绘制动态图
    F = None
    if M == 3:  # 如果目标个数为3
        fig = plt.figure(figsize=plt.figaspect(0.5))
    for Gene in range(Generations):  # 对于每一代
        MatingPool = F_mating(Population)  # 生成交配池
        Offspring = F_generator(MatingPool, Boundary, N)  # 遗传算子
        Population = np.concatenate([Population, Offspring])  # 新一代种群Rt = Pt  Qt
        FunctionValue = F_objective('value', Problem, M, Population)[0]  # 目标函数评估
        FrontValue, MaxFront = F_sort(deepcopy(FunctionValue))  # 使用非支配排序对种群进行排序
        # (F1, F2,...) = 非支配性排序(Rt)
        Next = np.zeros(N)
        NoN = np.count_nonzero(FrontValue[FrontValue < MaxFront])
        Next[:NoN] = (np.ravel(FrontValue.T) < MaxFront).nonzero()[0]
        Last = (np.ravel(FrontValue.T) == MaxFront).nonzero()[0]
        c = Next[:NoN].astype('int')
        Choose = F_choose(FunctionValue[c, :], FunctionValue[Last, :], N - NoN, Z)
        Next[NoN:N] = Last[Choose.astype(bool)]
        Population = Population[Next.astype('int'), :]
        F = F_objective('value', Problem, M, Population)[0]
        plt.cla()
        if M == 3:
            fig.suptitle(Problem + "   Generation=" + str(Gene + 1))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            fig.delaxes(fig.axes[0])
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], color='b')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.view_init(10, 50)
            ax2.scatter(F[:, 0], F[:, 1], F[:, 2], color='b')
            if Gene == Generations - 1:
                plt.ioff()
                plt.show()
            else:
                plt.pause(0.2)

        else:
            plt.title(Problem + "   Generation=" + str(Gene + 1))
            plt.scatter(F[:, 0], F[:, 1], color='b')
            if Gene == Generations - 1:
                plt.ioff()
                plt.scatter(F[:, 0], F[:, 1], color='b')
                plt.show()
            else:
                plt.pause(0.2)


if __name__ == "__main__":
    main(Problem="DTLZ2", M=3, Generations=500)

# 可选择的有 ‘test’ 、 ‘DTLZ1’ ~ ‘DTLZ7’ 、 ‘WFG1’ ~ ‘WFG9’
# 由于程序运行耗时较长，因此测试中限定双目标优化问题的迭代次数为300次，
# 三目标优化问题的迭代次数为500次
# 测试结果反映了算法的拟合性能，同时也反映了不同问题收敛难度的区别。
# 通过与真实Pareto前沿的直观对比，可以看出如DTLZ2、DTLZ3等问题的拟合难度比较低，
# 在500次迭代后拟合结果已经十分理想。相反，如DTLZ1、WFG4等问题的你和难度比较高，
# 在500次迭代后拟合结果仍比较粗陋，表明算法的性能有待提高。
