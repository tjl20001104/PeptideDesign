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
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import utils


# 生成参考点
# Das and Dennis's method
def uniformpoint(N,M):
    H1=1
    while (comb(H1+M-1,M-1)<=N):
        H1=H1+1
    H1=H1-1
    W=np.array(list(combinations(range(H1+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H1+M-1,M-1)),1))
    W=(np.hstack((W,H1+np.zeros((W.shape[0],1))))-np.hstack((np.zeros((W.shape[0],1)),W)))/H1
    if H1<M:
        H2=0
        while(comb(H1+M-1,M-1)+comb(H2+M-1,M-1) <= N):
            H2=H2+1
        H2=H2-1
        if H2>0:
            W2=np.array(list(combinations(range(H2+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H2+M-1,M-1)),1))
            W2=(np.hstack((W2,H2+np.zeros((W2.shape[0],1))))-np.hstack((np.zeros((W2.shape[0],1)),W2)))/H2
            W2=W2/2+1/(2*M)
            W=np.vstack((W,W2))#按列合并
    W[W<1e-6]=1e-6
    N=W.shape[0]
    return W,N


def F_objective(Operation, Model, dataset, labels, targets, Boundary, Input):
    if Operation == 'init':
        Population = Model.sample_z_prior(Input)
        Population = Population.detach().cpu().numpy()
        Population = np.clip(Population, Boundary[0], Boundary[1])
        # Population = torch.rand(Input, 100)*10 - 5
        # Population = Population.detach().cpu().numpy()
        output = Population
    elif Operation == 'value':
        Population = torch.from_numpy(Input).float().to(Model.device)
        sequences = utils.decode_from_z(Population, Model, dataset)
        sequences = [line.split(' ') for line in sequences]
        len_list = torch.tensor([len(line)+2 for line in sequences]).to(Model.device)
        sequences = torch.stack(dataset.sentences2idx(sequences)).to(Model.device)
        (z_mu, z_logvar), z, (dec_logits_z, dec_logits_gau) = Model(sequences, len_list, sample_z='max')
        clfs = [getattr(Model, label+'_classifier') for label in labels]
        preds = []
        for clf,target in zip(clfs,targets):
            pred = clf(z)
            pred = target*pred + (1-target)*(1-pred)
            preds.append(pred)
        preds = torch.cat(preds,dim=1).detach().cpu().numpy()
        output = preds
    return output



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
    DisM = 500  # 变异参数t2
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
        MaxValue = Boundary[0]  # 最大值
        MinValue = Boundary[1]  # 最小值
    else:
        MaxValue = np.ones_like(Offspring) * Boundary[1]  # 最大值
        MinValue = np.ones_like(Offspring) * Boundary[0]  # 最小值

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
    Offspring = np.clip(Offspring, Boundary[0], Boundary[1])
    return Offspring  # 子代


# 使用非支配排序对种群进行排序
def F_sort(FunctionValue, initial_population_num):
    Kind = 2
    N,M = FunctionValue.shape  # N 为种群规模，M 为目标数量
    MaxFront = 0  # 最高等级初始化为0
    cz = np.zeros(N)
    FrontValue = np.zeros(N) + np.inf  # 初始化所有等级为np.inf
    Rank = FunctionValue.argsort(axis=0)[:, 0]
    FunctionValue = FunctionValue[np.lexsort(FunctionValue[:, ::-1].T)]
    while (Kind == 1 and np.sum(cz) < N) or (Kind == 2 and np.sum(cz) < initial_population_num) or (Kind == 3 and MaxFront < 1):
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


def F_sift(Population, FunctionValue, percentile):
    thresholds = np.percentile(FunctionValue,20,axis=0)
    sift_dim = np.argmin(thresholds)
    # print('dim={},median={}'.format(sift_dim,thresholds[sift_dim]))
    position = FunctionValue[:,sift_dim]>thresholds[sift_dim]
    Population_sifted, FunctionValue_sifted = Population[position,:], FunctionValue[position,:]
    return Population_sifted, FunctionValue_sifted


# 主函数
def NSGAIII_searching(Model, labels, targets, Boundary, Thresholds, Generations, N, dataset):
    Thresholds_used = [target*threshold+(1-target)*(1-threshold) 
                       for target,threshold in zip(targets,Thresholds)]
    M = len(labels)
    Z, N = uniformpoint(N, M)  # 生成一致性参考点
    Z[Z == 0] = 0.000001  # 对于 wi = 0 ，我们用一个很小的数字 10^(−6) 来代替它
    Population = F_objective('init', Model, dataset, labels, targets, Boundary, N)  # 测试函数选择
    plt.ion()  # 绘制动态图
    F = None
    for Gene in range(Generations):  # 对于每一代
        MatingPool = F_mating(Population)  # 生成交配池
        breakpoint()
        Offspring = F_generator(MatingPool, Boundary, N)  # 遗传算子
        Population = np.concatenate([Population, Offspring])  # 新一代种群Rt = Pt  Qt
        FunctionValue = F_objective('value', Model, dataset, labels, targets, Boundary, Population)  # 目标函数评估
        Population, FunctionValue = F_sift(Population, FunctionValue, percentile=20)
        FrontValue, MaxFront = F_sort(deepcopy(FunctionValue), initial_population_num=N)  # 使用非支配排序对种群进行排序
        # (F1, F2,...) = 非支配性排序(Rt)
        Next = np.zeros(N)
        NoN = np.count_nonzero(FrontValue[FrontValue < MaxFront])
        Next[:NoN] = (np.ravel(FrontValue.T) < MaxFront).nonzero()[0]
        Last = (np.ravel(FrontValue.T) == MaxFront).nonzero()[0]
        c = Next[:NoN].astype('int')
        Choose = F_choose(FunctionValue[c, :], FunctionValue[Last, :], N - NoN, Z)
        Next[NoN:N] = Last[Choose.astype(bool)]
        Population = Population[Next.astype('int'), :]
        F = F_objective('value', Model, dataset, labels, targets, Boundary, Population)
        selected_position = (F[:,0]>-1)
        for i in range(len(Thresholds_used)):
            selected_position = selected_position&(F[:,i]>Thresholds_used[i])
        F_select = F[selected_position]
        if sum(selected_position) >= len(F)*0.2:
            break
        
        plt.cla()
        plt.title("Generation={},num_selected={}".format(Gene+1,sum(selected_position)))
        plt.scatter(F[:, 0], F[:, 1], color='#0000FF', s=10)
        plt.scatter(F_select[:,0], F_select[:,1], color='#FF0000', s=10)
        if Gene == Generations - 1:
            plt.ioff()
            plt.scatter(F[:, 0], F[:, 1], color='#0000FF', s=10)
            plt.scatter(F_select[:,0], F_select[:,1], color='#FF0000', s=5)
            plt.show()
        else:
            plt.pause(0.1)
    
    return F, selected_position, Population
