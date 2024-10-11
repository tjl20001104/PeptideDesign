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
import pandas as pd
from sample_pipeline import compute_modlamp


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


def Process_sequences(Population, dataset):
    s_all = []
    sequences = [line.split(' ') for line in Population]
    for sequence in sequences:
        sequence = ['<start>'] + sequence + ['<eos>']
        sequence = sequence + ['<pad>'] * (dataset.fixed_length + 2 - len(sequence))
        s_all.append(dataset.vocab(sequence))
    return np.array(s_all)


def F_objective(Operation, Model, dataset, labels, targets, Input):
    if Operation == 'init':
        Model.eval()
        Population = Model.sample_z_prior(Input)
        Population = utils.decode_from_z(Population, Model, dataset)
        Population = Process_sequences(Population, dataset)
        output = Population
    elif Operation == 'value':
        Model.eval()
        sequences = torch.from_numpy(Input).int().to(Model.device)
        sequences = dataset.idx2sentences(sequences, print_special_tokens=False)
        df = pd.DataFrame({'peptide': sequences})
        df = utils.clf_pred(labels, df, Model, dataset)
        value = df[labels].values
        output = np.zeros_like(value)
        for i in range(len(targets)):
            output[:,i] = targets[i]*value[:,i] + (1-targets[i])*(1-value[:,i])
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
def F_generator(MatingPool, MaxOffspring):
    N, D = MatingPool.shape  # N为种群规模，D为决策空间维数
    ProC = 1
    upper_segment_num = 3 # 多段交叉段数上限
    lower_segment_num = 1 # 多段交叉段数下限
    ProM = 0.005  # 变异概率
    Offspring = np.zeros((N, D))  # 子代

    # Qt = 交叉+变异（Pt）
    # 多处进行两点交叉
    for i in range(0, N, 2):
        segment_num = np.random.randint(low=lower_segment_num, high=upper_segment_num+1)
        places = np.random.choice(D, 2*segment_num, replace=False)
        if i + 1 < N:
            Offspring[i, :] = MatingPool[i, :]
            Offspring[i + 1, :] = MatingPool[i + 1, :]
            for j in range(segment_num):
                Offspring[i, places[j]:places[j+1]] = MatingPool[i + 1, places[j]:places[j+1]]
                Offspring[i + 1, places[j]:places[j+1]] = MatingPool[i, places[j]:places[j+1]]
    Offspring = Offspring[0:MaxOffspring, :]
    for i in range(Offspring.shape[0]):
        for j in range(Offspring.shape[1]):
            if Offspring[i,j] == 3:
                Offspring[i,(j+1):] = 1

    # 多点均匀变异
    for i in range(N):
        random_mutation = np.random.uniform(low=0.0,high=1.0,size=1)
        if_mutation = random_mutation < D*ProM
        if if_mutation:
            position_eos = np.where(Offspring[i,:]==3)[0].tolist()
            if position_eos == []:
                for j in range(D):
                    if j <= 5:
                        Offspring[i,j] = np.random.randint(low=4, high=24)
                    else:
                        Offspring[i,j] = np.random.randint(low=2, high=24)
            else:
                position_mutation = np.random.randint(low=0,high=52)
                position_eos = position_eos[0]
                if position_mutation > position_eos:
                    Offspring[i,position_eos:position_mutation] = \
                    np.random.randint(low=4, high=24, size=position_mutation-position_eos)
                elif position_mutation <= 5:
                    Offspring[i,position_mutation] = np.random.randint(low=4, high=24)
                else:
                    Offspring[i,position_mutation] = np.random.randint(low=2, high=24)
        # for j in range(D):
        #     if mutation_positions[i,j] == True:
        #         if j <= 5:
        #             Offspring[i,j] = np.random.randint(low=4, high=24)
        #         else:
        #             Offspring[i,j] = np.random.randint(low=2, high=24)

    for i in range(Offspring.shape[0]):
        for j in range(Offspring.shape[1]):
            if Offspring[i,j] == 3:
                Offspring[i,(j+1):] = 1

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
                            if FunctionValue[i, m] < FunctionValue[j, m]:  # 如果p支配q
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


def F_sift(Population, FunctionValue, Model, dataset, percentile):
    N, D = Population.shape
    keep_num = int(N*(1-percentile/100))
    sequences = torch.from_numpy(Population).int().to(Model.device)
    sequences = dataset.idx2sentences(sequences, print_special_tokens=False)
    len_list = np.array([(len(seq)+1)/2 for seq in sequences])
    target_lower_len = 10
    current_len_threshold = np.sort(len_list)[::-1][keep_num]
    if current_len_threshold > target_lower_len:
        current_len_threshold = target_lower_len
    keep_idx = (len_list >= current_len_threshold) & (len_list <= 50)
    Population = Population[keep_idx]
    FunctionValue = FunctionValue[keep_idx]
    if keep_idx.sum() > keep_num:
        sequences = [seq for idx,seq in zip(keep_idx,sequences) if idx==True]
        df = pd.DataFrame({'peptide': sequences})
        df = compute_modlamp(df)
        instability = df['instability_index'].values
        keep_idx = instability < np.sort(instability)[keep_num]
        Population = Population[keep_idx]
        FunctionValue = FunctionValue[keep_idx]
    return Population, FunctionValue


def F_reserve(FunctionValue, Population, Thresholds_used, num_reserve):
    good_individuals = FunctionValue[:,0]>-1
    for i in range(len(Thresholds_used)):
        good_individuals = good_individuals & (FunctionValue[:,i]>Thresholds_used[i])
    good_Population = Population[good_individuals, :]
    unique_rows, indexs = np.unique(good_Population, axis=0, return_index=True)
    print('num of distinct reserved individuals={}'.format(len(unique_rows)))
    if len(unique_rows) > num_reserve:
        good_values = FunctionValue[indexs]
        good_values = good_values.sum(axis=1)
        thre = np.sort(good_values)[-num_reserve]
        good_idx = FunctionValue.sum(axis=1) >= thre
        good_individuals = good_individuals & good_idx
    return good_individuals


# 主函数
def NSGAIII_searching(Model, labels, targets, Boundary, Thresholds, Generations, N, dataset):
    Thresholds_used = [target*threshold+(1-target)*(1-threshold) 
                       for target,threshold in zip(targets,Thresholds)]
    M = len(labels)
    Z, N = uniformpoint(N, M)  # 生成一致性参考点
    Z[Z == 0] = 0.000001  # 对于 wi = 0 ，我们用一个很小的数字 10^(−6) 来代替它
    Population = F_objective('init', Model, dataset, labels, targets, N)  # 测试函数选择
    # plt.ion()  # 绘制动态图
    F = None
    for Gene in range(Generations):  # 对于每一代
        MatingPool = F_mating(Population)  # 生成交配池
        Offspring = F_generator(MatingPool, N)  # 遗传算子
        Population = np.concatenate([Population, Offspring])  # 新一代种群Rt = Pt  Qt
        FunctionValue = F_objective('value', Model, dataset, labels, targets, Population)  # 目标函数评估
        Population, FunctionValue = F_sift(Population, FunctionValue, Model, dataset, percentile=10)
        reserved_idxs = F_reserve(FunctionValue, Population, Thresholds_used, 700)
        reserved_individuals = Population[reserved_idxs, :]
        unique_rows = np.unique(reserved_individuals, axis=0)
        other_idxs = (1-reserved_idxs).astype('bool')
        Population = Population[other_idxs]
        FunctionValue = FunctionValue[other_idxs]
        FrontValue, MaxFront = F_sort(deepcopy(FunctionValue), initial_population_num=N-len(unique_rows))  # 使用非支配排序对种群进行排序
        # (F1, F2,...) = 非支配性排序(Rt)
        Next = np.zeros(N - len(unique_rows))
        NoN = np.count_nonzero(FrontValue[FrontValue < MaxFront])
        Next[:NoN] = (np.ravel(FrontValue.T) < MaxFront).nonzero()[0]
        Last = (np.ravel(FrontValue.T) == MaxFront).nonzero()[0]
        c = Next[:NoN].astype('int')
        Choose = F_choose(FunctionValue[c, :], FunctionValue[Last, :], N - len(unique_rows) - NoN, Z)
        Next[NoN:N-len(unique_rows)] = Last[Choose.astype(bool)]
        Population = Population[Next.astype('int'), :]
        Population = np.concatenate([unique_rows, Population])
        F = F_objective('value', Model, dataset, labels, targets, Population)
        selected_position = (F[:,0]>-1)
        for i in range(len(Thresholds_used)):
            selected_position = selected_position&(F[:,i]>Thresholds_used[i])
        F_select = F[selected_position]
        print("Generation={},num_selected={},total_num={}".format(Gene+1,sum(selected_position),len(Population)))
        if sum(selected_position) >= len(F)*0.8:
            break
        
        # plt.cla()
        # plt.title("Generation={},num_selected={},total_num={}".format(Gene+1,sum(selected_position),len(Population)))
        # plt.scatter(F[:, 0], F[:, 1], color='#0000FF', s=5)
        # plt.scatter(F_select[:,0], F_select[:,1], color='#FF0000', s=5)
        # if Gene == Generations - 1:
        #     plt.ioff()
        #     plt.scatter(F[:, 0], F[:, 1], color='#0000FF', s=5)
        #     plt.scatter(F_select[:,0], F_select[:,1], color='#FF0000', s=5)
        #     plt.show()
        # else:
        #     plt.pause(0.1)
    
    return F, selected_position, Population
