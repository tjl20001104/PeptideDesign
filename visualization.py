import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sample_pipeline import compute_modlamp
import cfg

# def translate_z(z):
#     z_str = z[1:-1].split(',')
#     res = []
#     for ele in z_str:
#         if '-' in ele:
#             ele = ele[1:]
#             ele = -float(ele)
#         else:
#             ele = float(ele)
#         res.append(ele)
#     res = np.array(res)
#     return res

# df = pd.read_csv('./raw_samples.csv', index_col=0)
# # zs = np.stack([translate_z(z) for z in df['z']])
# # instability = df['instability_index'].values
# tox = df['tox'].values
# amp = df['amp'].values
# data = [tox, amp]
# positions = [1,2]

# fig,ax=plt.subplots()
# ax.violinplot(data,positions,showmeans=True,showmedians=True)
# plt.show()

# def read_data(path):
#     df = pd.read_csv(path)
#     peptides = df.iloc[0:4000]['text']
#     df = pd.DataFrame({'peptide': peptides})
#     df = df[~df['peptide'].isin([''])]
#     df = compute_modlamp(df)
#     return df

# df_all = read_data('data/all_amp.csv')
# df_trained = pd.read_csv('z_result/trained_samples.csv')
# df_untrained = pd.read_csv('z_result/untrained_samples.csv')

# labels = ['instability_index', 'aliphatic_index', 'hydrophobic_ratio', 'H', 'uH', 'charge','isoelectric_point']

# fig, ax = plt.subplots(3,2)
# i = 1
# for label in labels:
#     data_all = df_all[label].values
#     data_trained = df_trained[label].values
#     data_untrained = df_untrained[label].values
#     data = [data_all,data_trained,data_untrained]
#     positions = [1,2,3]
#     plt.subplot(3,2,i)
#     # plt.boxplot(data,positions,sym='')
#     plt.violinplot(data,positions,showmeans=True,showextrema=False,showmedians=True)
#     plt.title(label)
#     i = i + 1
# plt.show()

# plt.figure()
# i = 2
# label = labels[i]
# data_all = df_all[label].values
# data_trained = df_trained[label].values
# data_untrained = df_untrained[label].values
# data = [data_all,data_trained,data_untrained]
# positions = [0.5,1.25,2]
# # plt.boxplot(data,positions,sym='')
# part = plt.violinplot(data,positions,showmeans=True,showextrema=False,showmedians=True)
# plt.axis([0, 3.5, -0.1, 1.1])
# colors = ['c','pink','greenyellow']
# plt.title(label)
# for j in range(len(part["bodies"])):
#     pc = part["bodies"][j]
#     pc.set_facecolor(colors[j])
#     pc.set_edgecolor("black")
#     pc.set_alpha(1)
#     pc.set_linestyle("--")
# plt.legend(labels=['training\nset','trained\ngeneration','untrained\ngeneration'],loc='right', fontsize=10, ncol=1,
#            labelcolor=colors)
# plt.show()

# label = 'tox'
# classes=[label,'non-{}'.format(label)]
 
# classNamber=2 #类别数量
 
# # 混淆矩阵
# confusion_matrix = np.array([
#     (284,1452),
#     (26,1794)
#     ],dtype=np.float64)
 
# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
# plt.title('confusion_matrix-{}\nthrehold=0.02'.format(label))#改图名
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=-45)
# plt.yticks(tick_marks, classes)
 
# thresh = confusion_matrix.max() / 2.
# #iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# #ij配对，遍历矩阵迭代器
# iters = np.reshape([[[i,j] for j in range(classNamber)] for i in range(classNamber)],(confusion_matrix.size,2))
# for i, j in iters:
#     plt.text(j, i, format(confusion_matrix[i, j]),va='center',ha='center')   #显示对应的数字
 
# plt.ylabel('Groud Truth')
# plt.xlabel('Prediction')
# plt.tight_layout()
# plt.show()


df = pd.read_csv('optim/selected_samples.csv')
df = df[df['peptide'].str.len()>=9]
df = df.iloc[:,1:]
df = df.drop_duplicates()
df.to_csv('selected.csv')