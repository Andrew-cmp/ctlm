from scipy.stats import kendalltau
from scipy.stats import weightedtau
from count_inversions import count_inversions
import inspect, re
import numpy as np
# 示例数据
x = [2, 1, 3, 4, 5, 6]
y = [1, 2, 3, 4, 5, 6]
z = [1, 2, 3, 4, 6, 5]

k = [1, 2, 4, 3, 5, 6]

# print(end='    ')
# for i in vec_name:
#     print(i,end='   ')
# print()
    
# for index,i in enumerate(vec):
#     print(vec_name[index],end="  ")
#     for j in vec:
#         tau, p_value = kendalltau(i, j)
#         print("%.1f"%tau,end=' ')
#     print()
        
# 计算 Kendall Tau 及对应的 p-value

x_scores = (np.max(x) + 1) - x  
y_scores = (np.max(y) + 1) - y  
z_scores = (np.max(z) + 1) - z 
k_scores = (np.max(k) + 1) - k
x_minus_one =  [i-1 for i in x]
y_minus_one =  [i-1 for i in y]
z_minus_one =  [i-1 for i in z]
k_minus_one =  [i-1 for i in k]

print(f"{y} and {x} weightedtau is {weightedtau(x_scores,y_scores)}")
print(f"{y} and {z} weightedtau is {weightedtau(z_scores,y_scores)}")
#这两种是等效的，看gpt里计算过程的回答
print(f"{y} and {x} weightedtau rank=false is {weightedtau(x_minus_one,y_minus_one,rank=False)}")
print(f"{y} and {z} weightedtau rank=false is {weightedtau(z_minus_one,y_minus_one,rank=False)}")
## y应与z更相似，而不是与x更相似，符合预期
print(f"{k} and {z} weightedtau is {weightedtau(z_scores,k_scores)}")
print(f"{k} and {x} weightedtau is {weightedtau(x_scores,k_scores)}")
print(f"{k} and {z} weightedtau rank=false is {weightedtau(z_minus_one,k_minus_one,rank=False)}")
print(f"{k} and {x} weightedtau rank=false is {weightedtau(x_minus_one,k_minus_one,rank=False)}")
## k应该和z更相似，而不是和x更相似，符合预期
y = [1, 2, 3, 4, 5]
z = [5, 4, 2, 3, 1]
y_scores = (np.max(y) + 1) - y  
z_scores = (np.max(z) + 1) - z 
tau, p_value = weightedtau(z_scores,y_scores)
print(tau)
# print(1-(count_inversions(z)*2/10))
# print(count_inversions(z))
# print("Kendall's tau:", tau)
# print("p-value:", p_value)
