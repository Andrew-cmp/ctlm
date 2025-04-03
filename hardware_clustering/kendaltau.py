from scipy.stats import kendalltau
import inspect, re
# 示例数据
x = [5, 4, 3, 2, 1]
y = [1, 2, 3, 4, 5]
z = [4, 5, 2, 3, 1]
vec = [x,y,z]
vec_name=["x","y",'z']

def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)
spam = 42
print(varname(spam))

print(end='    ')
for i in vec_name:
    print(i,end='   ')
print()
    
for index,i in enumerate(vec):
    print(vec_name[index],end="  ")
    for j in vec:
        tau, p_value = kendalltau(i, j)
        print("%.1f"%tau,end=' ')
    print()
        
# 计算 Kendall Tau 及对应的 p-value
tau, p_value = kendalltau(y, x)

print("Kendall's tau:", tau)
print("p-value:", p_value)
