# 模版一：窗口类指针， j的循环移动，要把条件写到while后面
# 本质是对两层for循环的改进
for i in range(n):
  while j < n and i < j and j前进的条件：
    j += 1
    更新j状态
   
  if j停下的条件：
    return/ break 
  更新i的状态
