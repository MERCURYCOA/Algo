# 注意：序列型问题是 f[i]存得失前i-1个元素的性质，所以假设A有n个元素，f需要创建n+1， 然后初始化空序列f[0] = 0，f[1]存储A[0]的性质，f[2]存储A[0]和A[1]的性质（比如：和最小）。
# 对于二维数组n行m列，根据题意，一般创建f是n+1行m列，注意这里不需要多加1列，需要多加1行.同时初始化第一行，f[0][0] = 0, f[0][1] = 0 ..., 最后返回最后一行也就是第n行中最大或最小值。

#题一
# N个房子，每个房子可以漆3种颜色，相邻房子不可以喷一样的颜色，每个房子喷不同颜色的成本不同，有二维数组C[i][j]表示，C[i][0], C[i][1], C[i][2]分别表示第i栋房子分别喷成三种颜色的成本
# 求最少成本？
# 输入： Cost数组 [14,2,11],[11,14,5],[14,3,10]]，输出： 10， 表示最低成本为10

# 重要思想： 分类讨论： 三种不同颜色， 相邻不能同色，那就在f数组中多开一维，记录颜色，  f[n][0], 表示第n栋房子喷0号颜色时，前n栋房子总喷漆成本，
#                   f[n][1],表示第n栋房子喷1号颜色时，前n栋房子总喷漆成本， f[n][2],表示第n栋房子喷2号颜色时，前n栋房子总喷漆成本。最终的最低成本为min(f[n][0], f[n][1], f[n][2])
#                   子问题部分：当第n栋房子喷0号颜色时，第n-1栋房子只能喷1,2号色，技巧：给一个k，表示n-1栋房子的颜色， 让k 跟j 比较，当k！= j时，表示第n个房子跟n-1个房子颜色不同



import sys

class Solution:
    def paint_house_min_cost(self, C):
        n = len(C)
        if n<1:
            return 0
        f=[[None]*3 for _ in range(n+1)]
        f[0][0] = f[0][1] = f[0][2] = 0    
        
        for i in range(1,n+1):
            for j in range(0,3):
                f[i][j] = sys.maxsize
                for k in range(0,3):   # 用k 表示第i-1个房子的颜色
                    if k != j:
                        f[i][j] = min(f[i][j], (f[i-1][k] + C[i-1][j]))  # 注意i从1开始，但是C从[0][0]开始，所以 C[i-1][j]表示从1开始的第i个房子
        return min(f[n][0], f[n][1], f[n][2])
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.paint_house_min_cost([[14,2,11],[11,14,5],[14,3,10]]))

# 题二： 与题一类似，颜色改成k个
# 方法一：时间复杂度 O(NK^2)
import sys

class Solution:
    def paint_house_min_cost(self, C):
        n = len(C)
        m = len(C[0])
        f = [[0]*m for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(0,m):
                f[i][j] = sys.maxsize
                for k in range(0,m):
                    if k != j:
                        f[i][j] = min(f[i][j], f[i-1][k] + C[i-1][j])
        res = f[n][0]
        for t in range(1,m):
            res = min(res,f[n][t])
        return res

if __name__ == '__main__':
    solution = Solution()
    print(solution.paint_house_min_cost([[14,2,11],[11,14,5],[14,3,10]]))
  
# 方法二：时间复杂度O(NK)

