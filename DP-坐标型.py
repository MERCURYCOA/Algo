# |   |   |   |
# |   |   |   |
# |   |   |   |
# 动态规划 - 坐标型
# （也是计数型）
# 坐标型解法：对于输入数组A, 创建一个与其大小相同的F数组

# 题一：
# 机器人从网格左上角到右下角，有多少种unique的方式？一次只能向右或向下移动。

class Solution:
    def total_unique_paths(self, A):
        n = len(A)
        m = len(A[0])
        f = [[None] * (m+1) for _ in range(n+1)]  #创建空二维数组
        # 合规判断
        if n < 1 or m < 1:
           return -1
    
        for i in range(1, n+1):
            for j in range(1, m+1):
                if i == 1 or j == 1:     # initialization
                    f[i][j] = 1
                else:
                    f[i][j] = f[i-1][j] + f[i][j-1]
        return f[i][j]
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.total_unique_paths(3,3))
# 题二：
# 与题一类似，只是格子中有障碍， 令有障碍的格子表示为 f[i][j]=1， 坐标（0，0）和（n,m）有障碍直接返回0，因为永远无法到达

class Solution:
    def total_unique_paths(self, A):
        n = len(A)
        m = len(A[0])
        f = [[0] * m for _ in range(n)]  #创建空二维数组
        # 合规判断
        if n < 1 or m < 1:
           return -1
    
        for i in range(0, n):
            for j in range(0, m):
                # 不能用if else, 应该用多个if, 因为这些情况是分类，不是互斥的
                # 可能同时满足i>0, j>0, 这个时候如果不想执行剩下的语句，用continue
                if A[i][j] == 1:
                    f[i][j] = 0
                    continue
                if i == 0 and j ==0:
                    f[i][j] = 1
                    continue
                if i>0:
                    f[i][j] = f[i][j] + f[i-1][j]
                if j >0:
                    f[i][j] = f[i][j] + f[i][j-1]
        return f[n-1][m-1]
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.total_unique_paths([[0,0,1,0],[0,1,0,0],[0,0,0,0]]))
