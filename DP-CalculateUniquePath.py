# |   |   |   |
# |   |   |   |
# |   |   |   |
# 机器人从网格左上角到右下角，有多少种unique的方式？一次只能向右或向下移动。
# 动态规划 - 计数型问题




class Solution:
    def total_unique_paths(self, n,m):
        f = [[None] * (m+1) for _ in range(n+1)]  #创建空二维数组
        # initialization
       
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                if i == 1 or j == 1:
                    f[i][j] = 1
                else:
                    f[i][j] = f[i-1][j] + f[i][j-1]
        return f[i][j]
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.total_unique_paths(3,3))
