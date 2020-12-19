# 背包问题中，dp数组的大小跟背包总承重有关
# 题1 背包问题
# 记录前i个物品能拼出哪些重量：1 等于前i-1个物品能拼出的重量  2 前i-1个物品能拼出的重量 + 第i个物品重量
class Solution:
    """
    @param m: An integer m denotes the size of a backpack
    @param A: Given n items with size A[i]
    @return: The maximum size
    """
    def backPack(self, m, A):
        n = len(A)
        if n == 0:
            return 0 
            
        dp = [[None for _ in range(m+1) ] for _ in range(n+1)]   # 注意  m+1   n+1
        dp[0][0] = True
        
        for i in range(1, m+1):
            dp[0][i] = False
        
        for i in range(1, n+1):
            for j in range(0, m+1):   # 从0开始
                dp[i][j] = dp[i-1][j] 
                if j >= A[i-1]:       # 一定要记住 j >= A[i-1], 不然j-A[i-1]就越界了。其实就是想拼出10kg, 如果A[i-1]=11kg, 那么A[i-1]就不能入选了。
                    dp[i][j] = dp[i][j] or dp[i-1][j - A[i-1]]
        res = 0 
        for i in range(m, -1, -1):
            if dp[n][i] == True:
                res = i
                break
        return res
      
 
