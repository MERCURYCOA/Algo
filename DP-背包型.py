# 背包问题中，dp数组的大小跟背包总承重有关
# 题1， 题2中，A数组中元素的顺序是重要的，问题转化是：An放不放进背包
# 题3中， A数组中元素的顺序已经不重要了， 因为你可以重复使用元素，这是要考虑的是最后一个放进背包的元素是谁（硬币组合问题）

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
      
 # 背包问题 II
# 背包问题 III
# 原始方法（超时）- 时间优化 - 空间优化
# 方法一：f[i][w] = max{f[i-1][w], f[i-1][w-Ai-1] + Vi-1, f[i-1][w-2Ai-1] + 2Vi-1,...}
class Solution:
    """
    @param A: an integer array
    @param V: an integer array
    @param m: An integer
    @return: an array
    """
    def backPackIII(self, A, V, m):
        # write your code here
        # f[i]f[j] = f[i-1][j]
        n = len(A)
        f = [[0 for _ in range(m+1)] for _ in range(n+1)]
        for x in range(1, m+1):
            f[0][x] = -1
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                f[i][j] = f[i-1][j]
                k = 1
                while j >= k*A[i-1]:
                    if f[i-1][j-k*A[i-1]] != -1:
                        f[i][j] = max(f[i][j], f[i-1][j - k * A[i-1]] + k * V[i-1])
                    
                    k += 1   # 注意k不可以在if里面，不然的话k无法向后循环，后面可能还有k符合条件，但是无法到达那个k
        return max(f[n])
    
    
# 方法二：f[i][w] = max{f[i-1][w], f[i][w-Ai-1] + Vi-1}
# 因为这一坨 f[i-1][w-Ai-1] + Vi-1, f[i-1][w-2Ai-1] + 2Vi-1,...已经在前面算过了，就是求f[i][w-Ai-1] + Vi-1的时候， 
# f[i][w-Ai-1] + Vi-1在f[i][w] 同一行靠前面，中间差着Ai-1， 所以为了避免重复计算，时间优化，所以用f[i][w-Ai-1] + Vi-1 取代那一坨
class Solution:
    """
    @param A: an integer array
    @param V: an integer array
    @param m: An integer
    @return: an array
    """
    def backPackIII(self, A, V, m):
        # write your code here
        # f[i]f[j] = f[i-1][j]
        n = len(A)
        f = [[0 for _ in range(m+1)] for _ in range(n+1)]
        for x in range(1, m+1):
            f[0][x] = -1
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                f[i][j] = f[i-1][j]
                if j >= A[i-1] and f[i][j-A[i-1]] != -1:
                    f[i][j] = max(f[i][j], f[i][j - A[i-1]] + V[i-1])
                    
        
        return max(f[n])
# 方法三：        
# 空间优化：一维数组
class Solution:
    """
    @param A: an integer array
    @param V: an integer array
    @param m: An integer
    @return: an array
    """
    def backPackIII(self, A, V, m):
        # write your code here
        # f[i]f[j] = f[i-1][j]
        n = len(A)
        f = [0 for _ in range(m+1)]
        
        
        for i in range(1, n+1):
            for j in range(A[i-1],m+1):
                f[j] = max(f[j-A[i-1]] + V[i-1], f[j])
        return f[m]
    
# 背包问题 IV

# 题2 背包问题 V
# 解法1: 2维滚动数组
class Solution:
    """
    @param nums: an integer array and all positive numbers
    @param target: An integer
    @return: An integer
    """
    def backPackV(self, nums, target):
        n = len(nums)
        if n == 0:
            return 0 
            
        dp = [[0 for _ in range(target+1)] for _ in range(2)]
        dp[0][0] = 1 
        
            
        for i in range(1, n+1):
            for j in range(target+1):
                dp[i%2][j] = dp[(i-1)%2][j]
                if j >= nums[i-1]:
                    dp[i%2][j] = dp[i%2][j] + dp[(i-1)%2][j - nums[i-1]]
                    
        return dp[n%2][target]
    
# 解法2: 1维滚动数组 + 从后向前 【因为需要用到前面的旧值，如果从前向后更新，新值会覆盖需要用到的旧值】
class Solution:
    """
    @param nums: an integer array and all positive numbers
    @param target: An integer
    @return: An integer
    """
    def backPackV(self, nums, target):
        n = len(nums)
        if n == 0:
            return 0 
            
        dp = [0 for _ in range(target+1)]
        dp[0] = 1 
        
            
        for i in range(1, n+1):
            for j in range(target, -1, -1):
                if j >= nums[i-1]:
                    dp[j] += dp[j - nums[i-1]]
                    
        return dp[target]
# 题3  背包问题 VI 
# 可重复使用数字 且 顺序不同算作不同组合 （变成硬币组合问题）
class Solution:
    """
    @param nums: an integer array and all positive numbers, no duplicates
    @param target: An integer
    @return: An integer
    """
    def backPackVI(self, nums, target):
        n = len(nums)
        if n == 0:
            return 0 
        f = [0] * (target+1)    
        f[0] = 1 
        for i in range(1, target+1):
            for j in range(n):
                if i >= nums[j]:
                    f[i] += f[i - nums[j]]
                    
        return f[target]
