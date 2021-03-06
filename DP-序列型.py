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
        f[0][0] = f[0][1] = f[0][2] = 0    # 一定要明白为什么开2维数组，不是因为C是2维的，而是要记录颜色，如果对颜色没限制，用1维数组就可以记录当前最小cost。
        
        for i in range(1,n+1):
            for j in range(0,3):
                f[i][j] = sys.maxsize # 非常重要， 必须先让f[i][i]=无穷，后面才能用min 迭代
                for k in range(0,3):   # 用k 表示第i-1个房子的颜色
                    if k != j:
                        f[i][j] = min(f[i][j], (f[i-1][k] + C[i-1][j]))  # 注意i从1开始，但是C从[0][0]开始，所以 C[i-1][j]表示从1开始的第i个房子
        return min(f[n][0], f[n][1], f[n][2])
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.paint_house_min_cost([[14,2,11],[11,14,5],[14,3,10]]))
    
# 解法2: 滚动数组：
class Solution:
    """
    @param costs: n x 3 cost matrix
    @return: An integer, the minimum cost to paint all houses
    """
    def minCost(self, costs):
        n = len(costs)
        if n == 0:
            return 0
            
        INF = 0x7fffffff   # 记住这个用法
        f = [costs[0], [INF, INF, INF]]
        
        for i in range(1, n):
            for j in range(3):
                f[i&1][j] = INF    # i&1  - 判断奇偶的常见用法， 记住！！！！！！  1&1 = 1， 0&1 = 0， i是奇数， i&1就是1， i是偶数， i&1就是0
                for k in range(3):
                    if j != k:
                        f[i&1][j] = min(f[i&1][j], f[(i+1)&1][k] + costs[i][j])
        
        return min(f[(n-1)&1])
# 题二： 与题一类似，颜色改成k个
# 方法一：时间复杂度 O(NK^2), 时间复杂度太高，无法AC
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
# 思想：求f[i][j] = min(f[i-1][k]) + C[i-1][j],其中k != j  方法一是用两个for循环求最小值，复杂度为k^2,降低复杂度的方法是先求f[i-1][j]的最小值min1和次小值min2,下标分别为id1,id2
#      查看f[i][j]中如果j！=id1,就是当前j的颜色没有跟前一行最小值撞色，那么f[i][j] = C[i-1][j] + min1，如果j = id1,说明当前j跟前一行最小值撞色，那么当前f[i][j]就染成次小值的颜色，
#      即 方= C[i-1][j] + min2
#      查看f[i][j]中如果j！=id1,就是当前j的颜色没有跟前一行最小值撞色，那么f[i][j] = C[i-1][j] + min1，如果j = id1,说明当前j跟前一行最小值撞色，那么当前f[i][j]就染成次小值的颜色，
#      即f[i][j] = C[i-1][j] + min2
# 技巧：找到挖去一个值后的最小值的方法： 假设最小值为
import sys

class Solution:
    def paint_house_min_cost(self, C):
        n = len(C)
        m = len(C[0])
        f = [[0]*m for _ in range(n+1)]
        min1 = min2 = 0
        id1 = id2 = 0
        for i in range(1,n+1):
            min1 = min2 = sys.maxsize
            for j in range(0,m):
                if f[i-1][j] < min1:
                    min2 = min1
                    id2 = id1
                    min1 = f[i-1][j]
                    id1 = j
                else:
                    if f[i-1][j] < min2:
                        min2 = f[i-1][j]
                        id2 = j
            for j in range(0,m):
                f[i][j] = C[i-1][j]
                if j != id1:
                    f[i][j] += min1
                else:
                    f[i][j] += min2
        res = f[n][0]
        for t in range(1,m):
            res = min(res,f[n][t])
        return res

if __name__ == '__main__':
    solution = Solution()
    print(solution.paint_house_min_cost([[14,2,11],[11,14,5],[14,3,10]]))
# 方法三：滚动数组：
class Solution:
    """
    @param costs: n x k cost matrix
    @return: an integer, the minimum cost to paint all houses
    """
    def minCostII(self, C):
        if not C or not C[0]:
            return 0
        n = len(C)
        m = len(C[0])
        f = [[0]*m for _ in range(2)]
        min1 = min2 = 0
        id1 = id2 = 0
        for i in range(1,n+1):
            min1 = min2 = sys.maxsize
            for j in range(0,m):
                if f[(i-1)&1][j] < min1:
                    min2 = min1
                    id2 = id1
                    min1 = f[(i-1)&1][j]
                    id1 = j
                else:
                    if f[(i-1)&1][j] < min2:
                        min2 = f[(i-1)&1][j]
                        id2 = j
            for j in range(0,m):
                f[i&1][j] = C[i-1][j]
                if j != id1:
                    f[i&1][j] += min1
                else:
                    f[i&1][j] += min2
        res = f[n&1][0]
        for t in range(1,m):
            res = min(res,f[n&1][t])
        return res

 # 题三： 小偷偷金币，N栋房子， 每个房子有A[i]个金币，不能同时偷相邻房子
 # 方法一：
    class Solution:
    """
    @param A: An array of non-negative integers
    @return: The maximum amount of money you can rob tonight
    """
    def houseRobber(self, A):
        if not A:
            return 0 
        n = len(A)    
        f = [0]*(n + 1)
        f[1] = A[0]
        for i in range(2, n+1):
            f[i] = max(f[i-1], f[i-2]+A[i-1])
        return max(f[n], f[n-1])
   
# 方法二：
# 滚动数组的思想 - 这里用长度为2的数组，或者直接设两个值滚动向前

class Solution:
    def max_house_stealing(self, A):
        n = len(A)
        if n == 0:
            return 0
        new = A[0]
        old = 0
        for i in range(1,n):
            t = max(old + A[i], new)    # 注意： old初始化为0，old的初始角色类似f[0] = 0， 不可以直接 old = new, new = max(old + A[i], new), 这样的话 old 就是A[0]了，就错了
            old = new
            new = t
        return new

if __name__ == '__main__':
    solution = Solution()
    print(solution.max_house_stealing([1,2]))

# 题四：与题三类似，房子从一排变成一个圈，相邻不能同时偷，最多偷多少？
#技巧点： 拆分问题，会成排的，就把圈圈破成排
# 关键：破开圈圈，变成排 - 考虑两种情况 （一）有A[0]和没有A[0]  或者 （二）有A[n-1]和没有A[n-1] ，只需要考虑一种就可以，因为 有A[0] = 没有A[n-1], 没有A[0]=有A[n-1]
# 方法： 令开写一个函数，在其中调用2次max_house_stealing(),第1次参数为A[0:n-1] - 不考虑A[n-1], 第2次参数为A[1: n] - 不考虑A[0]
# 对两种情况求最大值就是圈圈情况的最大值

class Solution:
    """
    @param nums: An array of non-negative integers.
    @return: The maximum amount of money you can rob tonight
    """
    def houseRobber2(self, nums):
        if not nums:
            return 0 
            
        n = len(nums)
        if n <= 2:
            return max(nums)
        res = max(self.helper(nums[0:n-1]), self.helper(nums[1:n]))
        return res
        
    def helper(self, A):
        n = len(A)  
        f = [0]*(n + 1)
        f[1] = A[0]
        for i in range(2, n+1):
            f[i] = max(f[i-1], f[i-2]+A[i-1])
        return max(f[n], f[n-1])

# 题五： 股票买卖，给出连续n天股票价格，只能买一股卖一股，不能多次交易，最大利润？

# 不涉及储存每个f[n]，不用开数组，只维护一个最小值
class Solution:
    def max_stock_profits(self, A):
        n = len(A)
        if n == 0:
            return 0
        minV = A[0]
        res = 0
        for i in range(0,n):
            res = max(res, A[i] - minV)
            minV = min(minV, A[i])
        return res
            

if __name__ == '__main__':
    solution = Solution()
    print(solution.max_stock_profits([3,2,1]))

# 题六：可以多次交易，最大利润？

# 这里需要储存多次交易的利润，所以需要开数组
class Solution:
    def max_stock_profits(self, A):
        n = len(A)
        if n == 0:
            return 0
        f = [0]*(n+1)
        for i in range(2,n+1):
            if A[i-2] < A[i-1]:
                f[i] = A[i-1] - A[i-2] + f[i-1]
            else:
                f[i] = f[i-1]
        return f[n]
            

if __name__ == '__main__':
    solution = Solution()
    print(solution.max_stock_profits([2，1，2，1，2，3，0]))

    
# 题七：最多交易k次，最大利润
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        if not prices or len(prices) <= 1:
            return 0
        
        m, k = len(prices), 2
        res = [[0 for _ in range(2 * k + 1)] for _ in range(m + 1)]
        # res[0][0] = 0
        res[0][1], res[0][2], res[0][3], res[0][4] = -sys.maxsize, -sys.maxsize, -sys.maxsize, -sys.maxsize
        
        for i in range(1, m + 1):
            # stage 0 alreday initialized to 0 no need for extra set
            
            for j in range(1, 2 * k + 1):
                if j % 2:
                    # stage 1 and stage 3
                    res[i][j] = max(res[i - 1][j - 1], res[i - 1][j] + prices[i - 1] - prices[i - 2])
                    # 状态1， 3 -- 持有股票
                    # 1， 天改变 状态改变，今天买入，利润与前一天前一状态相同   2，天改变，状态不变（昨天就持有股票） + 持续获利
                else:
                    res[i][j] = max(res[i - 1][j], res[i - 1][j - 1] + prices[i - 1] - prices[i - 2])
                    # 状态2， 4  -- 不持有股票
                    # 1，天改变，状态不变，昨天就没有股票，利润与前一天相同  2，天改变，状态改变，昨天持有股票，今天卖掉，利润增加
        
        #return max(res[m][0], res[m][2], res[m][4])
       
        return max(res[m])
# 题八：最大上升子序列
# 方法一：O(n^2)
class Solution:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """
    def longestIncreasingSubsequence(self, nums):
        if nums is None or not nums:
            return 0
            
        # state: dp[i] 表示以第 i 个数结尾的 LIS 的长度
        # initialization: dp[0..n-1] = 1
        dp = [1] * len(nums)
        
        # function: dp[i] = max(dp[j] + 1), j < i && nums[j] < nums[i]
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        # answer, 任意一个位置都可能是 LIS 的结尾  
        return max(dp)
# 方法二：O(nlogn)
# 没看懂
