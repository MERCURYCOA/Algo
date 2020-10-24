# 一： 股票III - leetcode123： 最多交易2次，求最大利润
import sys

class Solution:
    def max_stock_profits(self, A):
        n = len(A)
        
        f = [0]*(n+1)

        f[0] = 0
        f[1] = 0
        
        if n == 0:
            return 0

        for i in range(2,n+1):
            if A[i-2] < A[i-1]:
                f[i] = f[i-1] + A[i-1] - A[i-2]
            else:
                f[i] = 0
        #这里求出f[n]，如：0，0，0，4，0，1，2,0,1,3,5
        #现在需要求出第一最大和第二最大值，这里最大值有特殊要求：f[n]中连续上升数组如1,3,5,是因为原数组A中股票价格连续上升，每个小的连续子数组起点和终点其实就是买点和卖点。
        #所以这里的最值是求数组中极值中的最大值如4,2,5中 最大为5，第二大为4
        #思路是设local_max, global_max1, global_max2,求得global_max1, global_max2是两次交易最大利润
        return global_max1 + global_max2
            

if __name__ == '__main__':
    solution = Solution()
    print(solution.max_stock_profits([2,1,2,1,2,3,0]))

# 二：岛屿数量II - 并查集 
# 三：word search II - Trie数+矩阵DFS
# 四：滑动窗口的中位数 
    # 方法： hash + heap  O(nlogk). 注意如果直接用heapq，remove的时间是O（k）, 总体时间就是O(nk),题目要求O(nlogk).只能用heap+hash,remove时间能达到logk
    # 核心： 窗口的滑动其实就是remove, add两步操作
# 五：building outlines  hash heap 
# 不能用heapq, 因为remove O(n)太慢
# 六： 子数组求和II 
# presum + 二分法找sum落在某区间的子数组数
# tricks: 求 1 <= presum[j+1] - presum[i] <= 3  转化为  presum[j+1]-3 <= presum[i] <= presum[j+1]-1
# 转化为： 遍历j，得到一系列区间，假设有n个区间[l, r]， 查看在presum数组中，假设有x个元素presum[i]落在区间[l,r]中，把n个区间中符合条件的元素个数累加，就是最后结果
# 那么怎么求一个数组有多少元素落在指定区间呢？ 分拆区间[l, r]，求第一个小于等于l的个数a， 求第一个小于等于r的个数b, （b - a）就是落在区间内的个数。注意这里有一个tricky的地方，（b-a）并不准确
# 如果第一个小于等于r的数正好就是r， 那b-a就少算了一个数，所以，这里应该求第一个小于等于r+1的的个数c， （c-a）是准确的。
# 那么可不可以求第一个严格小于r的数呢？
# 注意这个题的几个转化

class Solution:
    """
    @param A: An integer array
    @param start: An integer
    @param end: An integer
    @return: the number of possible answer
    """
    def subarraySumII(self, A, start, end):  # [1, 2, 3, 4] , 1, 3
        # write your code here
        n = len(A)
        presum = [0] * (n+1)               # presum: [0, 1, 3, 6, 10]
        for i in range(1, n+1):
            presum[i] = presum[i-1] + A[i-1]          
        
        cnt = 0
        for i in range(1, n+1):
            l = presum[i] - end
            r = presum[i] - start
            cnt += self.find(presum, r+1) - self.find(presum, l)
        return cnt
    
    def find(self, presum, target): #求在presum中，第一个小于等于target的数
        m = len(presum)
        if presum[m-1] < target:
            return m
        
        start, end = 0, m - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if target <= presum[mid]:
                end = mid 
            else:
                start = mid
        
        if presum[end] < target:
            return end + 1
        if presum[start] < target:
            return start + 1
        return 0
solution = Solution()
print(solution.subarraySumII([1, 2, 3, 4], 1, 3))
