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
  
