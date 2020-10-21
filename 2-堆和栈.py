# 题一：接雨水II
# tricks: 能不能存下水取决于边的最低点 - heaq：总是找到当前最低点
# 矩阵中点向4个方向搜索 - 模版记住
# 
import heapq
class Solution:
    """
    @param heights: a matrix of integers
    @return: an integer
    """
    def trapRainWater(self, heights):
        if not heights or not heights[0]:
            return 0
        heap = []
        res = 0
        n = len(heights)
        m = len(heights[0])
        visited = [[0 for i in range(m)] for j in range(n)] 
        for i in range(n):
            for j in range(m):
                if i == 0 or i == n-1 or j == 0 or j == m-1:
                    heapq.heappush(heap, (heights[i][j], i, j))
                    visited[i][j] = 1 
                    
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        while heap:
            val, x, y = heapq.heappop(heap)  # 找到当前围栏鹅最低点
            for i in range(4):
                x_ = x + dx[i]
                y_ = y + dy[i]
                if x_ >= 0 and x_ < n and y_ >= 0 and y_ < m and visited[x_][y_] == 0:
                    h = max(val, heights[x_][y_])
                    res += (h - heights[x_][y_])
                    heapq.heappush(heap, (h, x_,y_)) # 注意 (x_, y_)这个位置放入堆中的高度值不一定是height[x_][y_],应该是比较其相邻外围也就是(x,y)处的高度值和（x_,y_）处的高度值，
                                                     # 谁大取谁， 因为只要水不漏出去就可以。例如height[2][0] = 12,是第二行最左边的柱子，查看到它右边相邻的柱子时，也就是height[2][1] = 10，
                                                     # （2，1）这个位置要放到heap里的应该是（12，2，1）而不是（10，2，1），因为这里相当于已经把水注入到12了，那它下一个相邻的位置的水可以到达的高度就是12（不考虑其他方向的话）     
                                                     
                    visited[x_][y_] = 1 
        return res
