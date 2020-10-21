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
# 题二： 数据流中位数
# 思路：  维护maxheap，median, minheap，maxheap存小于median的数， minheap存大于等于median的数。 如果下一个进来的num>=当前median,num进入minheap， 如果num小于median, num进入maxheap
        # 之后查看两个堆的大小，因为题设说当nums有偶数个元素时，median取最中间的2个数中小的那一个，所以minheap的元素个数要么等于maxheap元素个数，要么多1个。
# 方法一：全局变量
    class Solution:
    """
    @param nums: A list of integers
    @return: the median of numbers
    """
    def medianII(self, nums):
        if not nums:
            return []
            
        self.median = nums[0]
        self.maxheap = []
        self.minheap = []
        
        medians = [nums[0]]
        for num in nums[1:]:
            self.add(num)
            medians.append(self.median)
            
        return medians

    def add(self, num):
        if num < self.median:
            heapq.heappush(self.maxheap, -num)
        else:
            heapq.heappush(self.minheap, num)
            
        # balanace
        if len(self.maxheap) > len(self.minheap):
            heapq.heappush(self.minheap, self.median)
            self.median = -heapq.heappop(self.maxheap)
        elif len(self.maxheap) + 1 < len(self.minheap):
            heapq.heappush(self.maxheap, -self.median)
            self.median = heapq.heappop(self.minheap)
# 方法二：不用全局变量，add函数每次返回调整过的minheap, maxheap, median    
import heapq
class Solution:
    """
    @param nums: A list of integers
    @return: the median of numbers
    """
    def medianII(self, nums):
        if not nums:
            return []
            
        medians = [nums[0]]
        maxheap = []
        minheap = []
        median = nums[0]
        for num in nums[1:]:
            maxheap, minheap, median = self.add(num, maxheap, minheap, median)
            medians.append(median)
        return medians

    def add(self, num, maxheap, minheap, median):
        if num >= median:
            heapq.heappush(minheap, num)
        else:
            heapq.heappush(maxheap, -num)
            
        if len(maxheap) > len(minheap):
            heapq.heappush(minheap, median)
            median = -heapq.heappop(maxheap)
        elif len(maxheap) +1 < len(minheap):
            heapq.heappush(maxheap, -median)
            median = heapq.heappop(minheap)
        return maxheap, minheap, median
