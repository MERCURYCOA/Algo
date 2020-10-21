# heap操作复杂度：push O(logn) , pop O(logn), remove O(n), 最大或最小值 O(1)
# stack 操作复杂度： pusg O(1), pop  O(1),  top  O(1)
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
# 方法一：全局变量  时间复杂度O(nlogn) - 遍历是O（n）, 每次遍历中都有堆的操作logn
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
# 题三： 滑动窗口的中位数 hash + heap  O(nlogk). 注意如果直接用heapq，remove的时间是O（k）, 总体时间就是O(nk),题目要求O(nlogk).只能用heap+hash,remove时间能达到logk
# 核心： 窗口的滑动其实就是remove, add两步操作
# 方法一：heap + hash
# 方法二： bisect模块
# 思路是：维护一个window,长度是k，被排序过。找到从nums找到该删除和该加入的数（其实就是中间隔着k-1个数，用当前下标i-k， 就是该删除的数），再通过bisect找到其在window中的位置。
import bisect
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer
    @return: The median of the element inside the window at each moving
    """
    def medianSlidingWindow(self, nums, k):
        if not nums:
            return [] 
        if k <= 0:
            return []
        if k > len(nums):
            return []
        n = len(nums)
        mid = (k-1)//2
        window = nums[:k]
        window.sort()  # 注意这里sort的是window不是nums
        res = [window[mid]]
        for i in range(k, n):
            out = nums[i-k]  # 正因为sort的是window 不是nums， 这里需要在nums里找到下一个删除的数  
            inn = nums[i]  # 下一个放进去的数
            window.pop(bisect.bisect_left(window, out))  # 上面找到需要加入和需要删除的数之后，再找到它们在window里是哪个位置（因为window被排序之后跟nums顺序不同）， 然后pop该删除的数，insert该加入的数
            window.insert(bisect.bisect_left(window, inn), inn)
            res.append(window[mid])
        return res
