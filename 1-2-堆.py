# 堆： 操作：O(logn) push， O（logn）remove, O(1) min or max 
# 为什么是logn? 原理：sift up / sift down    原理解释：https://blog.csdn.net/hrn1216/article/details/51465270
# 为什么是logn? Add操作是在二叉树的最后加入，成为最后一个叶子，然后向上调整，维持最大/最小堆，最坏情况是每层都调整，时间是logn. Remove操作是让树的最后一个叶子覆盖要删除的节点，
# 然后向上或向下调整树，时间也是logn
# 堆本质是完全二叉树，一般可以用数组表示，从上到下，从左到右将二叉树填满
# heapq的常见方法
# heapq.heappush(heap, item)
# heapq.heapify(list) 
# heapq.heappop(heap) 
# heapq.heapreplace(heap.item) 
# heapq.heappushpop(list, item)
# heapq.merge（…）
# heapq.nlargest(n,heap) 
# heapq.nsmallest(n,heap) 

# 用最小堆求前k大和前k小  -- 一定要分清，然后会用
# 前k大大大大！！！  # 看题三
for x in A:
    heapq.heappush(heap, x)
    if len(heap) >k:
        heapq.heappop(heap)
# 前k小：   # 看题六
for x in A:
    heapq.heappush(heap, x)
res = []
for _ in range(k):
    res.append(heapq.heappop(heap)
 # res就是heap中前k小的元素
# 如果不需要得到前k大/小，只是仅仅将最小堆变成最大堆的话，直接heappush相反数，pop相反数就可以了               
               
# 题一： 堆化
# 方法1: sift down 从中间，向下  时间O（n）-> 从第 N/2 个位置开始往下 siftdown，那么就有大约 N/4 个数在 siftdown 中最多交换 1 次，N/8 个数最多交换 2 次，N/16 个数最多交换 3 次。 所以O(1* n/4 + 2* n/8 + 3*n/16 + ...+ 1*logN) = O(N)
# 方法2: 新建数组，每次加入1个元素，从后向前找father，时间O(nlogn)
#1
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
       for i in range(len(A)//2, -1, -1):  # O(n)
           self.siftdown(A, i)
           
    def siftdown(self, A, index):   # O(logn)
        n = len(A)
        while index < n:
            left = index * 2 + 1
            right = index * 2 + 2
            minIndex = index
            if left < n and A[left] < A[minIndex]:
                minIndex = left
            if right < n and A[right] < A[minIndex]:
                minIndex = right
            if minIndex == index:
                break
            A[minIndex], A[index] = A[index], A[minIndex]
            index = minIndex
# 2 
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
        res = []
        n = len(A)

        for index in range(n):  # 2, 4, 5, 
            res.append(A[index])
            res = self.siftup(res)
        return res
        
    def siftup(self, res):
        if len(res) == 0 or len(res) == 1:
            return res
        last = len(res)-1
        while last > 0:
            if res[(last - 1) // 2] > res[last]:
                temp = res[(last - 1) // 2]
                res[(last - 1) // 2] = res[last]
                res[last] = temp
                
           
            last = (last - 1) // 2
        return res
               # 堆的应用 heapq
# 题二：ugly number II
# 边pop边往heap里放， for n次pop  同样做法看 题四：排序矩阵中的从小到大第k个数
import heapq
class Solution:
    """
    @param n: An integer
    @return: return a  integer as description.
    """
    def nthUglyNumber(self, n):
        # write your code here
        heap = [1]
        visited = set([1])
        for i in range(n):
            val = heapq.heappop(heap)
            for factor in [2,3,5]:
                if val*factor not in visited:
                    visited.add(val*factor)
                    heapq.heappush(heap, val*factor)
                    
        return val       
# 题三： 前K大数II

import heapq
class Solution:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self.heap = []
        self.k = k 
    """
    @param: num: Number to be added
    @return: nothing
    """
    def add(self, num):
        # write your code here
        heapq.heappush(self.heap, num)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
    """
    @return: Top k element
    """
    def topk(self):
        # sorted: 最小堆只是孩子比父亲节点小，不一定是从小到大排好的
        # reverse:heapq是最小堆，所以需要reverse
        return sorted(self.heap, reverse=True)
# 第2次做法： 错误做法
# 错误的原因是：没明白堆的排序不稳定性，最大堆/最小堆只维持最大值/最小值， 并不能对后面的所有元素排序
# 你的想法：将相反数存入堆，变最小堆为最大堆，长度超过k，将数组最后一个元素弹出。错误：你认为最后一个元素是这个最小堆的最大值，也就是原数组最小值，将其pop出，数组就留下了当前前k大。但是，最后一个元素并不一定是最小堆的最大值，因为最小堆只维护最小值，不对后面的元素排序，那么你pop出的就不一定是最大值。
import heapq
class Solution:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        self.heap = []
        self.k = k 
        
    """
    @param: num: Number to be added
    @return: nothing
    """
    def add(self, num):
        heapq.heappush(self.heap, 0-num)
        if len(self.heap) > self.k:
            self.heap.pop()
    """
    @return: Top k element
    """
    def topk(self):
        res = []
        for num in self.heap:
            res.append(0 - num)
        return sorted(res, reverse=True)
    
# 题四：排序矩阵中的从小到大第k个数 要求：klogn
# for k个循环 * 最多n的heappush (logn) = klogn
# 边判断边往heap里放  - 边走边放               
from heapq import heappush, heappop
class Solution:
    """
    @param matrix: a matrix of integers
    @param k: An integer
    @return: the kth smallest number in the matrix
    """
    def kthSmallest(self, matrix, k):
        if not matrix or not matrix[0]:
            return None 
        heap = [(matrix[0][0], 0, 0)]
        visited = set()
        visited.add((0,0))
        n, m = len(matrix), len(matrix[0])
        res = None       
        for _ in range(k):
            res, x, y = heappop(heap)
            if x+1 < n and (x+1, y) not in visited:
                heappush(heap, (matrix[x+1][y], x+1, y))
                visited.add((x+1, y))
            if y+1 < m and (x, y+1) not in visited:
                heappush(heap, (matrix[x][y+1], x, y+1))
                visited.add((x, y+1))
        return res
# 题五：N数组第K大元素 
# 关键：数组不是矩阵，每个数组长度不一， 
# 重新进行数组反排序
from heapq import heappush, heappop
class Solution:
    """
    @param s: A string
    @param k: An integer
    @return: An integer
    """
    def kthLargest(self, A, k):
        if not A:
            return None 
        sorted_A = []
        heap = []
        num = None
        for i, a in enumerate(A):
            if not a:
                continue 
            a.sort(reverse = True)
            heappush(heap, (-a[0], i, 0))
            sorted_A.append(a)
            
        for _ in range(k):
            num, x, y = heappop(heap)
            if y + 1 < len(sorted_A[x]) :
                heappush(heap, (-sorted_A[x][y+1], x, y+1))
        return -num

solution = Solution()
print(solution.kthLargest([[9,3,2,4,7],[1,2,3,4,8]], 3))
               
# 题四：合并K个排序链表
# 方法一 heapq  O(Nlogk)
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
import heapq

class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        if not lists:
            return None
        heap, count = [], 0
        dummy = ListNode(0)
        tail = dummy
        for head in lists:
            if head:
                heapq.heappush(heap, (head.val, count, head))  #couunt的作用是防止head.val相等时，向后找head比较，而head是不能比较的，所以需要中间有一个不会重复的数字可以在head,val相等时用来参考比较，这样就不会找到head了。
                count += 1
        while heap:
            value, count, node = heapq.heappop(heap)
            tail.next = node 
            tail = node 
            if node.next:
                heapq.heappush(heap, (node.next.val, count, node.next))
                count += 1
        return dummy.next
    
# 第2次做 ：
# 在将node放到heapq的时候，要判断存在与否
import heapq
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        head = ListNode(0)
        dummy = head
        heap = []
        count = 0
        for node in lists:
            if node:
                heapq.heappush(heap, (node.val, count, node))
                count += 1
        while heap:
            value, _, node = heapq.heappop(heap)
            dummy.next = node 
            dummy = dummy.next 
            if node.next:
                heapq.heappush(heap, (node.next.val, count, node.next))
                count += 1
            
        return head.next
# 方法二： 分治法  O(nlogk)
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]  # 注意，这里不能返回lists,而应该返回lists的第一个元素，因为lists是数组[],它的第一个元素才是node
        k = len(lists)
     
        mid = len(lists)//2
        sorted_left = self.mergeKLists(lists[:mid])  # 注意数组的边界，[:mid]不包括mid， [mid:]包括mid，数组包前不包后
        sorted_right = self.mergeKLists(lists[mid:])
        return self.merge(sorted_left, sorted_right)  #这里必须return
    
    def merge(self, left, right):
        if not left:
            return right
            
        if not right:
            return left 
        dummy = ListNode(-1)
        tail = dummy
        while left and right:
            if left.val < right.val:
                tail.next = left
                left = left.next
                tail = tail.next
            else:
                tail.next = right 
                right = right.next
                tail = tail.next
        if left:
            tail.next = left 
            
        if right:
            tail.next = right
        return dummy.next
    
# 方法三： 两两迭代  O(nlogk)
# 注意，不可行的方法： 1，2合并，再跟3合并，再跟4合并...  - 这样会超时
# 正确的迭代方法是：两两合并，成新的lists，再对新的lists，两两合并，一直到最后合并成1个链表
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        # write your code here
        if not lists:
            return None 
        while len(lists) > 1:
            next_lists = [] #两个合并， 加到新建的list
            for i in range(0,len(lists), 2):
                if i+1 < len(lists):
                    new = self.merge_two_lists(lists[i], lists[i+1])
                else:
                    new = lists[i]
                next_lists.append(new)
            lists = next_lists  # 让新的list= lists，再进入while循环， 知道最后合并到只剩一条链表
        return lists[0]
    
    def merge_two_lists(self, head1, head2):
        if not head1:
            return head2
        if not head2:
            return head1
        
        dummy = ListNode(-1)
        tail = dummy
        while head1 and head2:
            if head1.val < head2.val:
                tail.next = head1
                head1 = head1.next
            else:
                tail.next = head2
                head2 = head2.next
            tail = tail.next
        if head1:
            tail.next = head1 
            
        if head2:
            tail.next = head2
        return dummy.next
# 第2次做：
# 看注释， 错误的点
import heapq
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        if not lists:
            return None 
         # 不可以这里 n = len(lists), 下面的len(lists)都用n代替， 因为lists每次都变，因为lists = next_lists， 如果n固定， 后面就乱了。
        while len(lists) > 1: 
            next_lists = []
            for i in range(0, len(lists), 2):  # 用了for 就不可以在循环内i += 2, 这个只能在while用，不要犯低级错误
                if i < len(lists)-1:           # 对于长度的奇偶，分类讨论，也可以用len(lists) % n == 0
                    new = self.meregeTwoList(lists[i], lists[i+1])
                else:
                    new = lists[i]
                next_lists.append(new)
            lists = next_lists
        return lists[0]
        
    def meregeTwoList(self, l1, l2):
        if not l1:
            return l2 
        if not l2:
            return l1 
        head = dummy = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                dummy.next = l1 
                l1 = l1.next  # 记得l1也要后移
            else:
                dummy.next = l2
                l2 = l2.next
            dummy = dummy.next
        if l1:
            dummy.next = l1 
        if l2:
            dummy.next = l2 
        return head.next
# 题五: 优秀成绩

""" 每个学生有两个属性 id 和 scores。找到每个学生最高的5个分数的平均值。
record.id, record.score"""

# 想到多个capacity heap来维持前5高， 但是没想到用dict. # 如此明显， 返回的结果也是字典，都没想到用dict！！！！！！！ 
from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def highFive(self, records):
        dict = defaultdict(list)

        for record in records:
            heappush(dict[record.id], record.score)
            if (len(dict[record.id])) > 5:
                heappop(dict[record.id])

        scores_avg = {}
        for id in dict:
            scores_avg[id] = sum(dict[id]) / 5.0
            
        return scores_avg

# 题六：K个最近的点

               """
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""
import heapq
class Solution:
    """
    @param points: a list of points
    @param origin: a point
    @param k: An integer
    @return: the k closest points
    """
    def kClosest(self, points, origin, k):
        heap = []
        if not points:
            return heap
        res = []
        for point in points:
            distance_power = (point.x - origin.x)**2 + (point.y - origin.y)**2
            heapq.heappush(heap, (distance_power, point.x, point.y))
           
        print(heap)
        for _ in range(k):  # 前k小
            distance_power, point.x, point.y = heapq.heappop(heap)
            res.append([point.x, point.y])
        return res
# 题七：  数据流中位数
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

# 第2次做： 注意最小堆变最大堆，因为不需要返回前k大，所以直接push相反数，pop相反数就可以了。               
import heapq
class Solution:
    """
    @param nums: A list of integers
    @return: the median of numbers
    """
    def medianII(self, nums):
        if not nums:
            return []
        # [4,5,1,3,2,6,0]
        min_heap = []
        max_heap = []
        res = [nums[0]]
        median = nums[0]
        for num in nums[1:]:   
            if num <= median:    
                heapq.heappush(max_heap, -num)
                if len(max_heap) > len(min_heap):
                    heapq.heappush(min_heap, median)
                    median = -heapq.heappop(max_heap)
            elif num > median: 
                heapq.heappush(min_heap, num)
                if len(min_heap) > len(max_heap) + 1:
                    heapq.heappush(max_heap, -median)
                    median = heapq.heappop(min_heap)
            res.append(median)
        return res
# 题八：接雨水II
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
        visited = [[0 for i in range(m)] for j in range(n)]   # 用二维数组记录visited, 如果用一位数组，查找时间过长
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
