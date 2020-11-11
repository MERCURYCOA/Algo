# 堆： 操作：O(logn) Add， O（logn）remove, O(1) min or max 
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

# 题一： 堆化
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
# 堆的应用 heapq
# 题二：ugly number II

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
