# 哈希表原理：数组里存着node， 通过hash funtion计算存入数组位置的下标，冲突的时候将新节点加到tail。
# hash function:  key = number % capacity, 就是说number这个数存到数组里的位置的index是number对capacity取余数。
# 倍增： capacity是hash table大小,即len(hash_table)，size是里边node的个数。当size/capacity == 1/10时，hash table倍增，所以节点rehashing.

# 题一： rehashing
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param hashTable: A list of The first node of linked list
    @return: A list of The first node of linked list which have twice size
    """
  
    def collision(self, node, number):
        if node.next != None:
            self.collision(node.next, number)
        else:
            node.next = ListNode(number)
            
    def add_node(self, table, number):
        p = number % len(table)
        if table[p] == None:
            table[p] = ListNode(number)
        else:
            self.collision(table[p], number)
            
    def rehashing(self, hashTable):
        CAPACITY = len(hashTable) * 2 
        new_hash_table = [None] * CAPACITY
        for i in hashTable:
            node = i
            while node:
                self.add_node(new_hash_table, node.val)
                node = node.next
        return new_hash_table    
  
# 题二：LRU cache - least recently used 缓存， 维护一定长度，最新访问的节点在最前面，超过规定长度要删掉最远访问的节点
# 关键： 字典+双向链表  {key: node}, node存4个值： prev, next, key, value
# 链表： head -> node -> node -> tail, 链表维护head, tail, 长度capacity
# 链表查找O(n), 通过字典查找node，获取node的prev, next，只需O(1) - 提高效率

class LinkedNode:
    def __init__(self, prev = None, next = None, key = None, value = None):
        self.prev = prev
        self.next = next
        self.key = key
        self.value = value
        
class LRUCache:         # 题意：创造一种新的数据结构
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        # do intialization if necessary
        self.dict = {}          # 此结构维护一个字典和一个链表（head, tail）， 固定长度
        self.head = None
        self.tail = None 
        self.capacity = capacity
    """
    @param: key: An integer
    @return: An integer
    """
    def add_node(self, node):
        self.dict[node.key] = node
        if not self.head:
            self.head = node 
            self.tail = node 
            return
        self.tail.next = node 
        node.prev = self.tail
        self.tail = node 
        
    def remove_node(self, node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        del self.dict[node.key]
        node.prev = None
        node.next = None
            
    def get(self, key):         
        # write your code here
        if key not in self.dict:
            return -1
        node = self.dict[key]
        self.remove_node(node)
        self.add_node(node)
        return node.value        # 注意返回的是value, 不是key
         
    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """
    def set(self, key, value):
        # write your code here
        if key in self.dict:    
            self.dict[key].value = value       # 修改key 的value值
            self.get(key)                      # 把刚刚访问过的node放到链表的tail
            return
        
        if len(self.dict) == self.capacity:
            self.remove_node(self.head)
            
        node = LinkedNode(self.tail, None, key, value)
        self.add_node(node)

# ========================================================================================
# 堆： 操作：O(logn) Add， O（logn）remove, O(1) min or max 
# 为什么是logn? Add操作是在二叉树的最后加入，成为最后一个叶子，然后向上调整，维持最大/最小堆，最坏情况是每层都调整，时间是logn. Remove操作是让树的最后一个叶子覆盖要删除的节点，
# 然后向上或向下调整树，时间也是logn
# 堆本质是完全二叉树，一般可以用数组表示，从上到下，从左到右将二叉树填满
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
# 题四：合并K个排序链表
# 方法一 heapq
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
# 方法二： 分治法

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
