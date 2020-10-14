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
# 堆： 操作：O(logn) Add， O（logn）pop, O(1) min or max 
# 堆本质是完全二叉树，一般可以用数组表示，从上到下，从左到右将二叉树填满
# 题一： 堆化
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
       for i in range(len(A)//2, -1, -1):
           self.siftdown(A, i)
           
    def siftdown(self, A, index):
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
