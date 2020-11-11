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
# 第2次做：
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
    def addNode(self, num, hashTable):
        CAPACITY = len(hashTable)
        index = num % CAPACITY
        node = ListNode(num)
        if hashTable[index]:
            cur = hashTable[index]
            while cur.next:
                cur = cur.next
            cur.next = node 
        else:
            hashTable[index] = node 
    
    def rehashing(self, hashTable):
        CAPACITY = 2 * len(hashTable)
        new_table = [None] * CAPACITY
        for node in hashTable:
            while node:
                self.addNode(node.val, new_table)
                node = node.next  
        return new_table           
                
        
        
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
# 题三：乱序字符串
# 对字符串排序，作为key
class Solution:
    """
    @param strs: A list of strings
    @return: A list of strings
    """
    def anagrams(self, strs):
        res = []
        dict = {}
        if not strs:
            return 
        for char in strs:
            sortedword = ''.join(sorted(char))
            if sortedword not in dict:
                dict[sortedword] = [char]
            else:
                dict[sortedword].append(char)
        for key in dict:
            if len(dict[key]) > 1:
                res.extend(dict[key])  # 不能用append, 因为结果是一维数组
        return res
# 题四：
# 拿那个100, 4, 200, 1, 3, 2样例，你该怎么数数呢？你先从100数，然后呢，就没有了。再从4开始数，唉，不对，不应该，因为后面还有3，2，1 所以应该把4跳过去，待会从小的数开始数。再后面是200，因为没有199，所以应该从200开始。
# 或者这样看，每一个连续序列都可以被这个序列的最小值代表，要找到最小值才开始数，这样无重复，才能做到O(N)
# 具体来看，这个代码做了三个N的操作： 1. 建dict 2. for循环里，看每一个数字n是否有n-1存在 3. while循环，从小到大的数连续序列
    class Solution:
    
    def longestConsecutive(self, nums) -> int:
        num = list(set(num))
        max_len, table = 0, {num:True for num in nums}

        for lo in nums:
            if lo - 1 not in table:
                hi = lo + 1 
                while hi in table:
                    hi += 1 
                max_len = max(max_len, hi - lo)
                
        return max_len
    
# 如果让求这个最长序列而不是求长度呢？
# 那就把hi都加进来
num = list(set(num))
max_len, table = 0, {num:[] for num in nums}  # 字典的value都变成[]

for lo in nums:
    if lo - 1 not in table:
        hi = lo + 1 
        while hi in table:
            table[lo].append(hi)  # 把hi都加进lo的value里
            hi += 1 
        max_len = max(max_len, hi - lo)
res = []            
for key, value in table.items(): # 扫一遍dict，找value最长的，记得把key加进value作为res
    cur = [key] + value
    if len(cur) > max_len:
        max_len = len(res)
        res = cur
return res
    # ========================================================================================

