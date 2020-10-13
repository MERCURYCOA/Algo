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
        
