# 技巧： 改变链表结构的题，用 dummy node 
# 题一：reverse nodes in k group   每k个node翻转一次链表，不够k个，不翻转
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: a ListNode
    @param k: An integer
    @return: a ListNode
    """
    def reverseKGroup(self, head, k):
        # write your code here
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        while prev:
            prev = self.reverse(prev, k)
        return dummy.next
    
    def reverse(self, head, k):
        curt = head
        n1 = head.next
        
        for i in range(k):
            curt = curt.next
            if curt == None:
                return None
        nk = curt
        nkplus = curt.next
        prev = head
        curt = head.next
        while curt != nkplus:
            temp = curt.next
            curt.next = prev
            prev = curt
            curt = temp
        head.next = nk
        n1.next = nkplus
        return n1

# 题二： copy list with random pointer
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

class Solution:
    # @param head: A RandomListNode
    # @return: A RandomListNode
    def copyRandomList(self, head):
        # write your code here
        list1 = self.copyList(head)
        list2 = self.randomPointer(list1)
        return self.split(list2)
    def copyList(self, head):
        cur = head
        while cur:
            node = RandomListNode(cur.label)
            node.next = cur.next
            cur.next = node
            cur = cur.next.next
        return head
    def randomPointer(self, head):
        old = head
        copy = head.next
        while copy.next:
            if old.random:
                copy.random = old.random.next
            else:
                copy.random =  None
            old = copy.next
            copy = old.next
        if copy:
            if old.random:
                copy.random = old.random.next
            else:
                copy.random =  None
        return head
    def split(self, head):
        prev = head
        res = cur = head.next
        while cur.next:    
            prev.next = prev.next.next
            cur.next = cur.next.next
            prev = prev.next
            cur = cur.next
        return res
