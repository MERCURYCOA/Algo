# 技巧： 改变链表结构的题，用 dummy node 


# 翻转链表：模版记住
def reverse(self, head):  # 双指针 prev, cur
        cur = head 
        prev = None  
        while cur:
            temp = cur.next  # 先把next保存一下
            cur.next = prev  # 断开cur到cur.next， 让cur.next连到prev
            prev = cur      # 双指针后移
            cur = temp
        return prev

# 题1: 翻转链表II   
# 注意不要死板， 2个指针， prev, cur,只有满足这2个指针指向2个相邻节点就可以进行翻转，至于指针叫什么不重要，很多变种题，指针还有干其他事情，翻转中间一段，至于2个指针走到需要翻转的地方就用这2个指针，不要重新造指针。
# 断开，重新接上，翻转，3个基本操作，不要太多指针，需要在哪里断开，或接上，就在指针经过的时候，重新命名一个就好。

# 方法：
# 先定义一个dummy node，dummy的next是head。p1指向dummy，p2指向head。
# 开始翻转前先将p1和p2共同前进m-1步，用p1_frozen和p2_frozen记录下当前位置。然后再多前进一步，这样一共前进了m步。
# 然后开始翻转，方法类似题目35. Reverse Linked List。
# 翻转结束后p1_frozen和p2_frozen的next分别指向p1和p2。
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: ListNode head is the head of the linked list 
    @param m: An integer
    @param n: An integer
    @return: The head of the reversed ListNode
    """
    def reverseBetween(self, head, m, n):
        if not head:
            return None 
        dummy = ListNode(0)
        dummy.next = head
        p1 = head
        p2 = dummy
        for i in range(m-1):
           p1 = p1.next  
           p2 = p2.next 
        p1_frozen = p1   # 断口
        p2_frozen = p2 
        
        p1 = p1.next 
        p2 = p2.next 
        for i in range(n-m):  # 注意如果不是全链翻转，不要用while, 要用for， 可以控制走几步
            temp = p1.next
            p1.next = p2 
            p2 = p1
            p1 = temp
        p2_frozen.next = p2  # 接口
        p1_frozen.next = p1 
        
        return dummy.next

      
# 题2：reverse nodes in k group   每k个node翻转一次链表，不够k个，不翻转
# 困难题分解成中等题，中等题分解成简单题
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
        if not head:
            return None 
        cur = head
        count = 0
        while cur:
            count += 1 
            cur = cur.next
        if k >= count:
            k = count 
        if k <= 0:
            k = 1 
        x = count // k  # 代表有x段需要翻转          #例如 count = 5, k = 2
        
        for i in range(x):  # 0, 1
            head = self.reverse(head, i*k+1, (i+1)*k)    # (1, k)(K+1, 2k)(2k+1, 3k)
        return head
        
    def reverse(self, head, m, n):  # 翻转联邦II的模版
        dummy = ListNode(0)
        dummy.next = head 
        p1, p2 = dummy, head 
        for i in range(m-1):
            p1 = p1.next 
            p2 = p2.next 
        p1_frozen = p1 
        p2_frozen = p2 
        p1 = p1.next 
        p2 = p2.next 
        
        for i in range(n-m):
            temp = p2.next 
            p2.next = p1 
            p1 = p2 
            p2 = temp 
        p1_frozen.next = p1 
        p2_frozen.next = p2 
        
        return dummy.next
        
        
       
# 题一：链表划分
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The first node of linked list
    @param x: An integer
    @return: A ListNode
    """
    def partition(self, head, x):
        if not head:
            return None
        # 2个dummy    
        res = dummy_smaller = ListNode(0) # 区分变化指针和固定指针，最后返回的是固定的，变化的指针会跑走
        dummy = ListNode(0)  
        dummy.next = head
        prev = dummy  # 挖掉中间的节点，需要重新在前一节点和后一节点建立连接，所以需要记录前一个节点prev
        cur = head
        while cur:
            if cur.val < x:
                temp = cur.next
                dummy_smaller.next = cur 
                prev.next = temp 
                cur = temp
                dummy_smaller = dummy_smaller.next
            else:
                cur = cur.next
                prev = prev.next
        dummy_smaller.next = dummy.next
        return res.next  # 不能返回dummy_smaller
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
# 题三：linked list cycle
# trick: 快慢指针
class Solution:
    """
    @param head: The first node of linked list.
    @return: True if it has a cycle, or false
    """
    def hasCycle(self, head):
        # write your code here
        if head == None or head.next == None:
            return False 
            
        slow = head
        fast = head
        while fast and slow:
            if fast.next:  # 这里必须判断
                slow = slow.next
                fast = fast.next.next
                if slow and fast and slow == fast:
                    return True
            else:  # 如果fast存在，fast.next不存在，就会无限循环， 所以必须else break
                break
        return False 
# 题四：如果是linked list, 返回入口节点， 如果不是，返回None
class Solution:
    """
    @param head: The first node of linked list.
    @return: The node where the cycle begins. if there is no cycle, return null
    """
    def detectCycle(self, head):
        # write your code here
        if head == None or head.next == None:
            return None 
            
        slow = head
        fast = head
        while fast and slow:
            if fast.next:  # 这里必须判断
                slow = slow.next
                fast = fast.next.next
                if slow and fast and slow == fast:
                   break # 注意这里两个while, 内层不可以return, 这里要break
            else:  
                break
        if slow == fast:
            slow = head # slow撤回开头
            while slow != fast:  # 错误做法：while fast and slow: slow = slow.next fast = fast.next if slow == fast: return fast
                                 # 如果入口是第一个节点，那么他们相遇一定也是第一个节点，这时应该返回当前节点，按照错误做法，会进入while， 在第二节点处发现相等，
                                 # 返回的是第二节点，但其实入口在第一节点。 正确做法是：while slow!= fast, 只有二者不相等才能进入while
                slow = slow.next
                fast = fast.next
            return fast
        return None 
# 题五：链表排序 要求 时间nlogn 空间 O（1）
#方法一：归并排序   注意：数组归并需要空间O(n),因为你要新建数组，链表则只需要新建一个dummy node，比较大小后将小node加到后面即可，所以空间是O(1)
# 分治模版背过！！！！！！！！！！！
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The head of linked list.
    @return: You should return the head of the sorted linked list, using constant space complexity.
    """
    def sortList(self, head):
        # write your code here
        if head == None or head.next == None:
            return head
        mid = self.findMiddle(head)
        left = head
        right = mid.next
        mid.next = None
       
        # 递归的拆解
        sorted_left = self.sortList(left)
        sorted_right = self.sortList(right)
        
        # 递归的出口
        return self.merge(sorted_left, sorted_right)

    def findMiddle(self, head):
        if head == None or head.next == None:
            return head
        slow = head
        fast = head
        while fast.next and fast.next.next:  #这样mid才能停在靠左的地方，
                                             # mid.next才不会越界
            slow = slow.next
            fast = fast.next.next
        return slow
        
    def merge(self, head1, head2):
        if not head1:
            return head2
        if not head2:
            return head1
            
        
        dummy = ListNode(0)
        cur = dummy  # dummy最后返回需要用，需要一个指针
        while head1 and head2:
            if head1.val < head2.val:
                cur.next = head1
                cur = cur.next
                head1 = head1.next
            else:
                cur.next = head2
                cur = cur.next
                head2 = head2.next
            
        if head1:
            cur.next = head1
        if head2:
            cur.next = head2
        return dummy.next    
    
# 方法二： 快排


