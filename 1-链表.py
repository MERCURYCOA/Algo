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

# 方法1： 化整为零， 记录需要翻转的那段节点的前继节点，后继节点，翻转中间段，然后重新连接
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
        dummy = ListNode(0)
        dummy.next = head 
        p1, p2 = dummy, head
        for i in range(m-1):
            p1 = p1.next 
            p2 = p2.next 
        prev = p1
        
        for i in range(n - m+1):
            p1 = p1.next 
            p2 = p2.next
        post = p2 
        prev = self.reverse(prev, post)
        return dummy.next
    def reverse(self, prev, post):
        p1 = None
        p2 = prev.next
        while p2 and p2 != post:
            temp = p2.next
            p2.next = p1 
            p1 = p2
            p2 = temp 
        tail = prev.next
        prev.next = p1
        tail.next = post
        return prev


# 方法2：
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

# 题3: 交换链表当中的两个节点        
# 化整为零 ， 寻找节点，交换节点都写成方法， 交换节点的模版背过
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
    @param v1: An integer
    @param v2: An integer
    @return: a new head of singly-linked list
    """
    def swapNodes(self, head, v1, v2):
        dummy = ListNode(0)
        dummy.next = head 
        prev1, prev2 = self.findPrevs(dummy, v1, v2)
        if prev1 is None or prev2 is None:
            return head 
        if prev1 == prev2:
            return head 
        if prev1.next == prev2:
            self.swapAdj(prev1, prev2)
        elif prev2.next == prev1:
            self.swapAdj(prev2, prev1)
        else:
            self.swapRemote(prev1, prev2)
        return dummy.next
        
    def swapAdj(self, prev1, prev2):
        node1 = prev1.next
        node2 = node1.next
        post = node2.next 
        
        prev1.next = node2
        node2.next = node1 
        node1.next = post
        
    def swapRemote(self, prev1, prev2):
        node1 = prev1.next 
        post1 = node1.next 
        
        node2 = prev2.next 
        post2 = node2.next 
        
        prev1.next = node2
        node2.next = post1 
        prev2.next = node1 
        node1.next = post2 
        
        
        
    def findPrevs(self, dummy, v1, v2):
        prev1, prev2 = None, None 
        cur = dummy
        while cur.next:
            if cur.next.val == v1:
                prev1 = cur 
            if cur.next.val == v2:
                prev2 = cur 
            cur = cur.next 
        return prev1, prev2
        
# 题4: 重排链表
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
    @return: nothing
    """
    def reorderList(self, head):
        if not head:
           return None 
        length = 0
        cur = head 
        while cur:
           length += 1 
           cur = cur.next 
        dummy = ListNode(0)  # 这里用dummy是怕head就一个节点，不能上2个节点，也可以先判断length==1，return heda, length > 1， p1, p2 = head, head.next。这样这里就不用dummy了。
        dummy.next = head
        p1, p2 = dummy, head
        for i in range((length+1)//2):
            p1 = p1.next
            p2 = p2.next 
        
        post = p2
        p1.next = None
        post = self.reverse(post)
        cur1 = head
        cur2 = post 
        res = dummy1 = ListNode(0)
        is_1 = True
        while cur1 and cur2:
            if is_1:
                dummy1.next = cur1
                cur1 = cur1.next 

            if not is_1:
                dummy1.next = cur2
                cur2 = cur2.next 

            dummy1 = dummy1.next
            is_1 = bool(1-is_1)  # bool控制加1链还是2链，注意不可以将这句加到上面两个if里，因为假设进入第一个if, 加完cur1之后，is_1变成false， 就会进入第二个if，这是不对的。两个if只能进1个。也可以改成if,else形式。

            
        if cur1:
            dummy1.next = cur1 
        if cur2:
            dummy1.next = cur2 
        return res.next
    def reverse(self, head):
        cur = head 
        prev = None 
        while cur:
            temp = cur.next 
            cur.next = prev 
            prev = cur 
            cur = temp 
        return prev
# 题5: 旋转链表
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: the List
    @param k: rotate to the right k places
    @return: the list after rotation
    """
    def rotateRight(self, head, k):
        if not head:
            return None 
        cur = head
        length = 1
        while cur.next:  # 找到长度
            length += 1 
            cur = cur.next 
        
        if length == 1:
            return head
        if k >= length:
            k = k % length 
        if k == 0:
            return head 
        cur.next = head   #变成环     
        step = length - k   # 断口向前走几步
        
        for i in range(step):
            head = head.next 
            cur = cur.next 
        cur.next = None   # 断开尾部
        return head     # 这个head已经不是最开始的head了， head已经向前走了step步 


        
            
      
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
class Solution:
    """
    @param head: The head of linked list.
    @return: You should return the head of the sorted linked list, using constant space complexity.
    """
    def sortList(self, head):
        if not head or head.next is None:
            return head 
        pivot, cur = head, head.next
        pivot.next = None
        
        larger_dummy = ListNode(0)
        p1 = larger_dummy
        smaller_dummy = ListNode(0)
        p2 = smaller_dummy
        p3 = pivot
        
        while cur:
            if cur.val > pivot.val:
                p1.next = cur 
                p1 = p1.next 
            elif cur.val < pivot.val:
                p2.next = cur 
                p2 = p2.next 
            else:
                p3.next = cur  # 可能会有重复数字出现，pivot可能是2个及以上节点的链表
                p3 = p3.next
            cur = cur.next 
        larger = larger_dummy.next
        smaller = smaller_dummy.next 
        
        p1.next = None  # 注意断后， 因为2个节点可以指向同一个节点，假设 0->1->-1, 0是pivot, 理想中想要larger_dummy -> 1(p1), smaller_dummy -> -1(p2), 但是，如果没有断后，实际出来的是 larger_dummy->1(p1) -> -1, smaller_dummy ->-1(p2)
                        # 因为1本来就连着-1， 让smaller_dummy指向-1不会断开1->-1, 需要你来断后
        p2.next = None
        p3.next = None
        sorted_larger = self.sortList(larger)
        sorted_smaller = self.sortList(smaller)
        return self.joinList(pivot, sorted_smaller, sorted_larger)

    def joinList(self, pivot, sorted_smaller, sorted_larger):
        cur2 = pivot 
        while cur2.next:
            cur2 = cur2.next 
        if sorted_smaller is None:
            cur2.next = sorted_larger
            return pivot
        # head1, head2 = sorted_smaller, sorted_larger
        cur1 = sorted_smaller 
        while cur1.next:
            cur1 = cur1.next 
        
        cur1.next = pivot 
        cur2.next = sorted_larger 
        return sorted_smaller


