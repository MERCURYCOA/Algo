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

#===========================================================================================
#数组
# 题一：找到2个数组的中位数  要求：logn

#思路：根据中位数定义， n是数组A和B元素数之和， n是基数，中位数是n//2+1, n是偶数，中位数是最中间两个数的平均。
# 问题转化为2个数组中求第k小的数。想用logn解决，就要用O(1)的时间将问题降为k/2。分别给A，B各一个指针，先找到各自第K//2个元素，比较大小，较小的值所在的数组，例如A，的前k//2个元素一定
# 不包括第k小元素，所以让A的指针指向前走k//2，接下来就变成找第(k-k//2)小的数了,进入递归
class Solution:
    """
    @param: A: An integer array
    @param: B: An integer array
    @return: a double whose format is *.5 or *.0
    """
    def findMedianSortedArrays(self, A, B):
        # write your code here
        
        n = len(A) + len(B)
        if n%2 == 1:
            return self.findKth(0,A, 0, B, n//2 + 1)
            
        else:
            smaller = self.findKth(0,A, 0,B, n // 2)
            bigger = self.findKth(0,A, 0, B, n // 2 + 1)
            return (smaller + bigger) / 2  # 不可以直接让k = (n//2 + n//2+1)/2  因为n是偶数时，需要先求出最中间的两个数，然后求平均
        
        
    def findKth(self, index_a, A, index_b, B, k):
        if len(A) == index_a:  # A到头了
            return B[index_b+k-1]
        if len(B) == index_b:  # B到头了
            return A[index_a+k-1]
        if k == 1:    # 递归出口  找当前最小的数，只有比较2个指针当前的数的大小就可以
            return min(A[index_a], B[index_b])
            
        a = A[index_a + k//2 -1] if index_a + k//2 <= len(A) else None   # a是指针向前走k//2后指向的元素， 因为index从0开始，正如第k个元素是A[k-1], 这里a=A[index_a + k//2 -1]
        b = B[index_b + k//2-1] if index_b + k//2 <= len(B) else None   # 注意不越界
        
        if b is None or (a is not None and a < b):
            return self.findKth(index_a+k//2, A, index_b, B, k-k//2)  # 这里k的递归必须是k-k//2， 不可以是k//2, 因为考虑奇偶
        return self.findKth(index_a, A, index_b + k//2, B, k-k//2)

    
# 题二： maximum subarray  包含负数  时间O(n)
# 记录sum,min_sum, sum和min_sum的最大差值,最后的最大差值就是max subarray
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        if not nums or (len(nums) == 0):
            return None
        n = len(nums)    
        max_diff = float("-inf")  #负无穷
        sum = 0
        min_sum = 0
        for j in range(n):
            sum += nums[j]
            max_diff = max(max_diff, sum - min_sum)
            min_sum = min(sum, min_sum)
            
        return max_diff
# 题三： minimum subarray 
# 思路：所有元素取相反数，求maximum就可以了
