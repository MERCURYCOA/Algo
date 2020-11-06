# 内容一：排序数组    内容二：前缀和 + 字典 求subarray sum ， O(n)  注意，最大平均值子数组也可以转化为最大和子数组问题，用前缀和

# 合并排序数组 - 不要想的太简单       
 # 题1: 合并排序数组II 
# 方法1:
class Solution:
    #@param A and B: sorted integer array A and B.
    #@return: A new sorted integer array
    def mergeSortedArray(self, A, B):
        # write your code here
        ans = []
        i, j = 0, 0
        n, m = len(A), len(B)
        while i < n and j < m :
            if A[i] < B[j] :
                ans.append(A[i])
                i += 1
            else :
                ans.append(B[j])
                j += 1
        while i < n :
            ans.append(A[i])
            i += 1
        while j < m :
            ans.append(B[j])
            j += 1
        return ans
# 方法二：
class Solution:
   
    def mergeSortedArray(self, A, B):
        return sorted(A + B)
# 题2 ： 合并排序数组
# A非常大，B并入A
# 关键：从后往前排
class Solution:
    """
    @param: A: sorted integer array A which has m elements, but size of A is m+n
    @param: m: An integer
    @param: B: sorted integer array B which has n elements
    @param: n: An integer
    @return: nothing
    """
    def mergeSortedArray(self, A, m, B, n):
        if not B:
            return A 
        pos = n + m - 1 
        i = m - 1
        j = n - 1
        while i >= 0 and j >= 0 and pos >= 0:
            if A[i] > B[j]:
                A[pos] = A[i]
                i -= 1 
                pos -= 1 
            else:
                A[pos] = B[j]
                j -= 1 
                pos -= 1 
        while i >= 0:
            A[pos] = A[i]
            i -= 1 
            pos -= 1
        while j >= 0:
            A[pos] = B[j]
            j -= 1 
            pos -= 1 
        return A
# 题3: 两数组的交集
# 方法一： 排序 + 双指针  O(mlogm + nlogn)
class Solution:
    """
    @param nums1: an integer array
    @param nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        if nums1 is None or nums2 is None:
            return []
        res = set()
        nums1.sort()
        nums2.sort()
        i , j = 0,0 
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                res.add(nums1[i])
                i += 1 
                j += 1 
            elif nums1[i] > nums2[j]:
                j += 1 
            else:
                i += 1 
            
        return list(res)

# 方法二：2个set()   O(m+n)
class Solution:
    """
    @param nums1: an integer array
    @param nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        if nums1 is None or nums2 is None:
            return []
        res = set() 
        visited = set()
        for x in nums1:
            if x not in visited:
                visited.add(x)
        for y in nums2:
            if y in visited:
                res.add(y)
        return list(sorted(res))

# 题1： 找到两个排序数组（A, B）第k小的数  -- 每次删 k//2 个一定不是第k小的元素，
# 1 对A, B每次向前移动k//2， 找到A， B从当前offset开始第k//2个元素a, b。  2 如果a < b, （或者b== None），说明a及之前的元素一定没有第k小元素，b有可能是第k小，所以让A的offset向后再移动k//2 
# 3 如果b < a, 就舍去b及之前的k//2个元素，offset向后移动k//2

def findKth(index_a, A, index_b, B, k):
        if len(A) == index_a:
            return B[index_b+k-1]
        if len(B) == index_b:
            return A[index_a+k-1]
        if k == 1:
            return min(A[index_a], B[index_b])
            
        a = A[index_a + k//2 -1] if index_a + k//2 <= len(A) else None   # 第k//2个数的下标是k//2 - 1, 要分清第i个和i下标的元素
        b = B[index_b + k//2-1] if index_b + k//2 <= len(B) else None
        print('a  %s' %a)
        print('b  %s' %b)
        if b is None or (a is not None and a < b):
            return findKth(index_a+k//2, A, index_b, B, k-k//2)
        return findKth(index_a, A, index_b + k//2, B, k-k//2)
print(findKth(0,[1,4,6], 0, [2,7,9], 4))
# 6
# 题一：找到2个数组的中位数  要求：logn

#思路：根据中位数定义， n是数组A和B元素数之和， n是基数，中位数是n//2+1, n是偶数，中位数是最中间两个数的平均。
# 问题转化为2个数组中求第k小的数。想用logn解决，就要用O(1)的时间将问题降为k/2。分别给A，B各一个指针，先找到各自第K//2个元素，比较大小，较小的值所在的数组，例如A，的前k//2个元素一定
# 不包括第k小元素，所以让A的指针指向前走k//2，接下来就变成找第(k-k//2)小的数了,进入递归
# 化繁为简，化整为零，中位数问题转化为找第n//2小的数
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

# 连续子数组
    
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
# 题二-1: 子数组的最大平均值

class Solution:
    """
    @param nums: an array
    @param k: an integer
    @return: the maximum average value
    """
    def findMaxAverage(self, nums, k):
        # Write your code here
        n = len(nums)
        sum = [0 for i in range(n + 1)]
        for i in range(1, n + 1):
            sum[i] = sum[i - 1] + nums[i - 1]
        ans = sum[k]
        for i in range(k + 1, n + 1):
            ans = max(ans, sum[i] - sum[i - k])
        return ans * 1.0 / k
    
    # 题三： minimum subarray , 和大于S的最小子数组
import sys
class Solution:
    """
    @param nums: an array of integers
    @param s: An integer
    @return: an integer representing the minimum size of subarray
    """
    def minimumSize(self, nums, s):
        if not s:
            return -1
            
        min_size = sys.maxsize
        sum, j = 0, 0
        for i in range(len(nums)):
            while j < len(nums) and sum < s: # j循环的条件 - while
                sum += nums[j]
                j += 1 
            if sum >= s:                    # j停下的条件 - if
                min_size = min(min_size, j-i)
            sum -= nums[i]
        if min_size == sys.maxsize:
            return -1
        return min_size
    
    
# 题四： 子数组之和为0  #前缀和存入dict
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        # write your code here
        if not nums or (len(nums) == 0):
            return None
        res = []
        sum = 0
        sums =  {0:-1}  # {sum : index} 不可以反过来，因为后面需要根据sum求index， 也就是根据key求value， 如果反过来，根据value求key很麻烦
        for i, num in enumerate(nums):  # enumerate 数组index和元素
            sum += num
            if sum in sums:
                res.append(sums[sum] + 1)  # prefixSum[i]记录A[0]到A[i-1]的和，prefixSum数组起始点是index = -1， 所以子数组为0的起点下标要+1
                res.append(i)
                break
            sums[sum] = i
            
        return res
# 题五：子数组之和最接近0，返回index

class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySumClosest(self, nums):
        # write your code here
        prefix_sum = [(0, -1)]
        for i, num in enumerate(nums):
            prefix_sum.append((prefix_sum[-1][0] + num, i))  # [-1]指的是最后一对元素
        
        prefix_sum.sort()  #按照sum大小排序，排过序之后index是乱的
        
        closest, answer = sys.maxsize, []
        for i in range(1, len(prefix_sum)):
            if closest > prefix_sum[i][0] - prefix_sum[i - 1][0]:  #不要字典就是因为不能进行i-1, i的比较
                closest = prefix_sum[i][0] - prefix_sum[i - 1][0]  #这里一直记录的是 最小的prefix_sum差值。 注意： 在没排序的时候 prefix_sum[i]和prefix_sum[i-1]不一定是相邻的，这就是我们要找的prefix_sum[j+1] - prefix_sum[i]
                left = min(prefix_sum[i - 1][1], prefix_sum[i][1]) + 1   # 因为index是乱的，所以要找到prefix_sum[i]和prefix_sum[i-1]谁的坐标靠左就是起点，靠右就是终点
                right = max(prefix_sum[i - 1][1], prefix_sum[i][1])
                answer = [left, right]
        
        return answer
# 题六：连续子数组求和 ， 返回和最大的子数组的起点和终点坐标，如果有相同的最大和，返回坐标字典序最小，假设[0,3]和[1,3]的和都是最大的，返回[0,3]       
class Solution:
    """
    @param: A: An integer array
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def continuousSubarraySum(self, A):
        if not A:
            return []
            
        f = [A[0]]
        start = 0
        for i in range(1, len(A)):     # f[i]是当前子数组的最大和
            if f[i-1] + A[i] <= A[i]:  # 如果A[i]大于f[i-1]+A[i], 当前i就是新的子数组的起点，将A[i]存入f[i]
                f.append(A[i])
            else:
                f.append(f[i-1] + A[i])
        max_value = max(f)
        end = 0
        for i in range(len(f)):
            if f[i] == max_value:    # 遍历f,找到最大和子数组结束的位置
                end = i 
                break 
        start = 0                    # 接下来找起点的位置
        sum_ = sum(A[:end+1])        # 把元素组A中，end位置之前的所有元素求和，用这个和从A[0]开始减，一直减到这个和等于最大值，说明找到了这个子数组的起点
        for i in range(0, end+1):
            if sum_ == max_value:
                start = i
                break 
            sum_ -= A[i]
            
        return [start, end]
# 题七：连续子数组求和
# 给定一个整数循环数组（头尾相接），请找出一个连续的子数组，使得该子数组的和最大。输出答案时，请分别返回第一个数字和最后一个数字的值。
# 方法 - 取反：
# 1， 求最大和 sum1。 
# 2 求最小和 
# 3 用数组总和减去最小和就是循环数组的潜在最大和 sum2。 
# 4 比较sum1和sum2， 返回更大的那一个的起点和终点坐标

# trick: 怎么求最小和？ 将数组A中所有元素取相反数，求新数组的最大和。函数用最大和的函数就可以，参数A取反。
  @param: A: An integer array
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def continuousSubarraySumII(self, A):
        max_start, max_end, max_sum = self.find_maximum_subarray(A)
        min_start, min_end, min_sum = self.find_maximum_subarray([-a for a in A])
        min_sum = -min_sum  # *-1 after reuse find maximum array
        
        total = sum(A)
        if max_sum >= total - min_sum or min_end - min_start + 1 == len(A):
            return [max_start, max_end]
        
        return [min_end + 1, min_start - 1]
        
    def find_maximum_subarray(self, nums):
        max_sum = -sys.maxsize
        curt_sum, start = 0, 0
        max_range = []
        
        for index, num in enumerate(nums):
            if curt_sum < 0:
                curt_sum = 0
                start = index
            curt_sum += num
            if curt_sum > max_sum:
                max_sum = curt_sum
                max_range = [start, index]
                
        return max_range[0], max_range[1], max_sum


