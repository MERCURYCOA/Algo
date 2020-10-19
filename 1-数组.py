
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
    
    
# 题四： 子数组之和为0
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
        sums =  {0:-1}
        for i, num in enumerate(nums):  # enumerate 数组index和元素
            sum += num
            if sum in sums:
                res.append(sums[sum] + 1)  # prefixSum[i]记录A[0]到A[i-1]的和，记得+1
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
        

