# 二分法模版一：条件：sorted数组，需返回符合条件的任意位置，第一个，或最后一个。
# while start+1 < end
# mid = start + (end - start)/2
# A[mid] ==, <, >
# 出循环之后，判断A[mid], A[start], A[end]谁是答案

#题一： 给定sorted数字序列和一个target数，找到target在数组中最后的位置，如果没有，return -1

class Solution:
    def last_target(self, A,target):
        n = len(A)
        if A == None or n == 0:
            return -1
        
        start = 0
        end = n-1
        while start+1 < end:  #如果用start < end, 需要在后面做很多分类工作，这里start + 1 < end确保start和end不相邻，也就不会让mid一直等于start或end，而陷入死循环。
                              # 注意：如果这样走下去，start最终会停在倒数第二个位置，不会到达最后一位，end会停在第二个位置，不会到达第一位，这样就避免了死循环
            mid = start + (end-start)//2   # 除法向下取整用//, 取浮点用/
            if A[mid] == target:
                start = mid
            elif A[mid] < target:
                start = mid
            else:
                end = mid
        if A[end] == target: 
            return end
        if A[start] == target:
            return start
        return -1
        

            

if __name__ == '__main__':
    solution = Solution()
    print(solution.last_target([0,1,2,2,5], 2))

# 题二： find first bad version  第一个错误的代码版本     OOOOOOOOXXXXXXXXXX

#class SVNRepo:
#    @classmethod
#    def isBadVersion(cls, id)
#        # Run unit tests to check whether verison `id` is a bad version
#        # return true if unit tests passed else false.
# You can use SVNRepo.isBadVersion(10) to check whether version 10 is a 
# bad version.
class Solution:
    """
    @param n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        if n <= 0:
            return 0 
        
        start, end = 1, n 
        while start + 1 < end:
            mid = start + (end - start)//2 
            if SVNRepo.isBadVersion(mid) == True:
                end = mid 
            else:
                start = mid 
                
        if SVNRepo.isBadVersion(start)  == True:
            return start 
        return end 
 
# 题三： 一串很长的sorted数组，你无法得到其长度，给定target， 找到target的位置,可以用reader.get(k)找到第k个数的值
# 思想：倍增找到右边界 + 二分法  查看1，2，4，8，16...发现
# 思想：倍增找到右边界 + 二分法  查看第1，2，4，8，16...，如果A[16] >k， 说明k在A[8]到A[16]之间，因为之前查看过A[8]及之前的数。

class Solution:
    def get_k(self, A,target):
        kth = 1  # 不可以从0开始，因为要乘2
        while reader.get(kth - 1) < target:  # 检查到kth-1， 1， 可以查到0。 2， kth留给后面一次倍增去检查
            kth = kth * 2
                
        # start 也可以是 kth // 2，但是我习惯比较保守的写法
        # 因为写为 0 也不会影响时间复杂度
        start, end = 0, kth - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if reader.get(mid) < target:
                start = mid
            else:
                end = mid
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        return -1

#题四：find mountain number,单峰数列，找到峰值
# 注意：如果数列有相等的数，不能用二分法，只能用for循环
class Solution:
    def get_mountain(self, A):
        n = len(A)
        if(n == 0 or A == None):
            return -1
        start = 0
        end = n-1
        while start+1 < end:
            mid = start + (end - start)//2
            if A[mid] > A[mid+1]:
                end = mid
            else:
                start = mid
        return max(A[start], A[end])

if __name__ == '__main__':
    solution = Solution()
    print(solution.get_mountain([9,10,9,8]))
    
#题五： find a peak number， 多峰数列，找到一个极值
#注意：如果是找到所有极值，不可以用二分法，只能用for循环，因为二分法的思想是取一半丢一半，这样不能求到所有极值
#如果有相等的数，不能用二分法
class Solution:
    def find_peak(self, A):
        n = len(A)
        if(n == 0 or A == None):
            return -1
        start = 0
        end = n-1
        res = list()
        while start+1 < end:
            mid = start + (end - start)//2
            if A[mid] > A[mid+1] and A[mid] > A[mid-1]:
                return mid
            elif A[mid] < A[mid+1] and A[mid] > A[mid+1]:
                start = mid
            else:
                end = mid
        return -1
if __name__ == '__main__':
    solution = Solution()
    print(solution.find_peak([9,10,9,8,11,7,6]))

# 翻转排序数组 rotated sorted array 如 4，5，6，7，1，2，3 本来是sorted数列，在某一翻转
# 题六：在rotated sorted array中找到某数target?
# 关键：A[mid] 的位置不确定，所以要分情况讨论，根据rotated sorted数列的特点，翻转点A[0]将数组分成两组，大于等于A[0]和小于等于A[0] （画图可以直观看到，分别在第2，4象限）
# A[mid]不同位置的情况内，再通过判断target位置对start,end指针进行移动

class Solution:
    def serch_in_rotated_array(self, A, target):
        n = len(A)
        if(n == 0 or A == None):
            return -1
        start = 0
        end = n-1
        while start+1 < end:
            mid = start + (end - start)//2
            if A[mid] == target:
                return mid
            if A[start] < A[mid]:
                if A[start] < target and target < A[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if target < A[mid] and target < A[end]:
                    start = mid
                else:
                    end = mid
solution = Solution()
print(solution.serch_in_rotated_array([4,5,6,7,0,1,2,3], 6))


#题七：find minimum in rotated sorted array  - 没有重复元素
# 注意要与nums[end]比，不要跟nums[start]比，因为最小值一定<nums[end]

class Solution:
    def findMin(self, nums):
        if not nums:
            return None 
            
        start, end = 0, len(nums)-1 
        while start + 1 < end:
            mid = start + (end - start)//2 
            if nums[end] < nums[mid]:  # 注意不要犯低级错误， end < nums[mid],不能拿index和element比
                start = mid
            if nums[mid] <= nums[end]:
                end = mid 
        return min(nums[start], nums[end])
    
    
    
# 题八： find minimum in rotated sorted array  - 有重复元素
# 分类讨论
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        if not nums:
            return None 
            
        start, end = 0, len(nums)-1 
        while start + 1 < end:
            if nums[start] < nums[end]:  # 1， 无翻转
                return nums[start]
            mid = start + (end - start)//2 
            if nums[end] < nums[mid]:     # 2， mid大于end
                start = mid
            if  nums[end] > nums[mid]:    # 3, mid 小于end
                end = mid 
            if  nums[end] == nums[mid]:   # ！！！！ 4，mid == end, 无法判断翻转点在哪里， 让start向后移动，直到找到翻转点  【1，1，1，-1，1】【1， -1， 1， 1， 1】
                start += 1
        return min(nums[start], nums[end])
# 题九：搜索区间
# 先find first position, 然后find last position
# 2个while就可以，不用写两个函数
class Solution:
    """
    @param A: an integer sorted array
    @param target: an integer to be inserted
    @return: a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        if not A:
            return [-1,-1]
            
        start, end = 0, len(A)-1
        while start + 1 < end:
            mid = (start + end)//2 
            if A[mid] < target:
                start = mid 
            else:
                end = mid 
        if A[start] == target:
            leftBound = start
        elif A[end] == target:
            leftBound = end 
        else:
            return[-1, -1] 
        
        start, end = leftBound, len(A)-1 
        while start + 1 < end:
            mid = (start + end) // 2 
            if A[mid] <= target:
                start = mid 
            else:
                end = mid 
        if A[end] == target:
            return [leftBound, end]
        else:
            return [leftBound, start]
