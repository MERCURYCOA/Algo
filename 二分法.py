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

# 题二： find first bad version      OOOOOOOOXXXXXXXXXX

class Solution:
    def get_first_bad_version(self, A):
        n = len(A)
        if(n == 0 or A == None):
            return -1
        start = 0
        end = n-1
        while start+1 < end:
            mid = start + (end - start)//2
            if A[mid] == 1:
                end = mid
            else:
                start = mid
        
        if A[end] == 1:
            return end
        if A[start] == 1:
            return start
        return -1

if __name__ == '__main__':
    solution = Solution()
    print(solution.get_first_bad_version([0,0,0,1,1,1]))
 
# 题三： 一串很长的sorted数组，你无法得到其长度，给定target， 找到target的位置,可以用reader.get(k)找到第k个数的值
# 思想：倍增找到右边界 + 二分法  查看1，2，4，8，16...发现
# 思想：倍增找到右边界 + 二分法  查看第1，2，4，8，16...，如果A[16] >k， 说明k在A[8]到A[16]之间，因为之前查看过A[8]及之前的数。

class Solution:
    def get_k(self, A,target):
        n = len(A)
        if(n == 0 or A == None):
            return -1
        count = 1
        while reader.get(count-1) < target:
            count = count*2
        start = count/2
        end = count - 1
        while start+1 < end:
            mid = start + (end - start)//2
            if A[mid] == target:
                return mid
            elif A[mid] > target:
                end = mid
            else:
                start = mid
        
        if A[end] == target:
            return end
        if A[start] == target:
            return start
        return -1

#题四：find mountain number,单峰数列，找到峰值
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
