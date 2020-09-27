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
            if mid == target:
                start = mid
            elif mid < target:
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

