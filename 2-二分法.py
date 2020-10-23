# 二维二分： 行的二分 O(NlogN)，列的二分O(MlogM)， 行列交叉二分O(N+M)
# 题一：find peak element II 
# 方法一：行的二分 

import sys
class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def findPeak(self, A):
        if len(A) == 0 or len(A[0]) == 0:
            return [-1, -1]
        return self.binary(A, 0, len(A))
    
    def binary(self, A, sky, ground):
        mid = sky + (ground - sky)//2
        peak = -sys.maxsize-1
        peak_col = 0
        for i, x in enumerate(A[mid]): #找到中间行的最大值 peak
            if x > peak:
                peak = x
                peak_col = i
        # A[mid][peak_col]
        if mid > 0 and mid < ground-1: # 根据题意，第一行一定大于第二行，倒数第二行一定大于最后以后，所以A最少3行， 这里防止mid溢出
            up = A[mid-1][peak_col]
            down = A[mid+1][peak_col]
            if max(up, down) < peak: # 情况1:上下都比peak小，说明行peak值也是列peak值
                return peak
            elif up> down:          # 情况2：上面大于下面，接下来就向上寻找
                self.binary(A, sky, mid)
            elif down > up:         # 情况3: 下面大于上面，向下寻找
                self.binary(A, mid, ground)

            
              
solution = Solution()
print(solution.findPeak([
      [1, 2, 3, 6,  5],
      [16,41,23,22, 6],
      [15,17,24,21, 7],
      [14,18,19,20,10],
      [13,14,11,10, 9]
    ]))
# 24


  
