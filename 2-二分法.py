# 主要内容 ： 二维二分， 二分答案（找到满足某条件的最大值或最小值）， partition (quick select)模版

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
                 # return peak            #返回极值本身
                   return [mid, peak_col] # 返回坐标
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

# 方法二：行列交替二分
import sys
class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def findPeak(self, A):
        if len(A) == 0 or len(A[0]) == 0:
            return [-1, -1]
        return self.binary_row(A, 0, len(A)-1, 0, len(A[0])-1)
    
    def binary_row(self, A, sky, ground, left, right):
        mid = sky + (ground - sky)//2
        peak = -sys.maxsize-1
        peak_col = 0
        for i  in range(left, right+1): #找到中间行的最大值 peak
            if A[mid][i] > peak:
                peak = A[mid][i]
                peak_col = i
        # A[mid][peak_col]
        if mid > 0 and mid < len(A)-1: # 根据题意，第一行一定大于第二行，倒数第二行一定大于最后以后，所以A最少3行， 这里防止mid溢出
            up = A[mid-1][peak_col]
            down = A[mid+1][peak_col]
            if max(up, down) < peak: 
                 # return peak            #返回极值本身
                   return [mid, peak_col] # 返回坐标
            elif up> down:          
                self.binary_col(A, sky, mid, left, right) # 行二分后，确定取上半部分，调用binary_col,对列进行二分
            elif down > up:         
                self.binary_col(A, mid, ground, left, right)
    
    def binary_col(self, A, sky, ground, left, right):
        mid = left + (right - left)//2
        peak = -sys.maxsize-1
        peak_row = 0
        for i in range(sky, ground+1): 
            if A[i][mid] > peak:
                peak = A[i][mid]
                peak_row = i
        if mid >0 and mid < len(A[0])-1:
            l = A[peak_row][mid-1]
            r = A[peak_row][mid+1]
            if max(l, r) < peak:
                # return peak 
                return A[peak_row, mid]
            elif l > r:
                self.binary_row(A, sky, ground, left, mid)  # 对列二分后，确定取左半部分，调用binary_row,再对行进行二分
            elif r > l:
                self.binary_col(A, sky, ground, mid, right)
            
solution = Solution()
print(solution.findPeak([
      [1, 5, 3],
      [4,10, 9],
      [2, 8, 7]
    ]))

# 二分答案
# 步骤/ 模版：
#   start = 1, end = max    1, 找到可行解范围
#   while start + 1 < end:
#       mid = start + (end + start)//2    2,猜一个答案
#       if check mid:                     3, 检验答案
#           start = mid                   4, 调整搜索范围
#       else:
#           end = mid
# if end: return
# if start: return

# 题二：实现sqrt(x) x是int
# 找到第一个a使得a^2 <= x

class Solution:
    """
    @param x: An integer
    @return: The sqrt of x
    """
    def sqrt(self, x):
        start, end = 0, x 
       
        while start+1 < end:
            mid = start + (end-start)//2
            if mid*mid <= x:
                start = mid
            else:
                end = mid 
            
        if end*end <= x:
            return end 
        return start
# 题三：实现sqrt(x) x是float

class Solution:
    """
    @param: x: a double
    @return: the square root of x
    """
    def sqrt(self, x):
        # write your code here
        start, end = 0, x if x > 1 else 1
        while start + 1e-12 < end:
            mid = (start + end) / 2
            if mid * mid + 1e-12 < x:
                start = mid
            else:
                end = mid
        
        return start
# 题四： wood cut
# 满足条件：总共至少k段，使得每段木头最长多少（最大值）
class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """
    def woodCut(self, L, k):
        if not L:
            return 0
        start, end = 1, max(L) # 每段木头的范围 （注意右侧范围是max(L), 假设k=1, 最大长度就应该是L里面最长的那一个）
        while start + 1 < end:
            mid = (end + start)//2     # 猜一个值 mid
            if self.get_pieces(L, mid) >= k:  # 检验每段木头长mid时，L最多能切多少， 如果>=k，说明mid小了，每段木头还能再切长点
                start = mid 
            else:
                end = mid                    #  如果<k，说明mid大了，每段木头还能再切短点
        if self.get_pieces(L, end) >= k:   #注意，应该先检验end,因为end一定在start右边，代表end比start更长一点，如果start和end都能满足>=k, 那么应该返回end, 因为要求最大长度，所以end在前
            return end
        if self.get_pieces(L, start) >= k:
            return start
      
        return 0
            
    def get_pieces(self, L, a):
        pieces = 0
        for l in L:
            pieces += l//a 
        return pieces
# 题五： copy books

class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        if not pages:
            return 0
            
        start, end = max(pages), sum(pages)  # 最少时间：一个人复制1本最厚的书。最长时间：1个人复制所有书
        while start + 1 < end:
            mid = (start+end)//2
            if self.get_least_people(pages, mid) <= k:
                end = mid
            else:
                start = mid
        if self.get_least_people(pages, start) <= k:
            return start
        return end
    
    def get_least_people(self, pages, time_limit):  # 给定pages和最少时间， 求最少需要几个人
        people = 1 
        time_cost = 0 #累计时间
        for page in pages:
            if time_cost + page > time_limit:
                people += 1 
                time_cost = 0 # 新加一个人，时间累积清零
            time_cost += page #当前page加给新来的这个人
        return people 
