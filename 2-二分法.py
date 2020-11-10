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
# Last number that number^2 <= x
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
# Last number that number^2 <= x - eps
class Solution:
    """
    @param: x: a double
    @return: the square root of x
    """
    def sqrt(self, x):
        # write your code here
        start, end = 0, x if x > 1 else 1  # 注意：如果x<1，让x=1,因为如果直接用小于1的小数，返回的是该数本身，例如x= 0.5， 返回的也是0.5，所以这里让小于1的x=1。
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
# 题五：寻找重复的数

class Solution:
    """
    @param nums: an array containing n + 1 integers which is between 1 and n
    @return: the duplicate one
    """
    def findDuplicate(self, nums):
        start, end = 1, len(nums) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if self.smaller_than_or_equal_to(nums, mid) > mid: # 小于mid的元素个数大于等于mid，说明重复元素在前面
                end = mid
            else:                                               # 小于mid的元素个数小于mid，说明重复元素在后面
                start = mid
                
        if self.smaller_than_or_equal_to(nums, start) > start: #先检查前面的
            return start
            
        return end
        
    def smaller_than_or_equal_to(self, nums, val):
        count = 0
        for num in nums:
            if num <= val:
                count += 1
        return count
    
    
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

# 题五： 找到两个排序数组（A, B）第k小的数  -- 每次删 k//2 个一定不是第k小的元素，
# 1 对A, B每次向前移动k//2， 找到A， B从当前offset开始第k//2个元素a, b。  2 如果a < b, （或者b== None），说明a及之前的元素一定没有第k小元素，b有可能是第k小，所以让A的offset向后再移动k//2 
# 3 如果b < a, 就舍去b及之前的k//2个元素，offset向后移动k//2

def findKth(index_a, A, index_b, B, k):
        if len(A) == index_a:
            return B[index_b+k-1]
        if len(B) == index_b:
            return A[index_a+k-1]
        if k == 1:
            return min(A[index_a], B[index_b])
            
        a = A[index_a + k//2 -1] if index_a + k//2 <= len(A) else None
        b = B[index_b + k//2-1] if index_b + k//2 <= len(B) else None
        print('a  %s' %a)
        print('b  %s' %b)
        if b is None or (a is not None and a < b):
            return findKth(index_a+k//2, A, index_b, B, k-k//2)
        return findKth(index_a, A, index_b + k//2, B, k-k//2)
print(findKth(0,[1,4,6], 0, [2,7,9], 4))

# Partition 

 # 题六：第k大元素   

# 先找一个pivot, 把pivot 排在正确的位置，也就是pivot左边的数比pivot大，右边的数比pivot小（因为找的是第k大）
# 然后判断现在这个pivot的位置是不是k，如果是，皆大欢喜，返回当前pivot的位置，
# 如果不是，就看第k大的位置在pivot位置的左边还是右边，如果在左边，就向左搜索，如果在右边，就向右搜索，注意向右搜索，就不是找第k大了，而是找 k-(i-start)了
# 在向左或向右递归时，再找一个pivot，把这个pivot放到正确的位置，然后看第k位置与pivot的位置的相对位置

# 核心：不断更新pivot，直到pivot排好之后的位置就是k，那么这个pivot就是第k大的数，返回当前pivot

# 递归调用本身需要return
# 调用其他函数不影响该return的地方
class Solution:
    """
    @param n: An integer
    @param nums: An array
    @return: the Kth largest element
    """
    def kthLargestElement(self, n, nums):
        if nums == None:
            return -1  
            
        return self.quickSelect(nums, 0, len(nums)-1, n)
        
    def quickSelect(self, nums, start, end, n):
        if start == end:
            return nums[start]
        i, j = start, end 
        pivot = nums[(i+j)//2]
        while i <= j:
            while i <= j and nums[i] > pivot:
                i += 1 
            while i <= j and nums[j] < pivot:
                j -= 1 
            if i <= j:
                nums[i], nums[j] = nums[j], nums[i]       # 这个循环之后，i,j与数组的位置变成   _____ j _ i ____
                i += 1                                    #                  |
                j -= 1                                    #                  |
                                                          #                  V
        if start + n - 1 <= j:                            #  第k大的位置在j左边   【 start + k - 1就是第k大元素的位置（细节：start起点，第k大就 +k， 但是多算了一个数，所以-1），
            return self.quickSelect(nums, start, j, n)   
        if start + n - 1 >= i:                            #  第k大的位置在i右边
            return self.quickSelect(nums, i, end, n - (i-start))
        return nums[j+1]                                  #  第k大的位置就在j和i的中间， 返回中间位置的数 nums[j+i]

    
# 上面的partition与递归结合，在数组中不断搜索
# 下面有的想partition， 但是其实是二分法， 找数组中第一个小于等于target位置
# 题七：第一个小于等于target
def find(self, presum, target): #求在presum中，第一个小于等于target的位置
        m = len(presum)
        if presum[m-1] < target:
            return m
        
        start, end = 0, m - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if target <= presum[mid]:
                end = mid 
            else:
                start = mid
        
        if presum[end] < target:
            return end + 1
        if presum[start] < target:
            return start + 1
        return 0
# partition 变形
# 题八： 摆动排序
# 可以用sort,然后相邻互换 复杂度O(nlogn)
# partition 复杂度 O（n）
class Solution:
    """
    @param: nums: A list of integers
    @return: nothing
    """
    def wiggleSort(self, nums):
        if not nums:
            return []
        res=[]  
        mid = self.partition(nums, 0, len(nums)-1, (len(nums)+1)//2)  # 注意中位数，这里需要左右个数相等或差1， 这里的nums已经被排过了， 左边是小于mid的, 右边是大于mid的， 但不是完全排序
        for i in range(len(nums)):
            res.append(mid)
        if len(nums)%2 == 1:
            l, r = 0, len(nums)-2
            for i in range(len(nums)):
                if nums[i] > mid:
                    res[r] = nums[i]
                    r-=2
                elif nums[i] < mid:
                    res[l] = nums[i]
                    l+=2
        else:
            l, r = 1, len(nums)-2  # 保证 第一个必须是小的数。 不可以是 l, r = 0, len(nums)-1,这样的话 第一个就成了大的数
            for i in range(len(nums)):
                if nums[i] > mid:
                    res[l] = nums[i]
                    l+=2
                elif nums[i] < mid:
                    res[r] = nums[i]
                    r-=2
            
            
        for i in range(len(nums)):
            nums[i] = res[i]
        return nums
        
    def partition(self, nums, start, end, k):
        if start == end:
            return nums[start]
        i, j = start, end 
        pivot = nums[(start+end)//2]
        while i <= j:
            while i <= j and nums[i] < pivot:
                i += 1 
            while i <= j and nums[j] > pivot:
                j -= 1 
            if i <= j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1 
                j -= 1 
        if start + k - 1 <= j:
            return self.partition(nums, start, j, k)
        if start + k - 1 >= i:
            return self.partition(nums, i, end, k - (i - start))
        return nums[j+1]
