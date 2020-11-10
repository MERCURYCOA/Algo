# Partition 

 # 1：第k大元素   

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
            while i <= j and nums[i] > pivot: # 不可以包含， 如果是等于pivot, 那么是可以用来交换的
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
# 2 非排序数组第k小元素
# 跟1一样，只是while循环中变成小于pivot的在前面，大于pivot的在后面
class Solution:
    """
    @param n: An integer
    @param nums: An array
    @return: the Kth largest element
    """
    def kthSmallestElement(self, n, nums):
        if nums == None:
            return -1  
            
        return self.quickSelect(nums, 0, len(nums)-1, n)
        
    def quickSelect(self, nums, start, end, n):
        if start == end:
            return nums[start]
        i, j = start, end 
        pivot = nums[(i+j)//2]
        while i <= j:
            while i <= j and nums[i] < pivot:  # 这里跟求第k大元素做法相反
                i += 1 
            while i <= j and nums[j] > pivot:
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



    
solution = Solution()
print(solution.kthSmallestElement(2, [2, 5, 3, 8, 1]))



# 2 交错正负数
class Solution:
    """
    @param: A: An integer array.
    @return: nothing
    """
    def rerange(self, A):
        if not A:
            return []
        A, pos = self.partition(A)
        n = len(A)
        if n > 2*pos:
            # -1,-2,-3, 1, 2, 3, 4
            i, j = 0, n-2
        if n == 2*pos:
            # -1, -2, -3, 1, 2, 3
            i, j = 0, n-1
        if n < 2*pos:
            # -1, -2, -3, 1, 2
            i, j = 1, n-1
        while i < j:
                if i < j and A[i] < 0 and A[j] > 0:
                    A[i], A[j] = A[j], A[i]
                    i += 2 
                    j -= 2
        return A
        
        
    def partition(self, A):
        pos = len(A)-1  # first positive
        i, j = 0, len(A) - 1
        while i < j:
            while i < j and A[i] < 0:
                i += 1 
            while i < j and A[j] > 0:
                j -= 1 
            if i < j:
                A[i], A[j] = A[j], A[i]
                i += 1 
                j -= 1 
        if A[j] > 0:
            pos = j
            return A, pos
        if A[i] > 0:
            pos = i 
            return A, pos
        return A, pos
# 3: sort colors 

# 三指针，分成3部分：2定1动。 left, i, right。 跟right交换之后，i不前进； 跟left交换，i和left一起前进；i==1， i前进
class Solution:
    """
    @param nums: A list of integer which is 0, 1 or 2 
    @return: nothing
    """
    def sortColors(self, nums):
        # write your code here
        if not nums:
            return []
            
        left, right = 0, len(nums)-1
        i = 0
        while i <= right:       #这里有等号，因为i停在right停的位置可能是0，1，还得进去循环判断一下
            if nums[i] == 0:
                nums[i],nums[left] = nums[left], nums[i]  
                left += 1 
                i += 1  #跟左边交换之后，i可以前进，因为左边的不是1，就是0， i走过的地方不可能有2
            elif nums[i] == 2:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1      # 这里right交换后i不前进，因为换过来的数可能还是2，所以只让right向左走，一直找到不是2的数，跟i交换，这样在下个循环i 可以前进  #这就是为什么i走过的地方不可能有2
            else:
                i += 1  #只有i前进，因为left要坚守0的真谛，i一旦发现0，就跟left换，left前进， 也就是left永远在左边第一个不是0 的数的位置
                
        return nums
       
# 第二次做：先把2与0，1分开，再把0，1分开

class Solution:
    """
    @param nums: A list of integer which is 0, 1 or 2 
    @return: nothing
    """
    def sortColors(self, nums):
        if not nums:
            return []
        i, j = 0, len(nums)-1 
        while i < j:
            while i < j and nums[i] < 2:
                i += 1 
            while i < j and nums[j] == 2:
                j -= 1 
            if i < j:
                nums[i], nums[j] = nums[j],  nums[i]
                i += 1 
                j -= 1 
        i = 0
        while i < j:
            while i < j and nums[i] == 0:
                i += 1 
            while i < j and nums[j] != 0:
                j -= 1 
            if i < j:
                nums[i], nums[j] = nums[j],  nums[i]
                i += 1 
                j -= 1 
        return nums
# 4: sort colorsII
# k种颜色分成k部分
# nlogk, 对颜色进行二分  [3,2,2,1,4], k=4, 找到颜色的中间是2，九把数组分成大于2和小于等于2的， [2,2,1] [3,4] 再对两个子数组递归
# 需要记录颜色，也就是k的开始和结束， index的开始和结束
# 不可以把k//2做参数传入，因为右边被分开不是从1到k//2， 而是从k//2+1到k， 所以k也要二分
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        # write your code here
        if not colors or len(colors) == 0:
            return []
            
        self.colorsPatition(colors, 0, len(colors)-1, 1, k)
        
    def colorsPatition(self, colors, index_left, index_right, color_start, color_end):
        if color_start == color_end:
            return
        if index_left >= index_right:
            return
        
        left, right = index_left, index_right  # 记下开始和结束，后面递归要用的
        
        color_mid =  (color_end + color_start)//2   #这里的color_mid就是（1+k）//2, 不是index，数字就代表颜色
        while left < right:                            #后面递归要用到停止的left和right的位置，作为下一步递归的开始或停止的地方，所以用left<= right,他们会交错，left停在较大值的第一个，right停在较小值的最后一个，
                                                        # 不用额外判断left=right的地方的值是大于还是小于color_mid. 如果用left<right,还有判断他们相遇的地方与color_mid的大小，然后才能判断下一层递归的起点或终点
            while left < right and colors[left] <= color_mid:
                left += 1 
                
            while left < right and colors[right] > color_mid:
                right -= 1 
            if left < right:
                colors[left], colors[right]= colors[right], colors[left]
                left += 1 
                right -= 1 
    
        self.colorsPatition(colors, index_left,right, color_start, color_mid)  # left和right最后的位置有2种  1  __right left__  2 指向同一位置
        self.colorsPatition(colors,left, index_right, color_mid+1, color_end)
