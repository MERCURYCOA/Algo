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
