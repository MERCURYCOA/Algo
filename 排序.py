# 归并排序  时间复杂度 nlogn
#分治法,类似二分法：A对半分成left, right, 分别对left和right再对半分，一直分到left = right，就是只有一个数，返回数本身，一左一右返回后得到两个数就是这一层的left和right，比较这两个数，得到sorted的
#的array， 这个array假设是这一层的left, 跟这一层的right array进行比较，返回得到到上一层的sorted array  ppt动画：https://www.bilibili.com/video/BV14E411o7Fb?from=search&seid=11676767501746059818

def sortIntegers2(self, A):
        # write your code here
        self.merge_sort(A)
        return A
def merge_sort(self, arr):
        if len(arr) <= 1:
            return arr
        left = arr[:len(arr)//2]  # 特殊：在选择区间进行分治
        right = arr[len(arr)//2:]

        self.merge_sort(left)   # 递归
        self.merge_sort(right)
        k = ileft = iright = 0
        ileft = iright = 0
        while ileft < len(left) and iright < len(right):
            if left[ileft] < right[iright]:
                arr[k] = left[ileft]   #输入为arr, 最终返回的是排好序的arr  #考虑：只有1个数时，返回本身， 两个数时，返回排好序的两个数
                ileft+=1
            else:
                arr[k] = right[iright]
                iright+=1
            k+=1
        while ileft < len(left): 
            arr[k] = left[ileft]
            ileft+=1
            k+=1
        while iright < len(right):
            arr[k] = right[iright]
            iright+=1
            k+=1

#快速排序
#思想： 找到一个p， 让p左边的数都小于p， p右边的数都大于p， 一般这个p挑选最中间的数
class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers2(self, A):
        # write your code here
        if len(A)<1:
            return A 
        self.quickSort(A, 0, len(A)-1)
        return A
    def quickSort(self, A, start, end):
        if start >= end:  # 当递归到一个数时， start = end, 直接return，意思是这个数不动
            return        
        left, right = start, end #简洁的写法
        p = A[start + (end-start)//2] #必须先把中间index的数赋给p， 不可以用p = start + (end - start)//2, 然后用A[p]做比较，因为最中间的数可能会变，必须将其固定
        while left <= right: # 这个必须有， 否则后面left <= right里面的无法循环，只执行一次
            while A[left] < p:
                left += 1 
            while A[right] > p:
                right -=1 
            if left <= right:
                A[left], A[right] = A[right], A[left]  #简洁的写法
                left += 1 
                right -= 1 
                
        self.quickSort(A, start, right)  # left <= right循环，这里left > right, 所以左边分治的区间是start, right
        self.quickSort(A, left, end)  # 同理
