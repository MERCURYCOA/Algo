# 归并排序  时间复杂度 nlogn
#分治法
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
