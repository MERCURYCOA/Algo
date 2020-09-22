# 求最大连续子序列相乘得到最大乘积， i 到 j， 使得a[i]*a[i+1]*...*a[j]的最大值

# 迭代方法 - 时间复杂度 O(N^2)
import sys

class Solution:
    def max_product_subarray(self, A):
        n = len(A)
        max_product = -sys.maxsize-1
        for i in range(0,n-1):
            product = A[i]
            for j in range(i+1,n):
                product = product * A[j]
                max_product = max(max_product, product)
        return max_product

if __name__ == '__main__':
    solution = Solution()
    print(solution.max_product_subarray([2,3,-2,4]))
    
# 动态规划 时间复杂度 O(N)
# 关键：利用乘法的规律，前面所以数的乘积与当前数相乘，可能变大（正数），可能变小（负数），所以每一步要比较，当前数，当前数*之前的最大乘积，当前数*之前的最小乘积

import sys

class Solution:
    def max_product_subarray(self, A):
        n = len(A)
        f = [A[0]]*n
        g = [A[0]]*n
        res = -sys.maxsize-1

        if n == 0:
            return 0

        for i in range(1,n):
            f[i] = max(A[i], f[i-1]*A[i], g[i-1]*A[i])
            g[i] = min(A[i], f[i-1]*A[i], g[i-1]*A[i])
            res = max(res, f[i])
            print(A[i],f[i],g[i],res)
        return res

if __name__ == '__main__':
    solution = Solution()
    print(solution.max_product_subarray([2,3,5,6,-1,4,-6]))
