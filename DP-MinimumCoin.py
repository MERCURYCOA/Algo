# 用2元，5元，7元硬币拼出27元，硬币个数最少是多少？
# https://pan.baidu.com/play/video#/video?path=%2F%E4%B9%9D%E7%AB%A0%E7%AE%97%E6%B3%95%2F07-%E4%B9%9D%E7%AB%A0%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%2F%E4%B9%9D%E7%AB%A0%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%921_%E7%AE%80%E4%BB%8B.mov&t=-1



import sys

class Solution:

    # A = [2, 5, 7]   M = 27
    def minCoin(self, A, M):
    # create array to store f[X]
        f = [None]*(M+1)
        n = len(A)
        # initialization
        f[0] = 0
        #f[1], f[2],..., f[27]
        for i in range(1, M+1):
            # f[x] = min{f[x-2]+1, f[x-5]+1, f[x-7]+1}
            f[i] = sys.maxsize
            for j in range(n):
                if f[i-A[j]] != sys.maxsize and i >= A[j]:
                    f[i] = min(f[i - A[j]] + 1, f[i])
                #j=j+1
            #i=i+1
        if f[M] == sys.maxsize:
            f[M] = -1
        return f[M]

if __name__ == '__main__':
    solution = Solution()
    print(solution.minCoin([2,5,7], 27))
