# f需要n+1个格子，初始化f[-1] =0, f里的i跟A里的i可以对齐，最后返回f[n-1],如果初始化f[0] = 1, f的i和A的i不对齐，即f[i]储存的是A[i-1]之前的总计数，最后返回f[n]
class Solution:
    def decode_path(self, S):
        A = [int(x) for x in list(S)]
        n = len(A)
        f = [0]*(n+1)
        
        for i in range(1,n+1):
            f[0] = 1
            if A[i-1] >= 1 and A[i-1] <= 9:
                f[i] += f[i-1]
            if i > 1:
                if A[i-2]==1 or A[i-2] ==2 and A[i-1]<=6:
                    f[i] += f[i-2]
        return f[n]

if __name__ == '__main__':
    solution = Solution()
    print(solution.decode_path("12"))
